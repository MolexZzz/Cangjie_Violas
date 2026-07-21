"""Unified Milvus/Qdrant/Chroma benchmark driver launched by the Cangjie runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from precomputed_artifacts import DATASET_FILES, Folder, load_file


ROOT = Path(__file__).resolve().parents[1]
DATASET_IDS = {"1": "news20", "2": "ohsumed", "3": "yahoo", "4": "caltech", "5": "cub", "6": "coco"}


class ExternalVectorBackend(ABC):
    """Small common contract; database-specific IDs never escape this layer."""

    name: str

    @abstractmethod
    def reset(self, collection: str, dimension: int) -> None: ...

    @abstractmethod
    def upsert(self, records: list[dict]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]: ...

    def close(self) -> None:
        return None

    @property
    def config(self) -> dict:
        return {"backend": self.name}


class MockExactBackend(ExternalVectorBackend):
    name = "mock"

    def __init__(self) -> None:
        self.records: list[dict] = []

    def reset(self, collection: str, dimension: int) -> None:
        self.records = []

    def upsert(self, records: list[dict]) -> None:
        self.records.extend(records)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        query = np.asarray(vector, dtype=np.float32)
        scored = []
        for record in self.records:
            candidate = np.asarray(record["vector"], dtype=np.float32)
            score = float(np.dot(query, candidate))
            scored.append((record["recordId"], score))
        scored.sort(key=lambda row: (-row[1], row[0]))
        return scored[:limit]


class QdrantBackend(ExternalVectorBackend):
    name = "qdrant"

    def __init__(self, url: str) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("Qdrant backend requires: pip install qdrant-client") from exc
        self.models = models
        self.client = QdrantClient(url=url)
        self.url = url
        self.collection = ""
        self.id_map: dict[int, str] = {}

    def reset(self, collection: str, dimension: int) -> None:
        self.collection = collection
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=self.models.VectorParams(size=dimension, distance=self.models.Distance.COSINE),
        )

    def upsert(self, records: list[dict]) -> None:
        self.id_map = {index: record["recordId"] for index, record in enumerate(records)}
        batch_size = 256
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            points = [
                self.models.PointStruct(id=start + offset, vector=record["vector"], payload={"recordId": record["recordId"]})
                for offset, record in enumerate(batch)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        response = self.client.query_points(collection_name=self.collection, query=vector, limit=limit, with_payload=True)
        return [(point.payload["recordId"], float(point.score)) for point in response.points]

    @property
    def config(self) -> dict:
        return {"backend": self.name, "url": self.url, "metric": "cosine", "index": "engine-default HNSW"}


class MilvusBackend(ExternalVectorBackend):
    name = "milvus"

    def __init__(self, uri: str, token: str) -> None:
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("Milvus backend requires: pip install pymilvus") from exc
        self.DataType = DataType
        self.client = MilvusClient(uri=uri, token=token)
        self.uri = uri
        self.collection = ""

    def reset(self, collection: str, dimension: int) -> None:
        self.collection = collection
        if self.client.has_collection(collection_name=collection):
            self.client.drop_collection(collection_name=collection)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", datatype=self.DataType.INT64, is_primary=True)
        schema.add_field("recordId", datatype=self.DataType.VARCHAR, max_length=512)
        schema.add_field("vector", datatype=self.DataType.FLOAT_VECTOR, dim=dimension)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 80})
        self.client.create_collection(collection_name=collection, schema=schema, index_params=index_params)

    def upsert(self, records: list[dict]) -> None:
        batch_size = 256
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            rows = [{"id": start + offset, "recordId": record["recordId"], "vector": record["vector"]} for offset, record in enumerate(batch)]
            self.client.insert(collection_name=self.collection, data=rows)
        self.client.flush(collection_name=self.collection)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        rows = self.client.search(
            collection_name=self.collection,
            data=[vector],
            limit=limit,
            output_fields=["recordId"],
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit)}},
        )[0]
        return [(row["entity"]["recordId"], float(row["distance"])) for row in rows]

    @property
    def config(self) -> dict:
        return {"backend": self.name, "uri": self.uri, "metric": "cosine", "index": "HNSW", "M": 16, "efConstruction": 80}


class ChromaBackend(ExternalVectorBackend):
    name = "chroma"

    def __init__(self, host: str, port: int) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Chroma backend requires: pip install chromadb") from exc
        self.client = chromadb.HttpClient(host=host, port=port)
        self.host = host
        self.port = port
        self.collection = None

    def reset(self, collection: str, dimension: int) -> None:
        try:
            self.client.delete_collection(collection)
        except Exception:
            pass
        self.collection = self.client.create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, records: list[dict]) -> None:
        batch_size = 256
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            self.collection.add(
                ids=[record["recordId"] for record in batch],
                embeddings=[record["vector"] for record in batch],
            )

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        response = self.collection.query(query_embeddings=[vector], n_results=limit, include=["distances"])
        return [(item_id, 1.0 - float(distance)) for item_id, distance in zip(response["ids"][0], response["distances"][0])]

    @property
    def config(self) -> dict:
        return {"backend": self.name, "host": self.host, "port": self.port, "metric": "cosine", "index": "engine-default HNSW"}


def normalize(vector: list[float]) -> list[float]:
    row = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(row))
    return (row / norm if norm else row).tolist()


def cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


def prepare_fold(folders: list[Folder], max_per_folder: int, max_queries: int) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    records, queries, key_vectors = [], [], {}
    for folder in folders:
        vectors = folder.vectors[:max_per_folder] if max_per_folder else folder.vectors
        fold_size = len(vectors) // 5
        train_rep, test_rep = folder.fold_reps["simple"][0]
        key_vectors[folder.key] = normalize(train_rep)
        for index, vector in enumerate(vectors):
            item_id = f"{folder.name}/{index:08d}"
            if index < fold_size:
                queries.append({"queryId": f"fold-0/{item_id}", "vector": normalize(vector), "keyVector": normalize(test_rep)})
            else:
                records.append({"recordId": item_id, "key": folder.key, "vector": normalize(vector)})
    return records, queries[:max_queries], key_vectors


def exact_rank(records: list[dict], query: dict, key_vectors: dict[str, list[float]], beta: float) -> list[str]:
    scored = []
    for record in records:
        emb = cosine_distance(query["vector"], record["vector"])
        sem = cosine_distance(query["keyVector"], key_vectors[record["key"]])
        scored.append((beta * sem + (1.0 - beta) * emb, record["recordId"]))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [item_id for _, item_id in scored]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value, method="nearest")) if values else 0.0


def overlap(expected: list[str], actual: list[str], top_k: int) -> float:
    return len(set(expected[:top_k]) & set(actual[:top_k])) / float(top_k)


def ndcg(expected: list[str], actual: list[str], top_k: int) -> float:
    relevant = set(expected[:top_k])
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(actual[:top_k]) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(len(relevant)))
    return dcg / ideal if ideal else 0.0


def make_backend(args: argparse.Namespace) -> ExternalVectorBackend:
    if args.backend == "mock": return MockExactBackend()
    if args.backend == "qdrant": return QdrantBackend(args.qdrant_url)
    if args.backend == "milvus": return MilvusBackend(args.milvus_uri, args.milvus_token)
    if args.backend == "chroma": return ChromaBackend(args.chroma_host, args.chroma_port)
    raise ValueError(args.backend)


def git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mock", "milvus", "qdrant", "chroma"), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_IDS), required=True)
    parser.add_argument("--scale", choices=("smoke", "partial", "full"), default="smoke")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--candidate-multiplier", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN", "root:Milvus"))
    parser.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "127.0.0.1"))
    parser.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    args = parser.parse_args()
    if args.top_k <= 0 or args.candidate_multiplier < 1 or not 0.0 <= args.beta <= 1.0:
        parser.error("invalid top-k, candidate multiplier or beta")

    limits = {"smoke": (20, 20), "partial": (120, 200), "full": (0, 0)}
    max_per_folder, max_queries = limits[args.scale]
    dataset_name = DATASET_IDS[args.dataset]
    folders = load_file(ROOT / "dataset" / "precomputed" / DATASET_FILES[dataset_name])
    records, queries, key_vectors = prepare_fold(folders, max_per_folder, max_queries)
    backend = make_backend(args)
    collection = f"violas_{dataset_name}_{args.scale}".lower().replace("-", "_")
    build_start = time.perf_counter_ns()
    backend.reset(collection, len(records[0]["vector"]))
    backend.upsert(records)
    build_ms = (time.perf_counter_ns() - build_start) / 1_000_000.0

    raw_recalls, raw_ndcgs, mixed_recalls, mixed_ndcgs, db_ms, rerank_ms = [], [], [], [], [], []
    candidate_limit = min(len(records), args.top_k * args.candidate_multiplier)
    by_id = {record["recordId"]: record for record in records}
    for query in queries:
        exact_embedding = exact_rank(records, query, key_vectors, 0.0)
        exact_mixed = exact_rank(records, query, key_vectors, args.beta)
        search_start = time.perf_counter_ns()
        hits = backend.search(query["vector"], candidate_limit)
        db_ms.append((time.perf_counter_ns() - search_start) / 1_000_000.0)
        raw_ids = [item_id for item_id, _ in hits]
        raw_recalls.append(overlap(exact_embedding, raw_ids, args.top_k))
        raw_ndcgs.append(ndcg(exact_embedding, raw_ids, args.top_k))
        rerank_start = time.perf_counter_ns()
        rescored = []
        for item_id in raw_ids:
            record = by_id[item_id]
            emb = cosine_distance(query["vector"], record["vector"])
            sem = cosine_distance(query["keyVector"], key_vectors[record["key"]])
            rescored.append((args.beta * sem + (1.0 - args.beta) * emb, item_id))
        rescored.sort(key=lambda row: (row[0], row[1]))
        mixed_ids = [item_id for _, item_id in rescored]
        rerank_ms.append((time.perf_counter_ns() - rerank_start) / 1_000_000.0)
        mixed_recalls.append(overlap(exact_mixed, mixed_ids, args.top_k))
        mixed_ndcgs.append(ndcg(exact_mixed, mixed_ids, args.top_k))
    backend.close()

    result = {
        "schemaVersion": 1,
        "backend": backend.name,
        "dataset": dataset_name,
        "scale": {"fold": 0, "trainingVectors": len(records), "queries": len(queries), "topK": args.top_k},
        "config": {**backend.config, "beta": args.beta, "candidateMultiplier": args.candidate_multiplier},
        "buildMs": build_ms,
        "rawVector": {"recallAtK": mean(raw_recalls), "ndcgAtK": mean(raw_ndcgs)},
        "mixedRerank": {"recallAtK": mean(mixed_recalls), "ndcgAtK": mean(mixed_ndcgs)},
        "latencyMs": {
            "databaseMean": mean(db_ms), "databaseP50": percentile(db_ms, 50), "databaseP95": percentile(db_ms, 95),
            "rerankMean": mean(rerank_ms), "totalMean": mean([a + b for a, b in zip(db_ms, rerank_ms)]),
        },
        "provenance": {
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(), "gitCommit": git_commit(),
            "runner": "tools/external_db_benchmark.py", "os": platform.platform(), "numpyVersion": np.__version__,
        },
    }
    output = args.output or ROOT / "results" / "external" / f"{args.backend}-{dataset_name}-{args.scale}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EXTERNAL_DB_RESULT|" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
