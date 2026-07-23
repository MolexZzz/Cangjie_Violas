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
from importlib import metadata
from pathlib import Path

import numpy as np

from precomputed_artifacts import DATASET_FILES, Folder, load_file
from paper_artifact import load_artifact


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

    @abstractmethod
    def delete(self, record_ids: list[str]) -> None: ...

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
        replacements = {row["recordId"]: row for row in records}
        self.records = [row for row in self.records if row["recordId"] not in replacements]
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

    def delete(self, record_ids: list[str]) -> None:
        removed = set(record_ids)
        self.records = [row for row in self.records if row["recordId"] not in removed]


class QdrantBackend(ExternalVectorBackend):
    name = "qdrant"

    def __init__(self, url: str, *, local: bool = False) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("Qdrant backend requires: pip install qdrant-client") from exc
        self.models = models
        self.client = QdrantClient(":memory:") if local else QdrantClient(url=url)
        self.url = ":memory:" if local else url
        self.execution_mode = "in-process" if local else "service"
        try:
            self.server_version = str(self.client.info().version)
        except Exception:
            self.server_version = metadata.version("qdrant-client")
        self.collection = ""
        self.id_map: dict[int, str] = {}
        self.record_id_map: dict[str, int] = {}

    def reset(self, collection: str, dimension: int) -> None:
        self.collection = collection
        self.id_map = {}
        self.record_id_map = {}
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=self.models.VectorParams(size=dimension, distance=self.models.Distance.COSINE),
        )

    def upsert(self, records: list[dict]) -> None:
        batch_size = 256
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            point_ids = []
            for record in batch:
                if record["recordId"] not in self.record_id_map:
                    internal_id = len(self.record_id_map)
                    self.record_id_map[record["recordId"]] = internal_id
                    self.id_map[internal_id] = record["recordId"]
                point_ids.append(self.record_id_map[record["recordId"]])
            points = [
                self.models.PointStruct(id=point_ids[offset], vector=record["vector"],
                                        payload={"recordId": record["recordId"]})
                for offset, record in enumerate(batch)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        response = self.client.query_points(collection_name=self.collection, query=vector, limit=limit, with_payload=True)
        return [(point.payload["recordId"], float(point.score)) for point in response.points]

    def delete(self, record_ids: list[str]) -> None:
        point_ids = [self.record_id_map[item] for item in record_ids if item in self.record_id_map]
        if point_ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=self.models.PointIdsList(points=point_ids),
                wait=True,
            )

    @property
    def config(self) -> dict:
        return {"backend": self.name, "url": self.url, "executionMode": self.execution_mode,
                "serverVersion": self.server_version,
                "metric": "cosine", "index": "engine-default HNSW"}


class MilvusBackend(ExternalVectorBackend):
    name = "milvus"

    def __init__(self, uri: str, token: str, *, local_path: Path | None = None) -> None:
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("Milvus backend requires: pip install pymilvus") from exc
        self.DataType = DataType
        if local_path is not None:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.client = MilvusClient(uri=str(local_path))
            except Exception as exc:
                raise RuntimeError(
                    "paper-local Milvus requires Milvus Lite. Milvus Lite has no native Windows "
                    "runtime; run this mode under Linux/WSL with `pip install pymilvus[milvus_lite]`, "
                    "or use --execution-mode service for accuracy experiments whose latency is not "
                    "directly comparable to the paper."
                ) from exc
            self.uri = str(local_path)
            self.execution_mode = "in-process"
        else:
            self.client = MilvusClient(uri=uri, token=token)
            self.uri = uri
            self.execution_mode = "service"
        try:
            self.server_version = self.client.get_server_version()
        except Exception:
            self.server_version = metadata.version("pymilvus")
        self.collection = ""
        self.record_id_map: dict[str, int] = {}

    def reset(self, collection: str, dimension: int) -> None:
        self.collection = collection
        self.record_id_map = {}
        if self.client.has_collection(collection_name=collection):
            self.client.drop_collection(collection_name=collection)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", datatype=self.DataType.INT64, is_primary=True)
        schema.add_field("recordId", datatype=self.DataType.VARCHAR, max_length=512)
        schema.add_field("vector", datatype=self.DataType.FLOAT_VECTOR, dim=dimension)
        index_params = self.client.prepare_index_params()
        # Match the Python image benchmark's Milvus configuration.
        index_params.add_index(
            field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 1024}
        )
        self.client.create_collection(collection_name=collection, schema=schema, index_params=index_params)

    def upsert(self, records: list[dict]) -> None:
        batch_size = 256
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            rows = []
            for record in batch:
                if record["recordId"] not in self.record_id_map:
                    self.record_id_map[record["recordId"]] = len(self.record_id_map)
                rows.append({"id": self.record_id_map[record["recordId"]],
                             "recordId": record["recordId"], "vector": record["vector"]})
            self.client.upsert(collection_name=self.collection, data=rows)
        self.client.flush(collection_name=self.collection)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        rows = self.client.search(
            collection_name=self.collection,
            data=[vector],
            limit=limit,
            output_fields=["recordId"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )[0]
        return [(row["entity"]["recordId"], float(row["distance"])) for row in rows]

    def delete(self, record_ids: list[str]) -> None:
        ids = [self.record_id_map[item] for item in record_ids if item in self.record_id_map]
        if ids:
            self.client.delete(collection_name=self.collection, ids=ids)
            self.client.flush(collection_name=self.collection)

    @property
    def config(self) -> dict:
        return {"backend": self.name, "uri": self.uri, "executionMode": self.execution_mode,
                "serverVersion": self.server_version,
                "metric": "cosine", "index": "IVF_FLAT", "nlist": 1024, "nprobe": 10}


class ChromaBackend(ExternalVectorBackend):
    name = "chroma"

    def __init__(self, host: str, port: int, *, local: bool = False) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Chroma backend requires: pip install chromadb") from exc
        self.client = chromadb.EphemeralClient() if local else chromadb.HttpClient(host=host, port=port)
        self.host = "in-process" if local else host
        self.port = 0 if local else port
        self.execution_mode = "in-process" if local else "service"
        self.server_version = self.client.get_version()
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
            self.collection.upsert(
                ids=[record["recordId"] for record in batch],
                embeddings=[record["vector"] for record in batch],
            )

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        response = self.collection.query(query_embeddings=[vector], n_results=limit, include=["distances"])
        return [(item_id, 1.0 - float(distance)) for item_id, distance in zip(response["ids"][0], response["distances"][0])]

    def delete(self, record_ids: list[str]) -> None:
        if record_ids:
            self.collection.delete(ids=record_ids)

    @property
    def config(self) -> dict:
        return {"backend": self.name, "host": self.host, "port": self.port,
                "executionMode": self.execution_mode,
                "serverVersion": self.server_version, "metric": "cosine", "index": "engine-default HNSW"}


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


def key_hit_rate(expected: list[str], actual: list[str], by_id: dict[str, dict], top_k: int) -> float:
    relevant = {by_id[item_id]["key"] for item_id in expected[:top_k]}
    ranked = [by_id[item_id]["key"] for item_id in actual[:top_k]]
    return sum(1 for key in ranked if key in relevant) / float(top_k)


def key_ndcg(expected: list[str], actual: list[str], by_id: dict[str, dict], top_k: int) -> float:
    relevant = {by_id[item_id]["key"] for item_id in expected[:top_k]}
    ranked = [by_id[item_id]["key"] for item_id in actual[:top_k]]
    seen: set[str] = set()
    dcg = 0.0
    for rank, key in enumerate(ranked):
        if key in relevant and key not in seen:
            dcg += 1.0 / math.log2(rank + 2)
            seen.add(key)
    ideal_count = min(top_k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def mixed_relevance(record: dict, query: dict, key_vectors: dict[str, list[float]], beta: float) -> float:
    """Equation 5 similarity score used as the graded gain in Equation 14."""
    embedding_similarity = 1.0 - cosine_distance(query["vector"], record["vector"])
    entity_similarity = 1.0 - cosine_distance(query["keyVector"], key_vectors[record["key"]])
    return min(1.0, max(0.0, beta * entity_similarity + (1.0 - beta) * embedding_similarity))


def graded_mixed_ndcg(ideal_ids: list[str], actual_ids: list[str], by_id: dict[str, dict],
                      query: dict, key_vectors: dict[str, list[float]], beta: float, top_k: int) -> float:
    """Paper Equation 14; unlike the old helper, relevance is not binary."""
    def dcg(ids: list[str]) -> float:
        total = 0.0
        for rank, item_id in enumerate(ids[:top_k]):
            score = mixed_relevance(by_id[item_id], query, key_vectors, beta)
            total += (2.0 ** score - 1.0) / math.log2(rank + 2)
        return total

    ideal = dcg(ideal_ids)
    return dcg(actual_ids) / ideal if ideal else 0.0


def make_backend(args: argparse.Namespace) -> ExternalVectorBackend:
    if args.backend == "mock": return MockExactBackend()
    local = args.execution_mode == "paper-local"
    if args.backend == "qdrant": return QdrantBackend(args.qdrant_url, local=local)
    if args.backend == "milvus": return MilvusBackend(
        args.milvus_uri, args.milvus_token,
        local_path=(args.local_state_dir / "milvus-lite.db") if local else None,
    )
    if args.backend == "chroma": return ChromaBackend(args.chroma_host, args.chroma_port, local=local)
    raise ValueError(args.backend)


def git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def parse_betas(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",")]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("betas must be a comma-separated list in [0,1]")
    return values


def show_progress(label: str, completed: int, total: int, width: int = 30) -> None:
    """Dependency-free tqdm-style progress for terminal and persisted transcripts."""
    ratio = completed / total if total else 1.0
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label:<22} [{bar}] {completed:>4}/{total:<4} {ratio * 100:6.2f}%", end="", flush=True)
    if completed >= total:
        print(flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mock", "milvus", "qdrant", "chroma"), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_IDS))
    parser.add_argument("--artifact", type=Path, help="python-paper-90-10 artifact directory; disables local splitting")
    parser.add_argument("--scale", choices=("smoke", "partial", "full"), default="smoke")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--betas", type=parse_betas,
                        help="run several mixed weights after one database build, for example 0.0,0.1,...,1.0")
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--execution-mode", choices=("paper-local", "service"), default="paper-local",
                        help="paper-local matches the paper's in-memory/in-process boundary; service uses Docker/HTTP")
    parser.add_argument("--local-state-dir", type=Path,
                        default=ROOT / "results" / ".paper-local-state")
    parser.add_argument("--max-queries", type=int,
                        help="override the scale query limit; useful for matching Cangjie smoke runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN", "root:Milvus"))
    parser.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "127.0.0.1"))
    parser.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    args = parser.parse_args()
    betas = args.betas or [args.beta]
    if (args.top_k <= 0 or args.candidate_multiplier < 1
            or (args.max_queries is not None and args.max_queries < 0)
            or any(not 0.0 <= beta <= 1.0 for beta in betas)):
        parser.error("invalid top-k, candidate multiplier or beta")
    if not args.artifact and not args.dataset:
        parser.error("--dataset is required unless --artifact is supplied")

    limits = {"smoke": (20, 20), "partial": (120, 200), "full": (0, 0)}
    max_per_folder, max_queries = limits[args.scale]
    artifact_ground_truth = None
    if args.artifact:
        artifact_manifest, records, queries, key_vectors, artifact_ground_truth = load_artifact(args.artifact)
        if args.top_k > artifact_manifest["topK"]:
            parser.error(f"artifact only contains ground truth through topK={artifact_manifest['topK']}")
        dataset_name = artifact_manifest["dataset"]
        records = [{**row, "vector": normalize(row["vector"])} for row in records]
        queries = [{**row, "vector": normalize(row["vector"]), "keyVector": normalize(row["keyVector"])} for row in queries]
        key_vectors = {key: normalize(vector) for key, vector in key_vectors.items()}
        query_limit = args.max_queries if args.max_queries is not None else max_queries
        if query_limit:
            queries = queries[:query_limit]
        query_scope = ("full-10-percent-test-pool"
                       if len(queries) == artifact_manifest["counts"]["queryPool"]
                       else "debug-subset")
        split_config = {"splitProtocol": artifact_manifest["protocol"],
                        "artifactStatus": artifact_manifest["artifactStatus"],
                        "artifact": str(args.artifact)}
        scale_protocol = {"protocol": artifact_manifest["protocol"]}
    else:
        dataset_name = DATASET_IDS[args.dataset]
        folders = load_file(ROOT / "dataset" / "precomputed" / DATASET_FILES[dataset_name])
        query_limit = args.max_queries if args.max_queries is not None else max_queries
        records, queries, key_vectors = prepare_fold(folders, max_per_folder, query_limit)
        query_scope = "debug-or-five-fold"
        split_config = {"splitProtocol": "precomputed-five-fold", "fold": 0,
                        "keyVectorSource": "fold-train-representative"}
        scale_protocol = {"protocol": "precomputed-five-fold", "fold": 0}
    backend = make_backend(args)
    collection = f"violas_{dataset_name}_{args.scale}".lower().replace("-", "_")
    build_start = time.perf_counter_ns()
    backend.reset(collection, len(records[0]["vector"]))
    backend.upsert(records)
    build_ms = (time.perf_counter_ns() - build_start) / 1_000_000.0

    beta_stats = {
        beta: {
            "rawRecalls": [], "rawNdcgs": [], "rawDbMs": [],
            "mixedRecalls": [], "mixedNdcgs": [], "candidateDbMs": [], "rerankMs": [],
        }
        for beta in betas
    }
    candidate_limit = min(len(records), args.top_k * args.candidate_multiplier)
    by_id = {record["recordId"]: record for record in records}
    progress_total = len(queries) * len(betas)
    progress_completed = 0
    show_progress(f"{backend.name}/{dataset_name}", 0, progress_total)
    for query in queries:
        for beta in betas:
            if artifact_ground_truth is None:
                exact_mixed = exact_rank(records, query, key_vectors, beta)
            else:
                try:
                    exact_mixed = artifact_ground_truth[(query["queryId"], float(beta))]
                except KeyError as error:
                    raise ValueError(f"artifact does not contain requested query/beta ground truth: {error}") from error

            # Match caltech_bench.py: both the raw Top-K call and the expanded
            # candidate call are executed and timed independently for every beta.
            raw_start = time.perf_counter_ns()
            raw_hits = backend.search(query["vector"], args.top_k)
            beta_stats[beta]["rawDbMs"].append((time.perf_counter_ns() - raw_start) / 1_000_000.0)
            raw_ids = [item_id for item_id, _ in raw_hits]

            search_start = time.perf_counter_ns()
            hits = backend.search(query["vector"], candidate_limit)
            beta_stats[beta]["candidateDbMs"].append(
                (time.perf_counter_ns() - search_start) / 1_000_000.0
            )
            candidate_ids = [item_id for item_id, _ in hits]
            rerank_start = time.perf_counter_ns()
            rescored = []
            for item_id in candidate_ids:
                record = by_id[item_id]
                emb = cosine_distance(query["vector"], record["vector"])
                sem = cosine_distance(query["keyVector"], key_vectors[record["key"]])
                rescored.append((beta * sem + (1.0 - beta) * emb, item_id))
            rescored.sort(key=lambda row: (row[0], row[1]))
            mixed_ids = [item_id for _, item_id in rescored]
            beta_stats[beta]["rerankMs"].append((time.perf_counter_ns() - rerank_start) / 1_000_000.0)
            # Match the Python image benchmark exactly: every method is judged
            # against mixed ground truth for the current beta. At beta=1 IDs
            # are tied within a semantic key, so the paper code uses key recall.
            if beta > 0.999999:
                beta_stats[beta]["rawRecalls"].append(key_hit_rate(exact_mixed, raw_ids, by_id, args.top_k))
                beta_stats[beta]["mixedRecalls"].append(key_hit_rate(exact_mixed, mixed_ids, by_id, args.top_k))
            else:
                beta_stats[beta]["rawRecalls"].append(overlap(exact_mixed, raw_ids, args.top_k))
                beta_stats[beta]["mixedRecalls"].append(overlap(exact_mixed, mixed_ids, args.top_k))
            beta_stats[beta]["rawNdcgs"].append(
                graded_mixed_ndcg(exact_mixed, raw_ids, by_id, query, key_vectors, beta, args.top_k)
            )
            beta_stats[beta]["mixedNdcgs"].append(
                graded_mixed_ndcg(exact_mixed, mixed_ids, by_id, query, key_vectors, beta, args.top_k)
            )
            progress_completed += 1
            show_progress(f"{backend.name}/{dataset_name}", progress_completed, progress_total)
    backend.close()

    runs = []
    for beta in betas:
        raw_db_ms = beta_stats[beta]["rawDbMs"]
        candidate_db_ms = beta_stats[beta]["candidateDbMs"]
        rerank_ms = beta_stats[beta]["rerankMs"]
        raw_recall = mean(beta_stats[beta]["rawRecalls"])
        raw_ndcg = mean(beta_stats[beta]["rawNdcgs"])
        mixed_recall = mean(beta_stats[beta]["mixedRecalls"])
        mixed_ndcg = mean(beta_stats[beta]["mixedNdcgs"])
        candidate_mean = mean(candidate_db_ms)
        rerank_mean = mean(rerank_ms)
        total_mean = mean([a + b for a, b in zip(candidate_db_ms, rerank_ms)])
        runs.append({
            "beta": beta,
            "evaluationProtocol": "violas-paper-table2-v5",
            "queryScope": query_scope,
            "queries": len(queries),
            "rawVector": {
                "recallAtK": raw_recall,
                "ndcgAtK": raw_ndcg,
            },
            "rawLatencyMs": {
                "databaseMean": mean(raw_db_ms),
                "databaseP50": percentile(raw_db_ms, 50),
                "databaseP95": percentile(raw_db_ms, 95),
            },
            "mixedRerank": {
                "recallAtK": mixed_recall,
                "ndcgAtK": mixed_ndcg,
            },
            "latencyMs": {
                "databaseMean": candidate_mean,
                "databaseP50": percentile(candidate_db_ms, 50),
                "databaseP95": percentile(candidate_db_ms, 95),
                "rerankMean": rerank_mean,
                "totalMean": total_mean,
            },
            # Paper Table 2 baseline: the external database ranks instance
            # embeddings only and directly returns vector Top-K.
            "paperComparison": {
                "method": "direct-vector-top-k",
                "candidateK": args.top_k,
                "recallAtK": raw_recall,
                "ndcgAtK": raw_ndcg,
                "latencyMs": {
                    "databaseMean": mean(raw_db_ms),
                },
            },
            # Enhanced auxiliary method: vector candidate retrieval followed
            # by local mixed-score reranking.
            "mixedComparison": {
                "method": "vector-candidate-plus-local-mixed-rerank",
                "candidateK": candidate_limit,
                "recallAtK": mixed_recall,
                "ndcgAtK": mixed_ndcg,
                "latencyMs": {
                    "candidateDatabaseMean": candidate_mean,
                    "rerankMean": rerank_mean,
                    "totalMean": total_mean,
                },
            },
            # Backward-compatible alias retained for existing readers.
            "rawComparison": {
                "method": "direct-vector-top-k",
                "candidateK": args.top_k,
                "recallAtK": raw_recall,
                "ndcgAtK": raw_ndcg,
                "latencyMs": {
                    "databaseMean": mean(raw_db_ms),
                },
            },
        })

    result = {
        "schemaVersion": 3,
        "backend": backend.name,
        "dataset": dataset_name,
        "scale": {**scale_protocol, "trainingVectors": len(records), "queries": len(queries), "topK": args.top_k},
        "config": {
            **backend.config,
            **split_config,
            "evaluationProtocol": "violas-paper-table2-v5",
            "queryScope": query_scope,
            "paperComparisonMethod": "direct-vector-top-k",
            "auxiliaryMixedComparisonMethod": "vector-candidate-plus-local-mixed-rerank",
            "ndcgGain": "mixed-score-graded",
            "betas": betas,
            "candidateMultiplier": args.candidate_multiplier,
        },
        "buildMs": build_ms,
        "runs": runs,
        "provenance": {
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(), "gitCommit": git_commit(),
            "runner": "tools/external_db_benchmark.py", "os": platform.platform(), "numpyVersion": np.__version__,
            "clientVersions": {
                "qdrant-client": metadata.version("qdrant-client"),
                "pymilvus": metadata.version("pymilvus"),
                "chromadb": metadata.version("chromadb"),
            },
        },
    }
    if len(runs) == 1:
        result["config"]["beta"] = runs[0]["beta"]
        result["rawVector"] = runs[0]["rawVector"]
        result["rawLatencyMs"] = runs[0]["rawLatencyMs"]
        result["mixedRerank"] = runs[0]["mixedRerank"]
        result["latencyMs"] = runs[0]["latencyMs"]
    output = args.output or ROOT / "results" / "external" / f"{args.backend}-{dataset_name}-{args.scale}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EXTERNAL_DB_RESULT|" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
