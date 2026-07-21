"""Run reproducible Faiss Flat/IVF/HNSW cosine baselines on precomputed artifacts."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from precomputed_artifacts import DATASET_FILES, Folder, load_file


ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), p, method="nearest"))


def ndcg_at_k(ground_truth: list[int], result: list[int], top_k: int) -> float:
    relevant = set(ground_truth[:top_k])
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(result[:top_k]) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(top_k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def normalized_matrix(rows: list[list[float]]) -> np.ndarray:
    matrix = np.ascontiguousarray(np.asarray(rows, dtype=np.float32))
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("non-empty 2-D vectors are required")
    faiss.normalize_L2(matrix)
    return matrix


def fold_zero(folders: list[Folder], max_vectors_per_folder: int, max_queries: int) -> tuple[np.ndarray, np.ndarray]:
    training: list[list[float]] = []
    queries: list[list[float]] = []
    for folder in folders:
        vectors = folder.vectors[:max_vectors_per_folder] if max_vectors_per_folder else folder.vectors
        fold_size = len(vectors) // 5
        queries.extend(vectors[:fold_size])
        training.extend(vectors[fold_size:])
    if max_queries:
        queries = queries[:max_queries]
    if not training or not queries:
        raise ValueError("fold produced empty training or query vectors")
    return normalized_matrix(training), normalized_matrix(queries)


def ground_truth(training: np.ndarray, queries: np.ndarray, top_k: int) -> list[list[int]]:
    rows = []
    for query in queries:
        similarities = training @ query
        # stable sort makes equal-score behavior reproducible by original row id.
        order = np.argsort(-similarities, kind="stable")[:top_k]
        rows.append([int(item) for item in order])
    return rows


def build_index(method: str, training: np.ndarray, args: argparse.Namespace) -> tuple[faiss.Index, dict]:
    dimension = training.shape[1]
    count = training.shape[0]
    if method == "exact":
        index = faiss.IndexFlatIP(dimension)
        config = {"indexType": "IndexFlatIP", "metric": "cosine/IP"}
    elif method == "ivf":
        safe_nlist = max(1, min(args.nlist, int(math.sqrt(count)), max(1, count // 39)))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, safe_nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(args.nprobe, safe_nlist)
        config = {
            "indexType": "IndexIVFFlat",
            "metric": "cosine/IP",
            "nlist": safe_nlist,
            "nprobe": index.nprobe,
        }
    elif method == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction
        index.hnsw.efSearch = args.ef_search
        config = {
            "indexType": "IndexHNSWFlat",
            "metric": "cosine/IP",
            "m": args.hnsw_m,
            "efConstruction": args.ef_construction,
            "efSearch": args.ef_search,
        }
    else:
        raise ValueError(method)
    if not index.is_trained:
        index.train(training)
    index.add(training)
    return index, config


def evaluate(method: str, training: np.ndarray, queries: np.ndarray, gt: list[list[int]], args: argparse.Namespace) -> dict:
    build_start = time.perf_counter_ns()
    index, config = build_index(method, training, args)
    build_ms = (time.perf_counter_ns() - build_start) / 1_000_000.0
    recalls: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    for query, expected in zip(queries, gt):
        start = time.perf_counter_ns()
        _, ids = index.search(np.ascontiguousarray(query.reshape(1, -1)), args.top_k)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
        actual = [int(item) for item in ids[0] if item >= 0]
        recalls.append(len(set(expected) & set(actual)) / float(args.top_k))
        ndcgs.append(ndcg_at_k(expected, actual, args.top_k))
    serialized_size = len(faiss.serialize_index(index))
    return {
        "method": method,
        "config": config,
        "buildMs": build_ms,
        "indexBytes": serialized_size,
        "recallAtK": float(np.mean(recalls)),
        "ndcgAtK": float(np.mean(ndcgs)),
        "latencyMs": {
            "mean": float(np.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="news20")
    parser.add_argument("--max-vectors-per-folder", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.max_queries, args.top_k, args.nlist, args.nprobe, args.hnsw_m, args.ef_construction, args.ef_search) <= 0:
        parser.error("query/index parameters must be positive")

    source = ROOT / "dataset" / "precomputed" / DATASET_FILES[args.dataset]
    folders = load_file(source)
    training, queries = fold_zero(folders, args.max_vectors_per_folder, args.max_queries)
    gt = ground_truth(training, queries, args.top_k)
    result = {
        "schemaVersion": 1,
        "backend": "faiss-cpu",
        "scope": "single-vector cosine baseline; no Violas semantic mixed score",
        "dataset": args.dataset,
        "source": source.relative_to(ROOT).as_posix(),
        "scale": {
            "fold": 0,
            "folds": 5,
            "trainingVectors": int(training.shape[0]),
            "queries": int(queries.shape[0]),
            "dimension": int(training.shape[1]),
            "maxVectorsPerFolder": args.max_vectors_per_folder,
            "topK": args.top_k,
            "dtype": "float32",
        },
        "groundTruth": "stable NumPy brute-force cosine/IP",
        "results": [evaluate(method, training, queries, gt, args) for method in ("exact", "ivf", "hnsw")],
        "provenance": {
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "gitCommit": git_commit(),
            "runner": "tools/run_faiss_baseline.py",
            "faissVersion": getattr(faiss, "__version__", "unknown"),
            "numpyVersion": np.__version__,
            "os": platform.platform(),
            "machine": platform.machine(),
        },
    }
    output = args.output or ROOT / "results" / "faiss" / f"{args.dataset}-sample.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({row["method"]: {key: row[key] for key in ("recallAtK", "ndcgAtK", "buildMs", "indexBytes", "latencyMs")} for row in result["results"]}, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
