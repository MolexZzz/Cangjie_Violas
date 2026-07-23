"""Run reproducible Faiss Flat/IVF/HNSW cosine baselines on precomputed artifacts."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from precomputed_artifacts import DATASET_FILES, Folder, load_file
from paper_artifact import load_artifact


ROOT = Path(__file__).resolve().parents[1]


class PeakRssSampler:
    """Sample process RSS, including native Faiss allocations, when psutil exists."""

    def __init__(self) -> None:
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss = None
        try:
            import psutil

            process = psutil.Process()
            self._rss = lambda: int(process.memory_info().rss)
        except ImportError:
            if os.name == "nt":
                class ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                def windows_rss() -> int:
                    counters = ProcessMemoryCounters()
                    counters.cb = ctypes.sizeof(counters)
                    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
                    get_current_process.restype = ctypes.c_void_p
                    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
                    get_memory_info.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ProcessMemoryCounters),
                        ctypes.c_ulong,
                    ]
                    get_memory_info.restype = ctypes.c_int
                    ok = get_memory_info(
                        get_current_process(),
                        ctypes.byref(counters),
                        counters.cb,
                    )
                    if not ok:
                        raise OSError("GetProcessMemoryInfo failed")
                    return int(counters.WorkingSetSize)

                self._rss = windows_rss

    def __enter__(self) -> "PeakRssSampler":
        if self._rss is None:
            return self
        self.peak = self._rss()

        def sample() -> None:
            while not self._stop.wait(0.005):
                self.peak = max(self.peak or 0, self._rss())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._rss is None:
            return
        self.peak = max(self.peak or 0, self._rss())
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


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
    memory = PeakRssSampler()
    memory.__enter__()
    try:
        build_start = time.perf_counter_ns()
        index, config = build_index(method, training, args)
        build_ms = (time.perf_counter_ns() - build_start) / 1_000_000.0
        warmup_count = min(args.warmup_queries, len(queries))
        if warmup_count:
            index.search(np.ascontiguousarray(queries[:warmup_count]), args.top_k)
        recalls: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []
        for repeat in range(args.repeats):
            for query_index, (query, expected) in enumerate(zip(queries, gt)):
                start = time.perf_counter_ns()
                _, ids = index.search(np.ascontiguousarray(query.reshape(1, -1)), args.top_k)
                latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
                if repeat == 0:
                    actual = [int(item) for item in ids[0] if item >= 0]
                    recalls.append(len(set(expected) & set(actual)) / float(args.top_k))
                    ndcgs.append(ndcg_at_k(expected, actual, args.top_k))
                if args.progress_every and (query_index + 1) % args.progress_every == 0:
                    print(f"FAISS_PROGRESS|{method}|repeat={repeat + 1}/{args.repeats}|"
                          f"queries={query_index + 1}/{len(queries)}", flush=True)
        serialized_size = len(faiss.serialize_index(index))
    finally:
        memory.__exit__(None, None, None)
    peak_rss = memory.peak
    return {
        "method": method,
        "config": config,
        "buildMs": build_ms,
        "indexBytes": serialized_size,
        "peakRssBytes": peak_rss,
        "recallAtK": float(np.mean(recalls)),
        "ndcgAtK": float(np.mean(ndcgs)),
        "latencyMs": {
            "mean": float(np.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
    }


def write_markdown(result: dict, output: Path) -> None:
    lines = [
        f"# Faiss vector baseline: {result['dataset']}",
        "",
        "该表只比较单向量 cosine Top-K，不包含 Violas 的类别语义距离或 mixed score。",
        "",
        f"- training: {result['scale']['trainingVectors']}",
        f"- queries: {result['scale']['queries']}",
        f"- dimension: {result['scale']['dimension']}",
        f"- topK: {result['scale']['topK']}",
        f"- warmup queries: {result['measurement']['warmupQueries']}",
        f"- repetitions: {result['measurement']['repetitions']}",
        "",
        "| Index | Recall@K | NDCG@K | Build (ms) | Mean (ms/query) | P50 | P95 | Index size (MiB) | Peak RSS (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["results"]:
        rss = "N/A" if row["peakRssBytes"] is None else f"{row['peakRssBytes'] / 1048576.0:.2f}"
        lines.append(
            f"| {row['config']['indexType']} | {row['recallAtK']:.6f} | {row['ndcgAtK']:.6f} | "
            f"{row['buildMs']:.3f} | {row['latencyMs']['mean']:.4f} | "
            f"{row['latencyMs']['p50']:.4f} | {row['latencyMs']['p95']:.4f} | "
            f"{row['indexBytes'] / 1048576.0:.2f} | {rss} |"
        )
    lines.extend([
        "",
        "说明：Peak RSS 是整个 Python 进程的采样峰值，并非索引独占内存；索引独占磁盘/序列化大小见 Index size。",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="news20")
    parser.add_argument("--artifact", type=Path, help="python-paper-90-10 artifact directory; disables local splitting")
    parser.add_argument("--max-vectors-per-folder", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, default=32)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if min(args.top_k, args.nlist, args.nprobe, args.hnsw_m, args.ef_construction,
           args.ef_search, args.repeats) <= 0 or min(args.max_queries, args.warmup_queries,
                                                    args.progress_every) < 0:
        parser.error("index parameters/repeats must be positive; limits must be non-negative")

    if args.artifact:
        manifest, artifact_records, artifact_queries, _, artifact_gt = load_artifact(args.artifact)
        if args.top_k > manifest["topK"]:
            parser.error(f"artifact only contains ground truth through topK={manifest['topK']}")
        if args.max_queries:
            artifact_queries = artifact_queries[:args.max_queries]
        training = normalized_matrix([row["vector"] for row in artifact_records])
        queries = normalized_matrix([row["vector"] for row in artifact_queries])
        positions = {row["recordId"]: index for index, row in enumerate(artifact_records)}
        gt = [[positions[item_id] for item_id in artifact_gt[(row["queryId"], 0.0)]] for row in artifact_queries]
        source = args.artifact / "manifest.json"
        dataset_name = manifest["dataset"]
        split_config = {"protocol": manifest["protocol"], "artifactStatus": manifest["artifactStatus"]}
        source_limits = {}
    else:
        source = ROOT / "dataset" / "precomputed" / DATASET_FILES[args.dataset]
        folders = load_file(source)
        training, queries = fold_zero(folders, args.max_vectors_per_folder, args.max_queries)
        gt = ground_truth(training, queries, args.top_k)
        dataset_name = args.dataset
        split_config = {"protocol": "precomputed-five-fold", "fold": 0, "folds": 5}
        source_limits = {"maxVectorsPerFolder": args.max_vectors_per_folder}
    result = {
        "schemaVersion": 1,
        "backend": "faiss-cpu",
        "scope": "single-vector cosine baseline; no Violas semantic mixed score",
        "dataset": dataset_name,
        "source": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
        "scale": {
            "trainingVectors": int(training.shape[0]),
            "queries": int(queries.shape[0]),
            "dimension": int(training.shape[1]),
            "topK": args.top_k,
            "dtype": "float32",
            **split_config,
            **source_limits,
        },
        "groundTruth": "stable NumPy brute-force cosine/IP",
        "measurement": {
            "warmupQueries": min(args.warmup_queries, len(queries)),
            "repetitions": args.repeats,
            "latencyBoundary": "one Faiss index.search call for one query",
            "peakRss": "5 ms process-RSS sampling; null when psutil is unavailable",
        },
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
    output = args.output or ROOT / "results" / "faiss" / f"{dataset_name}-sample.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = args.markdown or output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(result, markdown)
    print(json.dumps({row["method"]: {key: row[key] for key in ("recallAtK", "ndcgAtK", "buildMs", "indexBytes", "latencyMs")} for row in result["results"]}, indent=2))
    print(f"wrote {output}")
    print(f"wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
