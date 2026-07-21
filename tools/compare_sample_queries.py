"""Compare frozen Python and Cangjie retrieval query-by-query on sample artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.dont_write_bytecode = True

import numpy as np

from precomputed_artifacts import DATASET_FILES, Folder, load_file


ROOT = Path(__file__).resolve().parents[1]
CJ_CORE = ROOT / "cj_core"
PYTHON_REFERENCE_ROOT = ROOT / "violas_python"
sys.path.insert(0, str(PYTHON_REFERENCE_ROOT))

from violas.storage.vectorgroup import VectorGroup  # noqa: E402
from violas.storage.vectormap import VectorMap  # noqa: E402


DATASETS = {
    "1": ("news20", "text"),
    "2": ("ohsumed", "text"),
    "3": ("yahoo", "text"),
    "4": ("caltech", "image"),
    "5": ("cub", "image"),
    "6": ("coco", "image"),
}
METHODS = ("exact", "representative", "mixed", "hdmg")


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1e-9 or right_norm < 1e-9:
        return 1.0
    return 1.0 - float(np.dot(left, right) / (left_norm * right_norm))


def record_id(folder: str, index: int) -> str:
    return f"{folder}/{index:08d}"


def stable_id(key: str, item_id: str) -> str:
    return f"{key}::{item_id}"


def result_ids(results: list, top_k: int) -> list[str]:
    output = []
    for result in results[:top_k]:
        description = result.group.descriptions[result.vector_idx].get("text", "")
        base_key = result.key.rsplit("-", 1)[0] if result.key.rsplit("-", 1)[-1].isdigit() else result.key
        output.append(stable_id(base_key, description))
    return output


def build_python_state(folders: list[Folder], vector_type: str) -> tuple[VectorMap, list, list]:
    vector_map = VectorMap()
    key_vectors: dict[str, np.ndarray] = {}
    flat: list[tuple[str, np.ndarray]] = []
    queries: list[tuple[str, np.ndarray, np.ndarray]] = []
    for folder in folders:
        vectors = [np.asarray(vector, dtype=np.float64) for vector in folder.vectors[:20]]
        fold_size = len(vectors) // 5
        test_end = fold_size
        train_vectors = vectors[test_end:]
        train_ids = [record_id(folder.name, index) for index in range(test_end, len(vectors))]
        train_rep, test_rep = folder.fold_reps["simple"][0]
        train_rep_array = np.asarray(train_rep, dtype=np.float64)
        test_rep_array = np.asarray(test_rep, dtype=np.float64)
        source_group = VectorGroup(
            group_name=folder.name,
            representative=train_rep_array,
            rep_description="simple",
            vectors=train_vectors,
            descriptions=[{"text": item} for item in train_ids],
            vector_type=vector_type,
            group_type="",
        )
        vector_map.insert_with_auto_cluster(folder.key, source_group, alpha=0.5)
        key_vectors[folder.key] = train_rep_array
        flat.extend((stable_id(folder.key, item_id), vector) for item_id, vector in zip(train_ids, train_vectors))
        for index in range(test_end):
            query_id = f"fold-0/{record_id(folder.name, index)}"
            queries.append((query_id, vectors[index], test_rep_array))
    vector_map.set_key_vectors(key_vectors)
    vector_map.build_hdmg(
        embedding_k=12,
        semantic_intra_k=20,
        semantic_bridge_keys=2,
        semantic_bridge_per_key=1,
        use_mutual_embedding=False,
    )
    return vector_map, flat, queries


def python_results(dataset_id: str, folders: list[Folder], vector_type: str, max_queries: int, beta: float) -> dict:
    vector_map, flat, queries = build_python_state(folders, vector_type)
    output: dict[str, dict[str, list[str]]] = {}
    for raw_query_id, query, query_key in queries[:max_queries]:
        query_id = f"{dataset_id}/{raw_query_id}"
        exact = sorted((cosine_distance(query, vector), item_id) for item_id, vector in flat)[:3]
        representative = vector_map.search_with_representative_rerank(
            query, query_key_vector=query_key, beta=beta, top_k=3, num_groups=3, distance_method="cosine"
        )
        mixed = vector_map.search_with_mixed_key_rep_vec(
            query,
            query_key_vector=query_key,
            beta=beta,
            top_k=3,
            gruop_expansion_factor=3,
            distance_method="cosine",
        )
        hdmg = vector_map.search_hdmg(
            query,
            query_key_vector=query_key,
            alpha=beta,
            top_k=3,
            max_steps=100,
            distance_method="cosine",
            entry_alpha_threshold=0.3,
            cluster_pool_size=9,
            top_key_candidates=5,
            extra_hops=0,
        )
        output[query_id] = {
            "exact": [item_id for _, item_id in exact],
            "representative": result_ids(representative, 3),
            "mixed": result_ids(mixed, 3),
            "hdmg": result_ids(hdmg, 3),
        }
    return output


def cangjie_results(dataset_id: str, max_queries: int, beta: float) -> dict:
    command = f"parity {dataset_id} {max_queries} {beta}"
    completed = subprocess.run(
        ["cjpm", "run"], cwd=CJ_CORE, input=command + "\n", text=True,
        encoding="utf-8", errors="replace", capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    output: dict[str, dict[str, list[str]]] = {}
    for line in completed.stdout.splitlines():
        if not line.startswith("PARITY|"):
            continue
        _, query_id, method, values = line.split("|", 3)
        output.setdefault(query_id, {})[method] = values.split(";") if values else []
    return output


def compare_dataset(dataset_id: str, max_queries: int, beta: float) -> dict:
    dataset_name, vector_type = DATASETS[dataset_id]
    folders = load_file(ROOT / "dataset" / "precomputed" / DATASET_FILES[dataset_name])
    python = python_results(dataset_id, folders, vector_type, max_queries, beta)
    cangjie = cangjie_results(dataset_id, max_queries, beta)
    if python.keys() != cangjie.keys():
        raise RuntimeError(f"query set differs for dataset {dataset_id}: Python={python.keys()}, Cangjie={cangjie.keys()}")

    method_rows = {}
    query_rows = []
    for method in METHODS:
        exact_orders = 0
        overlaps = []
        for query_id in python:
            left = python[query_id][method]
            right = cangjie[query_id][method]
            exact_order = left == right
            overlap = len(set(left) & set(right)) / float(max(1, len(left)))
            exact_orders += int(exact_order)
            overlaps.append(overlap)
            query_rows.append({
                "queryId": query_id,
                "method": method,
                "python": left,
                "cangjie": right,
                "sameOrder": exact_order,
                "overlapAt3": overlap,
            })
        method_rows[method] = {
            "queries": len(overlaps),
            "sameOrderRate": exact_orders / float(max(1, len(overlaps))),
            "meanOverlapAt3": sum(overlaps) / float(max(1, len(overlaps))),
        }
    return {
        "datasetId": dataset_id,
        "dataset": dataset_name,
        "maxVectorsPerFolder": 20,
        "fold": 0,
        "beta": beta,
        "topK": 3,
        "methods": method_rows,
        "queries": query_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="1,2,3,4,5,6")
    parser.add_argument("--max-queries", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=ROOT / "manifests" / "python-cangjie-sample-queries.json")
    args = parser.parse_args()
    if args.max_queries <= 0 or not 0.0 <= args.beta <= 1.0:
        parser.error("max-queries must be positive and beta must be in [0,1]")
    dataset_ids = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not dataset_ids or any(item not in DATASETS for item in dataset_ids):
        parser.error("datasets must be comma-separated ids from 1..6")

    report = {
        "schemaVersion": 1,
        "pythonReference": "violas_python (read-only)",
        "scope": "repository sample artifacts; not full datasets",
        "datasets": [compare_dataset(item, args.max_queries, args.beta) for item in dataset_ids],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({row["dataset"]: row["methods"] for row in report["datasets"]}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
