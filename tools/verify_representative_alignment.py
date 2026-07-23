"""Run the Python Representative method on a frozen paper artifact.

This isolates algorithm alignment from dataset, CLIP, split, and query-order
differences: Python consumes the exact micro-clusters and ground truth used by
the Cangjie paper runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "violas_python"))
sys.path.insert(0, str(ROOT / "tools"))

from paper_artifact import load_artifact  # noqa: E402
from violas.storage import VectorGroup, VectorMap  # noqa: E402


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def overlap(expected: list[str], actual: list[str], top_k: int) -> float:
    return len(set(expected[:top_k]) & set(actual[:top_k])) / float(top_k)


def result_id(result) -> str:
    return result.group.descriptions[result.vector_idx]["recordId"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-queries", type=int, default=200)
    args = parser.parse_args()

    manifest, records, queries, key_vectors, ground_truth = load_artifact(args.artifact)
    cluster_by_id = {row["recordId"]: row for row in jsonl(args.artifact / "microclusters.jsonl")}
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        cluster = cluster_by_id[record["recordId"]]
        groups.setdefault((record["key"], cluster["clusterGroup"]), []).append(record)

    vector_map = VectorMap()
    for (base_key, group_name), rows in groups.items():
        vectors = [np.asarray(row["vector"], dtype=np.float64) for row in rows]
        vector_map.insert(
            base_key,
            VectorGroup(
                group_name=group_name,
                representative=np.mean(np.asarray(vectors), axis=0),
                rep_description="simple_mean",
                vectors=vectors,
                descriptions=[{"recordId": row["recordId"]} for row in rows],
                vector_type="image",
            ),
        )
    vector_map.set_key_vectors({key: np.asarray(value, dtype=np.float64) for key, value in key_vectors.items()})

    top_k = int(manifest["topK"])
    recalls: list[float] = []
    mismatches = 0
    for query in queries[: args.max_queries]:
        results = vector_map.search_with_representative_rerank(
            np.asarray(query["vector"], dtype=np.float64),
            query_key_vector=np.asarray(query["keyVector"], dtype=np.float64),
            beta=args.beta,
            top_k=top_k,
            num_groups=top_k,
            distance_method="cosine",
        )
        actual = [result_id(result) for result in results]
        expected = ground_truth[(query["queryId"], float(args.beta))]
        recall = overlap(expected, actual, top_k)
        recalls.append(recall)
        if recall < 1.0:
            mismatches += 1

    print(json.dumps({
        "dataset": manifest["dataset"],
        "implementation": "violas_python.search_with_representative_rerank",
        "beta": args.beta,
        "queries": len(recalls),
        "averageRecallAtK": sum(recalls) / len(recalls),
        "mismatchedQueries": mismatches,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
