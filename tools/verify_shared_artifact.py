#!/usr/bin/env python3
"""Verify that Python JSONL and the Cangjie stream contain identical inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from paper_artifact import jsonl_rows, load_artifact


def vector(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",")]


def verify(path: Path) -> dict:
    manifest, records, queries, key_vectors, ground_truth = load_artifact(path)
    expected_records = {row["recordId"]: row for row in records}
    expected_clusters = {
        row["recordId"]: row for row in jsonl_rows(path / "microclusters.jsonl")
    }
    expected_queries = {row["queryId"]: row for row in queries}
    expected_keys = dict(key_vectors)
    expected_gt = dict(ground_truth)
    counts = {"META": 0, "KEY": 0, "TRAIN": 0, "QUERY": 0, "GT": 0}

    with (path / "cangjie_input.txt").open("r", encoding="utf-8") as source:
        for number, raw in enumerate(source, 1):
            parts = raw.rstrip("\r\n").split("\t")
            kind = parts[0]
            if kind not in counts:
                raise ValueError(f"line {number}: unknown command {kind!r}")
            counts[kind] += 1
            if kind == "META":
                actual = (parts[1], parts[2], int(parts[3]))
                expected = (manifest["protocol"], manifest["dataset"], manifest["topK"])
                if actual != expected:
                    raise ValueError(f"line {number}: META mismatch: {actual} != {expected}")
            elif kind == "KEY":
                expected = expected_keys.pop(parts[1], None)
                if expected is None or vector(parts[2]) != expected:
                    raise ValueError(f"line {number}: key vector mismatch for {parts[1]}")
            elif kind == "TRAIN":
                expected = expected_records.pop(parts[1], None)
                cluster = expected_clusters.pop(parts[1], None)
                if expected is None:
                    raise ValueError(f"line {number}: unexpected training record {parts[1]}")
                actual = (parts[2], parts[3], vector(parts[4]), parts[5], parts[6])
                wanted = (
                    expected["folder"], expected["key"], expected["vector"],
                    cluster["clusterKey"], cluster["clusterGroup"],
                )
                if actual != wanted:
                    raise ValueError(f"line {number}: training record mismatch for {parts[1]}")
            elif kind == "QUERY":
                expected = expected_queries.pop(parts[1], None)
                if expected is None:
                    raise ValueError(f"line {number}: unexpected query {parts[1]}")
                actual = (parts[2], parts[3], vector(parts[4]), vector(parts[5]))
                wanted = (
                    expected["sourceRecordId"], expected["trueKey"],
                    expected["vector"], expected["keyVector"],
                )
                if actual != wanted:
                    raise ValueError(f"line {number}: query mismatch for {parts[1]}")
            else:
                key = (parts[1], float(parts[2]))
                expected = expected_gt.pop(key, None)
                actual = parts[3].split(",") if parts[3] else []
                if expected is None or actual != expected:
                    raise ValueError(f"line {number}: ground truth mismatch for {key}")

    leftovers = {
        "keys": len(expected_keys),
        "training": len(expected_records),
        "microclusters": len(expected_clusters),
        "queries": len(expected_queries),
        "groundTruth": len(expected_gt),
    }
    if any(leftovers.values()) or counts["META"] != 1:
        raise ValueError(f"Cangjie stream is incomplete: counts={counts}, leftovers={leftovers}")
    return {
        "artifact": str(path),
        "status": "verified-identical",
        "protocol": manifest["protocol"],
        "dataset": manifest["dataset"],
        "checkpointSha256": manifest["preprocessing"]["clipModel"]["checkpointSha256"],
        "cangjieInputSha256": manifest["files"]["cangjie_input.txt"]["sha256"],
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--run-cangjie", action="store_true")
    parser.add_argument("--max-queries", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    result = verify(artifact)
    print(json.dumps(result, ensure_ascii=False))
    if args.run_cangjie:
        if args.max_queries <= 0 or not 0.0 <= args.beta <= 1.0:
            parser.error("max-queries must be positive and beta must be in [0,1]")
        project = Path(__file__).resolve().parents[1] / "cj_core"
        command = f"paper {artifact / 'cangjie_input.txt'} {args.max_queries} {args.beta}\n"
        subprocess.run(["cjpm", "run"], cwd=project, input=command, text=True, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
