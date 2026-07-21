"""Generate a frozen Python fixture and compare it with Cangjie TRACE output."""

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
import sklearn


ROOT = Path(__file__).resolve().parents[1]
PYTHON_REFERENCE_ROOT = ROOT / "violas_python"
sys.path.insert(0, str(PYTHON_REFERENCE_ROOT))

from violas.storage.vectorgroup import VectorGroup  # noqa: E402
from violas.storage.vectormap import VectorMap  # noqa: E402


def make_fixture() -> VectorGroup:
    vectors = []
    descriptions = []
    for index in range(16):
        x = -10.0 + index * 0.001 if index % 2 == 0 else 10.0 + index * 0.001
        vectors.append(np.array([x, 0.0], dtype=np.float64))
        descriptions.append({"text": f"item-{index}"})
    return VectorGroup(
        group_name="alternating",
        representative=np.array([0.0, 0.0], dtype=np.float64),
        rep_description="fixture representative",
        vectors=vectors,
        descriptions=descriptions,
        vector_type="test",
        group_type="fixture",
    )


def result_members(results) -> str:
    members = []
    for result in results:
        if result.vector_idx is None:
            members.append(result.group.group_name)
        else:
            members.append(result.group.descriptions[result.vector_idx].get("text", "missing"))
    return ",".join(members)


def python_trace() -> dict[str, str]:
    vector_map = VectorMap()
    created = vector_map.insert_with_auto_cluster("topic", make_fixture(), alpha=0.75)
    if created != 2:
        raise AssertionError(f"Python reference created {created} clusters, expected 2")

    clusters = []
    for key in sorted(vector_map.data):
        for group in vector_map.data[key]["groups"]:
            clusters.append(",".join(sorted(desc["text"] for desc in group.descriptions)))
    clusters.sort()
    return {
        "clusters": ";".join(clusters),
        "search.left": result_members(vector_map.search(np.array([-10.0, 0.0]), top_k=3, mode="single")),
        "search.right": result_members(vector_map.search(np.array([10.0, 0.0]), top_k=3, mode="single")),
    }


def cangjie_trace() -> tuple[dict[str, str], str]:
    completed = subprocess.run(
        ["cjpm", "run"],
        cwd=ROOT / "cj_core",
        input="2\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("TRACE|clusters|"):
            parsed["clusters"] = line.split("|", 2)[2]
        elif line.startswith("TRACE|search|"):
            _, _, query, members = line.split("|", 3)
            parsed[f"search.{query}"] = members
    return parsed, completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests" / "python-cangjie-fixture.json",
    )
    args = parser.parse_args()

    expected = python_trace()
    actual, stdout = cangjie_trace()
    keys = sorted(set(expected) | set(actual))
    comparisons = {
        key: {"python": expected.get(key), "cangjie": actual.get(key), "equal": expected.get(key) == actual.get(key)}
        for key in keys
    }
    payload = {
        "schemaVersion": 1,
        "pythonReference": "violas_python (read-only)",
        "pythonVersion": sys.version.split()[0],
        "numpyVersion": np.__version__,
        "sklearnVersion": sklearn.__version__,
        "fixture": "alternating-two-cluster-v1",
        "comparisons": comparisons,
        "allEqual": all(row["equal"] for row in comparisons.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["allEqual"]:
        print("\nCangjie output:\n" + stdout, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
