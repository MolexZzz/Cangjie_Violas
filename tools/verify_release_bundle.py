"""Validate the small, Git-tracked release result bundle."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results-summary" / "final-results.json"
ARTIFACTS = ROOT / "manifests" / "release-artifacts.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    artifacts = json.loads(ARTIFACTS.read_text(encoding="utf-8"))

    require(results["schemaVersion"] == 1, "unsupported result schema")
    require(artifacts["schemaVersion"] == 1, "unsupported artifact schema")
    require(results["queryScope"] == "full-10-percent-test-pool", "result is not full query scope")

    result_rows = results["datasets"]
    artifact_rows = artifacts["artifacts"]
    require(len(result_rows) == 3, "expected three result datasets")
    require(len(artifact_rows) == 3, "expected three artifact datasets")
    require(
        {row["name"] for row in result_rows} == {row["dataset"] for row in artifact_rows},
        "result and artifact datasets differ",
    )
    for row in result_rows:
        require(row["training"] > 0 and row["queries"] > 0, f"invalid counts for {row['name']}")
        require(0.0 <= row["mixedRecallAt3"] <= 1.0, f"invalid recall for {row['name']}")
        require(0.0 <= row["mixedNdcgAt3"] <= 1.0, f"invalid NDCG for {row['name']}")
        require(row["latencyMsPerQuery"] > 0.0, f"invalid latency for {row['name']}")
    for row in artifact_rows:
        require(bool(SHA256.fullmatch(row["sha256"])), f"invalid SHA-256 for {row['dataset']}")
        require(row["bytes"] > 0, f"invalid artifact size for {row['dataset']}")
    for name, digest in results["sourceArtifacts"].items():
        require(bool(SHA256.fullmatch(digest)), f"invalid source digest: {name}")

    for markdown in (
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "docs" / "guides").glob("*.md"),
        *(ROOT / "results-summary").glob("*.md"),
    ):
        for target in MARKDOWN_LINK.findall(markdown.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            require(
                (markdown.parent / relative).resolve().exists(),
                f"broken link in {markdown.relative_to(ROOT)}: {target}",
            )

    print("release bundle validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
