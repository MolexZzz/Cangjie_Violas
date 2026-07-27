"""Validate the small, Git-tracked release result bundle."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results-summary" / "final-results.json"
ARTIFACTS = ROOT / "manifests" / "release-artifacts.json"
CODE_CONTEXT = ROOT / "results-summary" / "code-context-case-study.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    artifacts = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    code_context = json.loads(CODE_CONTEXT.read_text(encoding="utf-8"))

    require(results["schemaVersion"] == 1, "unsupported result schema")
    require(artifacts["schemaVersion"] == 1, "unsupported artifact schema")
    require(code_context["schemaVersion"] == 2, "unsupported code-context schema")
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

    require(
        code_context["protocol"] == "code-context-four-paradigm-v2",
        "unexpected code-context protocol",
    )
    require(code_context["projectKeys"] == 4, "expected four code-context project keys")
    require(code_context["entities"] > 0, "code-context entities are empty")
    require(code_context["relations"] > 0, "code-context relations are empty")
    require(code_context["queries"] == 4, "expected four code-context queries")
    context_rows = code_context["results"]
    require(len(context_rows) == 8, "expected baseline and Violas rows for four paradigms")
    expected_paradigms = {"EAR", "DDR", "RER", "CMP"}
    require(
        {row["paradigm"] for row in context_rows} == expected_paradigms,
        "code-context paradigms are incomplete",
    )
    for paradigm in expected_paradigms:
        rows = [row for row in context_rows if row["paradigm"] == paradigm]
        require(len(rows) == 2, f"expected two rows for {paradigm}")
    for row in context_rows:
        require(0 <= row["hits"] <= row["total"], f"invalid hits for {row['query']}")
        require(row["total"] > 0, f"invalid total for {row['query']}")
        require(0.0 <= row["score"] <= 1.0, f"invalid score for {row['query']}")
        require(
            abs(row["score"] - row["hits"] / row["total"]) < 1e-9,
            f"inconsistent score for {row['query']}",
        )

    for markdown in (
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "docs" / "guides").glob("*.md"),
        *(ROOT / "docs" / "reports").glob("*.md"),
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
