"""Run a reproducible HDMG accuracy/latency parameter scan on frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CJ_CORE = ROOT / "cj_core"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "python-paper-90-10"
DATASETS = ("caltech", "cub", "coco")

CONFIGS = (
    {
        "name": "sparse-graph",
        "embeddingK": 8,
        "semanticIntraK": 12,
        "semanticBridgeKeys": 2,
        "semanticBridgePerKey": 1,
        "clusterPoolMultiplier": 3,
        "topKeyCandidates": 5,
    },
    {
        "name": "small-candidate-pool",
        "embeddingK": 12,
        "semanticIntraK": 20,
        "semanticBridgeKeys": 2,
        "semanticBridgePerKey": 1,
        "clusterPoolMultiplier": 2,
        "topKeyCandidates": 3,
    },
    {
        "name": "paper-default",
        "embeddingK": 12,
        "semanticIntraK": 20,
        "semanticBridgeKeys": 2,
        "semanticBridgePerKey": 1,
        "clusterPoolMultiplier": 3,
        "topKeyCandidates": 5,
    },
    {
        "name": "wide-candidate-pool",
        "embeddingK": 12,
        "semanticIntraK": 20,
        "semanticBridgeKeys": 2,
        "semanticBridgePerKey": 1,
        "clusterPoolMultiplier": 5,
        "topKeyCandidates": 8,
    },
    {
        "name": "dense-graph",
        "embeddingK": 16,
        "semanticIntraK": 24,
        "semanticBridgeKeys": 4,
        "semanticBridgePerKey": 2,
        "clusterPoolMultiplier": 3,
        "topKeyCandidates": 5,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_dirty() -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def executable_path() -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    return CJ_CORE / "target" / "release" / "bin" / f"main{suffix}"


def parse_summary(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.startswith("PAPER_SUMMARY|")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one PAPER_SUMMARY line, found {len(lines)}")
    values: dict[str, object] = {}
    for item in lines[0].split("|")[1:]:
        key, value = item.split("=", 1)
        if value == "true":
            values[key] = True
        elif value == "false":
            values[key] = False
        else:
            try:
                values[key] = float(value)
            except ValueError:
                values[key] = value
    return values


def run_one(
    executable: Path,
    artifact: Path,
    config: dict,
    queries: int,
    beta: float,
    cangjie_heap_size: str,
) -> dict:
    command = "paper-config {artifact} {queries} {beta} {embeddingK} {semanticIntraK} " \
        "{semanticBridgeKeys} {semanticBridgePerKey} {clusterPoolMultiplier} {topKeyCandidates}".format(
            artifact=artifact,
            queries=queries,
            beta=beta,
            **config,
        )
    environment = os.environ.copy()
    environment["cjHeapSize"] = cangjie_heap_size
    completed = subprocess.run(
        [str(executable)],
        cwd=CJ_CORE,
        input=command + "\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{config['name']} failed for {artifact.parent.name}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    summary = parse_summary(completed.stdout)
    return {
        "dataset": summary["dataset"],
        "config": config["name"],
        "queries": int(summary["queries"]),
        "beta": float(summary["beta"]),
        "mixedRecallAt3": float(summary["mixedRecall"]),
        "mixedNdcgAt3": float(summary["mixedNdcg"]),
        "latencyMsPerQuery": float(summary["mixedLatencyMs"]),
        "candidatePool": float(summary["hdmgCandidatePool"]),
        "hops": float(summary["hdmgHops"]),
        "buildParameters": {
            key: config[key]
            for key in (
                "embeddingK",
                "semanticIntraK",
                "semanticBridgeKeys",
                "semanticBridgePerKey",
            )
        },
        "searchParameters": {
            key: config[key]
            for key in ("clusterPoolMultiplier", "topKeyCandidates")
        },
    }


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for config in CONFIGS:
        selected = [row for row in rows if row["config"] == config["name"]]
        output.append(
            {
                "config": config["name"],
                "datasets": len(selected),
                "meanRecallAt3": statistics.fmean(row["mixedRecallAt3"] for row in selected),
                "minRecallAt3": min(row["mixedRecallAt3"] for row in selected),
                "meanNdcgAt3": statistics.fmean(row["mixedNdcgAt3"] for row in selected),
                "meanLatencyMsPerQuery": statistics.fmean(
                    row["latencyMsPerQuery"] for row in selected
                ),
                "meanCandidatePool": statistics.fmean(row["candidatePool"] for row in selected),
                "parameters": {
                    key: config[key]
                    for key in (
                        "embeddingK",
                        "semanticIntraK",
                        "semanticBridgeKeys",
                        "semanticBridgePerKey",
                        "clusterPoolMultiplier",
                        "topKeyCandidates",
                    )
                },
            }
        )
    return output


def render_markdown(payload: dict) -> str:
    lines = [
        "# HDMG accuracy and latency parameter scan",
        "",
        (
            f"Frozen Caltech-101, CUB-200-2011, and COCO artifacts; "
            f"first {payload['queriesPerDataset']} deterministic queries per dataset; "
            f"β={payload['beta']}; Recall@3 and NDCG@3 use exact mixed-search ground truth."
        ),
        "",
        "| Configuration | Mean Recall@3 | Minimum Recall@3 | Mean NDCG@3 | "
        "Mean latency (ms/query) | Mean candidate pool |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['config']} | {row['meanRecallAt3']:.6f} | "
            f"{row['minRecallAt3']:.6f} | {row['meanNdcgAt3']:.6f} | "
            f"{row['meanLatencyMsPerQuery']:.3f} | {row['meanCandidatePool']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The `paper-default` configuration remains the release default. "
            "A different configuration should only replace it if it improves the "
            "accuracy/latency trade-off consistently and is subsequently rerun on the full query pools.",
            "",
            "## Per-dataset results",
            "",
            "| Dataset | Configuration | Recall@3 | NDCG@3 | Latency (ms/query) | Candidate pool |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['config']} | {row['mixedRecallAt3']:.6f} | "
            f"{row['mixedNdcgAt3']:.6f} | {row['latencyMsPerQuery']:.3f} | "
            f"{row['candidatePool']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Violas commit at scan start: `{payload['provenance']['gitCommit']}`",
            f"- Generated at UTC: `{payload['provenance']['generatedAtUtc']}`",
            f"- Runner: `tools/run_accuracy_parameter_scan.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--cangjie-heap-size", default="2GB")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "accuracy-parameter-scan" / "hdmg-parameter-scan.json",
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.queries <= 0:
        parser.error("--queries must be positive")
    if not 0.0 <= args.beta <= 1.0:
        parser.error("--beta must be in [0, 1]")

    if not args.skip_build:
        subprocess.run(["cjpm", "build"], cwd=CJ_CORE, check=True)
    executable = executable_path()
    if not executable.exists():
        raise FileNotFoundError(executable)

    artifacts = {
        dataset: args.artifacts_root / f"{dataset}-full" / "cangjie_input.txt"
        for dataset in DATASETS
    }
    for path in artifacts.values():
        if not path.exists():
            raise FileNotFoundError(path)

    rows = []
    for config in CONFIGS:
        for dataset, artifact in artifacts.items():
            print(f"[scan] {config['name']} / {dataset}", flush=True)
            rows.append(
                run_one(
                    executable,
                    artifact,
                    config,
                    args.queries,
                    args.beta,
                    args.cangjie_heap_size,
                )
            )

    payload = {
        "schemaVersion": 1,
        "protocol": "violas-hdmg-parameter-scan-v1",
        "datasets": list(DATASETS),
        "queriesPerDataset": args.queries,
        "beta": args.beta,
        "rows": rows,
        "aggregate": aggregate(rows),
        "artifacts": {
            dataset: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for dataset, path in artifacts.items()
        },
        "provenance": {
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "gitCommit": git_commit(),
            "workingTreeDirty": git_dirty(),
            "os": platform.platform(),
            "executable": str(executable.relative_to(ROOT)),
            "cangjieHeapSize": args.cangjie_heap_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
