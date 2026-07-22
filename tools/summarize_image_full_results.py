"""Merge persisted Cangjie and external-database full-image benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BACKENDS = ("milvus", "qdrant", "chroma")
DATASETS = ("caltech", "cub", "coco")


def parse_cangjie(path: Path) -> dict[float, dict[str, float | str]]:
    rows: dict[float, dict[str, float | str]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "PAPER_SUMMARY|"
        if marker not in line:
            continue
        fields: dict[str, str] = {}
        for part in line.split(marker, 1)[1].split("|"):
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        beta = float(fields.pop("beta"))
        parsed: dict[str, float | str] = {}
        for key, value in fields.items():
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        rows[beta] = parsed
    return rows


def parse_external(path: Path) -> dict[float, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {float(run["beta"]): run for run in payload.get("runs", [])}


def number(value: object, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def summarize_dataset(run_root: Path, dataset: str) -> dict:
    summary_log = run_root / "cangjie" / f"{dataset}.summary.log"
    cangjie = parse_cangjie(summary_log if summary_log.exists()
                            else run_root / "cangjie" / f"{dataset}.log")
    external = {
        backend: parse_external(run_root / "external" / f"{backend}-{dataset}-full.json")
        for backend in BACKENDS
    }
    betas = sorted(set(cangjie) | {beta for rows in external.values() for beta in rows})
    rows = []
    for beta in betas:
        row = {"beta": beta, "cangjie": cangjie.get(beta), "external": {}}
        for backend in BACKENDS:
            row["external"][backend] = external[backend].get(beta)
        rows.append(row)
    return {"dataset": dataset, "rows": rows}


def markdown(summary: dict) -> str:
    lines = [f"# {summary['dataset']} full benchmark", "", "## Recall@3 / NDCG@3", ""]
    lines.append("| beta | Mixed R/N | Representative R/N | PythonFlat R/N | Milvus raw/mix | Qdrant raw/mix | Chroma raw/mix |")
    lines.append("| ---: | --- | --- | --- | --- | --- | --- |")
    for row in summary["rows"]:
        cj = row["cangjie"] or {}
        cells = [
            number(row["beta"], 1),
            f"{number(cj.get('mixedRecall'))}/{number(cj.get('mixedNdcg'))}",
            f"{number(cj.get('representativeRecall'))}/{number(cj.get('representativeNdcg'))}",
            f"{number(cj.get('pythonFlatRecall'))}/{number(cj.get('pythonFlatNdcg'))}",
        ]
        for backend in BACKENDS:
            data = row["external"].get(backend) or {}
            raw, mixed = data.get("rawVector", {}), data.get("mixedRerank", {})
            cells.append(f"{number(raw.get('recallAtK'))}/{number(mixed.get('recallAtK'))}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Latency (ms/query)", ""])
    lines.append("| beta | HDMG | Representative | PythonFlat | Milvus raw/mix | Qdrant raw/mix | Chroma raw/mix |")
    lines.append("| ---: | ---: | ---: | ---: | --- | --- | --- |")
    for row in summary["rows"]:
        cj = row["cangjie"] or {}
        cells = [number(row["beta"], 1), number(cj.get("mixedLatencyMs"), 2),
                 number(cj.get("representativeLatencyMs"), 2), number(cj.get("pythonFlatLatencyMs"), 2)]
        for backend in BACKENDS:
            data = row["external"].get(backend) or {}
            raw, mixed = data.get("rawLatencyMs", {}), data.get("latencyMs", {})
            cells.append(f"{number(raw.get('databaseMean'), 2)}/{number(mixed.get('totalMean'), 2)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    combined = {"schemaVersion": 1, "datasets": []}
    for dataset in DATASETS:
        summary = summarize_dataset(args.run_root, dataset)
        combined["datasets"].append(summary)
        rendered = markdown(summary)
        (args.run_root / f"summary-{dataset}.md").write_text(rendered, encoding="utf-8")
        if summary["rows"]:
            print("\n" + rendered)
    (args.run_root / "summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote summaries under {args.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
