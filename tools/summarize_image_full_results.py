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


def raw_comparison(row: dict) -> dict:
    paper = row.get("paperComparison", {})
    return paper if paper.get("method") == "direct-vector-top-k" else row.get("rawComparison", {})


def mixed_comparison(row: dict) -> dict:
    mixed = row.get("mixedComparison", {})
    if mixed.get("method") == "vector-candidate-plus-local-mixed-rerank":
        return mixed
    paper = row.get("paperComparison", {})
    return paper if paper.get("method") == "vector-candidate-plus-local-mixed-rerank" else {}


def dataset_run_root(run_root: Path, dataset: str) -> Path:
    # Never borrow sibling runs implicitly: doing so can silently combine
    # incompatible metric/protocol versions in one table.
    return run_root


def number(value: object, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def best_value(values: list[object], *, higher_is_better: bool) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return max(numeric) if higher_is_better else min(numeric)


def highlighted(value: object, best: float | None, digits: int) -> str:
    rendered = number(value, digits)
    if value is not None and best is not None and abs(float(value) - best) < 1e-12:
        return f"**{rendered}**"
    return rendered


def summarize_dataset(run_root: Path, dataset: str) -> dict:
    source_root = dataset_run_root(run_root, dataset)
    summary_log = source_root / "cangjie" / f"{dataset}.summary.log"
    cangjie = parse_cangjie(summary_log if summary_log.exists()
                            else source_root / "cangjie" / f"{dataset}.log")
    external = {
        backend: parse_external(source_root / "external" / f"{backend}-{dataset}-full.json")
        for backend in BACKENDS
    }
    betas = sorted(set(cangjie) | {beta for rows in external.values() for beta in rows})
    rows = []
    for beta in betas:
        row = {"beta": beta, "cangjie": cangjie.get(beta), "external": {}}
        if row["cangjie"] and row["cangjie"].get("evaluationProtocol") != "violas-paper-table2-v5":
            raise ValueError(f"{dataset} beta={beta}: refusing non-v5 Cangjie metrics")
        if row["cangjie"] and row["cangjie"].get("queryScope") != "full-10-percent-test-pool":
            raise ValueError(f"{dataset} beta={beta}: refusing incomplete query set")
        for backend in BACKENDS:
            backend_row = external[backend].get(beta)
            if (backend_row and mixed_comparison(backend_row).get("method")
                    != "vector-candidate-plus-local-mixed-rerank"):
                raise ValueError(f"{dataset}/{backend} beta={beta}: refusing non-v5 mixed baseline")
            if (backend_row and raw_comparison(backend_row).get("method")
                    != "direct-vector-top-k"):
                raise ValueError(f"{dataset}/{backend} beta={beta}: refusing missing raw paper baseline")
            if backend_row and backend_row.get("queryScope") != "full-10-percent-test-pool":
                raise ValueError(f"{dataset}/{backend} beta={beta}: refusing incomplete query set")
            row["external"][backend] = backend_row
        rows.append(row)
    return {"dataset": dataset, "rows": rows}


def markdown(summary: dict) -> str:
    lines = [f"# {summary['dataset']} full benchmark", "", "## Paper Table 2: Mixed Recall@3 / Mixed NDCG@3", ""]
    lines.append("| beta | Violas R/N | w/o HDMG R/N | Milvus R/N | Qdrant R/N | Chroma R/N |")
    lines.append("| ---: | --- | --- | --- | --- | --- |")
    for row in summary["rows"]:
        cj = row["cangjie"] or {}
        metric_pairs = [
            (cj.get("mixedRecall"), cj.get("mixedNdcg")),
            (cj.get("withoutHdmgRecall"), cj.get("withoutHdmgNdcg")),
        ]
        for backend in BACKENDS:
            data = row["external"].get(backend) or {}
            raw = raw_comparison(data)
            metric_pairs.append((raw.get("recallAtK"), raw.get("ndcgAtK")))
        best_recall = best_value([pair[0] for pair in metric_pairs], higher_is_better=True)
        best_ndcg = best_value([pair[1] for pair in metric_pairs], higher_is_better=True)
        cells = [number(row["beta"], 1)]
        cells.extend(
            f"{highlighted(recall, best_recall, 4)}/{highlighted(ndcg, best_ndcg, 4)}"
            for recall, ndcg in metric_pairs
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Paper Table 2: latency (ms/query)", ""])
    lines.append("| beta | Violas | w/o HDMG | Milvus | Qdrant | Chroma |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["rows"]:
        cj = row["cangjie"] or {}
        latencies = [cj.get("mixedLatencyMs"), cj.get("withoutHdmgLatencyMs")]
        for backend in BACKENDS:
            data = row["external"].get(backend) or {}
            raw = raw_comparison(data)
            latencies.append(raw.get("latencyMs", {}).get("databaseMean"))
        best_latency = best_value(latencies, higher_is_better=False)
        cells = [number(row["beta"], 1)]
        cells.extend(highlighted(value, best_latency, 2) for value in latencies)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Auxiliary mixed-rerank and diagnostic baselines", ""])
    lines.append("| beta | Representative-3 R/N | PythonFlat R/N | Milvus mixed R/N | Qdrant mixed R/N | Chroma mixed R/N |")
    lines.append("| ---: | --- | --- | --- | --- | --- |")
    for row in summary["rows"]:
        cj = row["cangjie"] or {}
        pairs = [
            (cj.get("representativeRecall"), cj.get("representativeNdcg")),
            (cj.get("pythonFlatRecall"), cj.get("pythonFlatNdcg")),
        ]
        for backend in BACKENDS:
            mixed = mixed_comparison(row["external"].get(backend) or {})
            pairs.append((mixed.get("recallAtK"), mixed.get("ndcgAtK")))
        cells = [number(row["beta"], 1)]
        cells.extend(f"{number(recall, 4)}/{number(ndcg, 4)}" for recall, ndcg in pairs)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "", "## Method notes", "",
        "- `Violas` is the Cangjie HDMG search.",
        "- `w/o HDMG` first routes five entity groups by entity score, directly scans their micro-clusters, keeps up to nine per group, and reranks members by mixed score.",
        "- In the main paper-style table, Milvus, Qdrant, and Chroma rank instance embeddings only and directly return vector Top-3; they do not use entity representations or local mixed reranking.",
        "- The auxiliary table reports the enhanced two-stage database variant: retrieve 30 vector candidates, then locally rerank them by mixed score. Representative/PythonFlat are retained as diagnostics.",
        "- A Table 2 full run uses every query in the frozen 10% test pool, not a 200-query cap.",
        "- NDCG uses the paper's graded gain `2^mixed_score - 1`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    combined = {"schemaVersion": 1, "datasets": []}
    for dataset in args.datasets:
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
