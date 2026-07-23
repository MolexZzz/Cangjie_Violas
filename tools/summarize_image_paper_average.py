"""Build the three-image-dataset paper-style average and mixed-rerank diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


DATASETS = ("caltech", "cub", "coco")
METHODS = ("Violas", "w/o HDMG", "Milvus", "Qdrant", "Chroma")
BACKEND_KEYS = {"Milvus": "milvus", "Qdrant": "qdrant", "Chroma": "chroma"}


def raw_comparison(row: dict) -> dict:
    paper = row.get("paperComparison", {})
    return paper if paper.get("method") == "direct-vector-top-k" else row.get("rawComparison", {})


def mixed_comparison(row: dict) -> dict:
    mixed = row.get("mixedComparison", {})
    if mixed.get("method") == "vector-candidate-plus-local-mixed-rerank":
        return mixed
    paper = row.get("paperComparison", {})
    return paper if paper.get("method") == "vector-candidate-plus-local-mixed-rerank" else {}


def find_summary(results_root: Path, dataset: str) -> Path:
    matches = list(results_root.glob(f"*-{dataset}-full-final/summary.json"))
    if not matches:
        raise FileNotFoundError(f"no full-final summary.json found for {dataset} under {results_root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_dataset(path: Path, dataset: str) -> dict[float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    for item in payload.get("datasets", []):
        if item.get("dataset") == dataset and item.get("rows"):
            rows = {float(row["beta"]): row for row in item["rows"]}
            if len(rows) != 11:
                raise ValueError(f"{path}: expected 11 beta rows for {dataset}, got {len(rows)}")
            for beta, row in rows.items():
                if row.get("cangjie", {}).get("evaluationProtocol") != "violas-paper-table2-v5":
                    raise ValueError(f"{path}: {dataset} beta={beta} is not Cangjie v5")
                if row.get("cangjie", {}).get("queryScope") != "full-10-percent-test-pool":
                    raise ValueError(f"{path}: {dataset} beta={beta} is not the complete test pool")
                for backend in BACKEND_KEYS.values():
                    backend_row = row.get("external", {}).get(backend, {})
                    method = mixed_comparison(backend_row).get("method")
                    if method != "vector-candidate-plus-local-mixed-rerank":
                        raise ValueError(f"{path}: {dataset}/{backend} beta={beta} is not database v5")
                    raw_method = raw_comparison(backend_row).get("method")
                    if raw_method != "direct-vector-top-k":
                        raise ValueError(f"{path}: {dataset}/{backend} beta={beta} has no raw paper baseline")
                    if backend_row.get("queryScope") != "full-10-percent-test-pool":
                        raise ValueError(f"{path}: {dataset}/{backend} beta={beta} is not the complete test pool")
            return rows
    raise ValueError(f"{path}: missing non-empty dataset entry for {dataset}")


def method_metrics(row: dict, method: str) -> tuple[float, float, float]:
    if method == "Violas":
        data = row["cangjie"]
        return float(data["mixedRecall"]), float(data["mixedNdcg"]), float(data["mixedLatencyMs"])
    if method == "w/o HDMG":
        data = row["cangjie"]
        return (
            float(data["withoutHdmgRecall"]),
            float(data["withoutHdmgNdcg"]),
            float(data["withoutHdmgLatencyMs"]),
        )
    data = row["external"][BACKEND_KEYS[method]]
    raw = raw_comparison(data)
    return (
        float(raw["recallAtK"]),
        float(raw["ndcgAtK"]),
        float(raw["latencyMs"]["databaseMean"]),
    )


def database_variant_metrics(row: dict, method: str, variant: str) -> tuple[float, float, float]:
    data = row["external"][BACKEND_KEYS[method]]
    if variant == "raw":
        return (
            float(data["rawVector"]["recallAtK"]),
            float(data["rawVector"]["ndcgAtK"]),
            float(data["rawLatencyMs"]["databaseMean"]),
        )
    paper = mixed_comparison(data)
    return (
        float(paper["recallAtK"]),
        float(paper["ndcgAtK"]),
        float(paper["latencyMs"]["totalMean"]),
    )


def aggregate(inputs: dict[str, dict[float, dict]]) -> list[dict]:
    beta_sets = [set(rows) for rows in inputs.values()]
    if any(values != beta_sets[0] for values in beta_sets[1:]):
        raise ValueError("datasets do not contain identical beta values")
    output = []
    for beta in sorted(beta_sets[0]):
        methods = {}
        for method in METHODS:
            values = [method_metrics(inputs[dataset][beta], method) for dataset in DATASETS]
            methods[method] = {
                "recallAtK": fmean(value[0] for value in values),
                "ndcgAtK": fmean(value[1] for value in values),
                "latencyMs": fmean(value[2] for value in values),
            }
        database_variants = {}
        for method in BACKEND_KEYS:
            database_variants[method] = {}
            for variant in ("raw", "mixed"):
                values = [database_variant_metrics(inputs[dataset][beta], method, variant) for dataset in DATASETS]
                database_variants[method][variant] = {
                    "recallAtK": fmean(value[0] for value in values),
                    "ndcgAtK": fmean(value[1] for value in values),
                    "latencyMs": fmean(value[2] for value in values),
                }
        output.append({"beta": beta, "methods": methods, "databaseVariants": database_variants})
    return output


def emphasized(value: float, best: float, digits: int) -> str:
    rendered = f"{value:.{digits}f}"
    return f"<strong>{rendered}</strong>" if abs(value - best) < 1e-12 else rendered


def render_markdown(rows: list[dict], sources: dict[str, Path]) -> str:
    lines = [
        "# Average image-dataset paper-style comparison",
        "",
        "Average over Caltech-101, CUB-200-2011, and COCO. Each dataset contributes equally; each run uses its complete 10% test pool.",
        "",
        "<table>",
        "  <thead>",
        "    <tr><th rowspan=\"2\">β</th><th colspan=\"5\">Mixed Recall@3</th><th colspan=\"5\">Mixed NDCG@3</th><th colspan=\"5\">Latency (ms/query)</th></tr>",
        "    <tr>" + "".join(f"<th>{method}</th>" for _ in range(3) for method in METHODS) + "</tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for row in rows:
        methods = row["methods"]
        best_recall = max(methods[name]["recallAtK"] for name in METHODS)
        best_ndcg = max(methods[name]["ndcgAtK"] for name in METHODS)
        best_latency = min(methods[name]["latencyMs"] for name in METHODS)
        cells = [f"<td>{row['beta']:.1f}</td>"]
        cells.extend(f"<td>{emphasized(methods[name]['recallAtK'], best_recall, 3)}</td>" for name in METHODS)
        if row["beta"] >= 1.0 - 1e-12:
            cells.extend("<td>—</td>" for _ in METHODS)
        else:
            cells.extend(f"<td>{emphasized(methods[name]['ndcgAtK'], best_ndcg, 3)}</td>" for name in METHODS)
        cells.extend(f"<td>{emphasized(methods[name]['latencyMs'], best_latency, 2)}</td>" for name in METHODS)
        lines.append("    <tr>" + "".join(cells) + "</tr>")
    lines.extend([
        "  </tbody>",
        "</table>",
        "",
        "Notes:",
        "",
        "- Violas is the Cangjie HDMG result. `w/o HDMG` first routes entity groups by entity score and directly searches micro-clusters inside those groups.",
        "- In the main paper-style table, Milvus, Qdrant, and Chroma rank instance embeddings only and directly return vector Top-3, without entity representations or local mixed reranking.",
        "- NDCG uses mixed score as the graded gain, following Equation 14.",
        "- NDCG at β=1.0 is shown as `—`, matching the paper table, because all records under the winning semantic key are tied.",
        "- This is a three-image-dataset average, not the paper's six-dataset average.",
        "",
        "## Auxiliary database two-stage mixed-rerank results", "",
        "Each database first retrieves 30 candidates by embedding similarity, then the benchmark locally reranks them by mixed score. These enhanced results are not used in the main paper-style table.", "",
        "| β | Milvus Recall | Qdrant Recall | Chroma Recall | Milvus NDCG | Qdrant NDCG | Chroma NDCG | Milvus latency | Qdrant latency | Chroma latency |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        cells = [f"{row['beta']:.1f}"]
        for metric in ("recallAtK", "ndcgAtK", "latencyMs"):
            for method in BACKEND_KEYS:
                mixed = row["databaseVariants"][method]["mixed"][metric]
                digits = 2 if metric == "latencyMs" else 3
                cells.append(f"{mixed:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Sources:",
        "",
    ])
    lines.extend(f"- {dataset}: `{path}`" for dataset, path in sources.items())
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/python-paper-90-10"))
    parser.add_argument("--summary-json", type=Path,
                        help="combined summary.json produced by run_image_full_suite.ps1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.summary_json:
        sources = {dataset: args.summary_json for dataset in DATASETS}
    else:
        sources = {dataset: find_summary(args.results_root, dataset) for dataset in DATASETS}
    inputs = {dataset: load_dataset(sources[dataset], dataset) for dataset in DATASETS}
    rows = aggregate(inputs)
    output = args.output or (
        args.summary_json.parent / "average-paper-table.md"
        if args.summary_json
        else Path("results/python-paper-90-10/average-paper-table.md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(rows, sources), encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps({
        "schemaVersion": 1,
        "protocol": "python-paper-90-10",
        "datasets": list(DATASETS),
        "paperDatabaseMode": "direct-vector-top-k",
        "auxiliaryDatabaseMode": "candidate-vector-plus-local-mixed-rerank",
        "sources": {key: str(value) for key, value in sources.items()},
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
