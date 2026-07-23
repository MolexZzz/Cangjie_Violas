"""Summarize three maintenance JSON files into a paper-style Table 3 report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METHODS = ("cangjie-violas", "milvus", "qdrant", "chroma")
LABELS = {
    "cangjie-violas": "Violas",
    "milvus": "Milvus",
    "qdrant": "Qdrant",
    "chroma": "Chroma",
}


def mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def format_value(value: float | None, seconds: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1000.0:.3f}" if seconds else f"{value:.2f}"


def row_values(payload: dict) -> dict[str, dict]:
    return {row["backend"]: row for row in payload["results"] if row["backend"] in METHODS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if len({item["dataset"] for item in payloads}) != len(payloads):
        parser.error("each input must represent a different dataset")
    rows_by_dataset = [(item["dataset"], row_values(item)) for item in payloads]

    lines = [
        "# 三个图像数据集的数据与索引维护结果",
        "",
        "表格列与论文 Table 3 一致：200 条向量插入、200 条向量更新、初始索引构建和索引更新。",
        "",
        "| Dataset | " + " | ".join(
            f"{metric} {LABELS[method]}"
            for metric in ("Insert(ms)", "Update(ms)", "Build(s)", "Index update(s)")
            for method in METHODS
        ) + " |",
        "| --- | " + " | ".join("---:" for _ in range(16)) + " |",
    ]

    all_rows = []
    for dataset, rows in rows_by_dataset:
        values = []
        for operation, seconds in (
            ("vectorInsertion", False),
            ("vectorUpdate", False),
            ("indexConstruction", True),
            ("indexUpdate", True),
        ):
            metric = "batchMs" if operation.startswith("vector") else "ms"
            for method in METHODS:
                value = rows.get(method, {}).get("operations", {}).get(operation, {}).get(metric)
                values.append(format_value(value, seconds))
        lines.append(f"| {dataset} | " + " | ".join(values) + " |")
        all_rows.append(rows)

    average_values = []
    for operation, seconds in (
        ("vectorInsertion", False),
        ("vectorUpdate", False),
        ("indexConstruction", True),
        ("indexUpdate", True),
    ):
        metric = "batchMs" if operation.startswith("vector") else "ms"
        for method in METHODS:
            value = mean([
                rows.get(method, {}).get("operations", {}).get(operation, {}).get(metric)
                for rows in all_rows
            ])
            average_values.append(format_value(value, seconds))
    lines.append("| Three-image Avg. | " + " | ".join(average_values) + " |")

    lines.extend([
        "",
        "## Measurement boundary",
        "",
        "- 每次重复都从重新创建后端状态开始；主表报告算术平均值，JSON 保留每轮样本和样本标准差。",
        "- Violas 的 index update 是数据更新后完整重建 HDMG。",
        "- 外部数据库的同步 upsert 同时触发内部索引维护，但开源 Violas 未发布论文 Table 3 "
        "所用的独立 index-update 计时代码，因此该列保留 N/A，不能用 update 数值重复代替。",
        "- Milvus、Qdrant、Chroma 当前通过 Docker 服务访问；延迟包含本机进程间通信，"
        "与论文机器上的绝对值不应直接比较。",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
