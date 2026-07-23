"""Build an evidence-backed Cangjie Violas/Faiss comparison report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def average(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loc", type=Path, required=True)
    parser.add_argument("--faiss", type=Path, nargs="+", required=True)
    parser.add_argument("--table2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loc = json.loads(args.loc.read_text(encoding="utf-8"))
    faiss_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.faiss]
    table2 = json.loads(args.table2.read_text(encoding="utf-8"))
    beta_half = next(row for row in table2["rows"] if abs(row["beta"] - 0.5) < 1e-9)

    faiss_by_method: dict[str, list[dict]] = {}
    for payload in faiss_payloads:
        for row in payload["results"]:
            faiss_by_method.setdefault(row["method"], []).append(row)

    cangjie_core = loc["scopes"]["Cangjie storage core"]["sourceLines"]
    cangjie_all = loc["scopes"]["Cangjie all src"]["sourceLines"]
    faiss_cpu = loc["scopes"]["Faiss CPU library"]["sourceLines"]
    faiss_all = loc["scopes"]["Faiss all library"]["sourceLines"]

    lines = [
        "# Cangjie Violas 与 Faiss 正式对比",
        "",
        "## 功能范围与代码规模",
        "",
        "| 项目 | 仓颉 Violas | Faiss |",
        "| --- | --- | --- |",
        "| 主要定位 | 语义类别距离与图片 embedding 距离联合检索 | 通用高性能向量相似度检索库 |",
        "| 当前索引 | HDMG；rep/single 当前仍是 exact-scan snapshot | Flat、IVF、HNSW、PQ 等成熟 CPU/GPU 索引 |",
        "| Mixed score | 原生支持 β 加权语义/embedding 距离 | 不原生支持，需要应用层重排 |",
        "| 数据维护 | 稳定 recordId 插入、更新、删除；变更后 HDMG 失效/重建 | 各索引能力不同；实验中对通用更新/删除采用重建 |",
        "| 向量精度 | 当前核心为 Float64 | 实验输入与常用 CPU 索引为 Float32 |",
        f"| 核心源码行（去纯注释） | storage `{cangjie_core}` 行；全部 src `{cangjie_all}` 行 | "
        f"CPU library `{faiss_cpu}` 行；CPU+GPU `{faiss_all}` 行 |",
        "| 代码规模解释 | 研究原型，覆盖 Violas 所需语义组织与实验框架 | 成熟工业库，覆盖量化、SIMD、多种索引、CPU/GPU 与广泛平台 |",
        "",
        "代码行数只能描述实现范围，不能单独证明性能或工程质量。",
        "",
        "## 实验性能（不同任务，分开报告）",
        "",
        "### 仓颉 Violas：β=0.5 的三图像数据集平均 Mixed Search",
        "",
        "| Method | Mixed Recall@3 | Mixed NDCG@3 | Mean latency (ms/query) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("Violas", "w/o HDMG"):
        row = beta_half["methods"][method]
        lines.append(
            f"| {method} | {row['recallAtK']:.4f} | {row['ndcgAtK']:.4f} | "
            f"{row['latencyMs']:.3f} |"
        )

    lines.extend([
        "",
        "### Faiss：三图像数据集平均纯 embedding cosine 检索",
        "",
        "| Index | Recall@3 | NDCG@3 | Mean latency (ms/query) | Build (ms) | Index (MiB) | Peak RSS (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method, label in (
        ("exact", "IndexFlatIP"),
        ("ivf", "IndexIVFFlat"),
        ("hnsw", "IndexHNSWFlat"),
    ):
        rows = faiss_by_method[method]
        lines.append(
            f"| {label} | {average([r['recallAtK'] for r in rows]):.4f} | "
            f"{average([r['ndcgAtK'] for r in rows]):.4f} | "
            f"{average([r['latencyMs']['mean'] for r in rows]):.3f} | "
            f"{average([r['buildMs'] for r in rows]):.2f} | "
            f"{average([r['indexBytes'] for r in rows]) / 1048576.0:.2f} | "
            f"{average([r['peakRssBytes'] for r in rows if r['peakRssBytes'] is not None]) / 1048576.0:.2f} |"
        )

    lines.extend([
        "",
        "Violas 表使用 mixed ground truth，Faiss 表使用纯 embedding exact ground truth，"
        "二者目标函数不同，不能把两张表的 Recall 或 latency 直接解释成同一算法的胜负。"
        "Faiss 在这里承担的是底层向量索引基线；若要做严格端到端对比，应让 Faiss 先召回候选，"
        "再使用与 Violas 完全相同的 mixed score 重排。",
        "",
        "## 可复现信息",
        "",
        f"- 仓颉仓库 commit：`{loc['repository']['gitCommit']}`",
        f"- Faiss commit：`{loc['faiss']['gitCommit']}`",
        f"- LOC 规则：`{loc['rules']['sourceLines']}`",
        f"- Faiss 数据集：{', '.join(payload['dataset'] for payload in faiss_payloads)}",
        f"- Faiss 重复次数：{faiss_payloads[0]['measurement']['repetitions']}；"
        f"预热查询：{faiss_payloads[0]['measurement']['warmupQueries']}",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
