"""Build an evidence-backed Cangjie Violas/Faiss comparison report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from count_source_lines import source_line_count


ROOT = Path(__file__).resolve().parents[1]


def average(values: list[float]) -> float:
    return statistics.fmean(values)


def module_source_lines(root: Path, files: list[str]) -> int:
    return sum(source_line_count(root / relative)[2] for relative in files)


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
    faiss_root = Path(loc["faiss"]["path"])

    faiss_by_method: dict[str, list[dict]] = {}
    for payload in faiss_payloads:
        for row in payload["results"]:
            faiss_by_method.setdefault(row["method"], []).append(row)

    module_scopes = [
        {
            "method": "仓颉 Violas HDMG/mixed",
            "root": ROOT,
            "files": [
                "cj_core/src/storage/vectormap.cj",
                "cj_core/src/storage/hdmg.cj",
                "cj_core/src/storage/mixed_scoring.cj",
            ],
            "boundary": "HDMG 构建/遍历、w/o HDMG、候选重排和 mixed score",
            "shared": "VectorGroup、向量距离工具、KMeans 聚类",
        },
        {
            "method": "Faiss IndexFlatIP",
            "root": faiss_root,
            "files": [
                "faiss/IndexFlat.cpp",
                "faiss/IndexFlat.h",
                "faiss/IndexFlatCodes.cpp",
                "faiss/IndexFlatCodes.h",
            ],
            "boundary": "Flat 精确索引及其连续向量编码存储",
            "shared": "Index 基类、公共距离计算和 SIMD 内核",
        },
        {
            "method": "Faiss IndexIVFFlat",
            "root": faiss_root,
            "files": [
                "faiss/IndexIVF.cpp",
                "faiss/IndexIVF.h",
                "faiss/IndexIVFFlat.cpp",
                "faiss/IndexIVFFlat.h",
            ],
            "boundary": "IVF 基础流程和 IVFFlat 编码/扫描",
            "shared": "Flat 量化器、Clustering、InvertedLists、距离内核",
        },
        {
            "method": "Faiss IndexHNSWFlat",
            "root": faiss_root,
            "files": [
                "faiss/IndexHNSW.cpp",
                "faiss/IndexHNSW.h",
                "faiss/impl/HNSW.cpp",
                "faiss/impl/HNSW.h",
            ],
            "boundary": "HNSW 索引封装、建图和图搜索",
            "shared": "IndexFlat 向量存储、公共距离计算和 SIMD 内核",
        },
    ]

    lines = [
        "# 仓颉 Violas 与 Faiss 检索实现对比",
        "",
        "## 可比代码范围",
        "",
        "| 实现 | 主实现文件数 | 主实现源码行 | 本列包含 | 未计入的共享依赖 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for scope in module_scopes:
        lines.append(
            f"| {scope['method']} | {len(scope['files'])} | "
            f"{module_source_lines(scope['root'], scope['files'])} | "
            f"{scope['boundary']} | {scope['shared']} |"
        )

    lines.extend([
        "",
        "这里统计的是本实验实际使用索引的**主实现模块**，而不是整个仓库。"
        "所有数字均为去除空行和纯注释后的源码行。公共依赖不重复计入某一种索引，"
        "因此这些数字用于比较实现规模，不表示完整可独立编译的代码量。",
        "",
        "仓颉 `vectormap.cj` 同时包含部分 CRUD、关系和上下文接口；Faiss 的索引文件也包含"
        "同系列变体。因此主实现源码行仍是模块级近似值，不应被解释为算法复杂度或代码质量排名。",
        "Faiss 使用 C++，表中同时统计 `.h` 接口声明和 `.cpp` 实现；仓颉没有同样的头文件分离方式，"
        "因此该表也不能用于评价两种编程语言谁更精简。",
        "",
        "## 功能边界",
        "",
        "| 项目 | 仓颉 Violas HDMG | Faiss Flat / IVF / HNSW |",
        "| --- | --- | --- |",
        "| 检索目标 | β 加权的语义距离与 embedding 距离 | embedding 向量相似度 |",
        "| 候选获取 | 实体路由、微簇和 HDMG 图遍历 | 精确扫描或 ANN 索引 |",
        "| Mixed score | 原生完成候选 mixed 重排 | 索引本身不支持，需应用层增加 |",
        "| 动态维护 | 数据变化后使 HDMG 失效并重建 | 能力随索引而异，统一实验采用重建 |",
        "| 当前向量精度 | Float64 | Float32 |",
        "",
        "## 实验性能",
        "",
        "### 仓颉 Violas：β=0.5 的三图像数据集平均 Mixed Search",
        "",
        "| Method | Mixed Recall@3 | Mixed NDCG@3 | Mean latency (ms/query) |",
        "| --- | ---: | ---: | ---: |",
    ])
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
        "## 主实现文件清单",
        "",
    ])
    for scope in module_scopes:
        lines.append(f"### {scope['method']}")
        lines.append("")
        for path in scope["files"]:
            lines.append(f"- `{path}`")
        lines.append("")

    lines.extend([
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
