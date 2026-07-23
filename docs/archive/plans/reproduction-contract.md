# 历史计划：并行开发公共契约

> 本文是开发阶段记录，不属于发布说明；当前结论以正式文档和冻结结果为准。

该契约在成员 A、B 开始编码前冻结。除非双方同意，不在各自分支修改。

## 1. 文件边界

| 范围 | Owner |
|---|---|
| `cj_core/src/storage/vectormap.cj` facade、`index/cluster*`、`index/mixed*` | A |
| `cj_core/src/index/hdmg*`、通用 benchmark 指标/配置 | B |
| `cj_core/src/bench/text/**`、文本核心测试 | A |
| `cj_core/src/bench/image/**`、HDMG/benchmark 测试 | B |
| `tools/text/**`、`manifests/text/**`、`results/text/**` | A |
| `tools/image/**`、`manifests/image/**`、`results/image/**` | B |
| 公共接口/schema | 7 月 21 日共同冻结，个人分支不得单独修改 |
| `violas_python/**` | 冻结，任何人不得修改 |

## 2. 统一记录 ID

每个向量必须有稳定的 `recordId`。新 artifact 使用 exporter 生成的 ID；当前 TXT 回归集使用：

```text
<dataset>/<folder>/<vector-line-index>
```

结果比较只使用 `recordId`，不使用数组下标、数据库内部 ID 或临时 UUID。

## 3. Backend 输入输出

所有 native/exact/external backend 对 benchmark 暴露相同语义：

```text
build(records, config) -> BuildStats
search(query, topK) -> Array<BackendHit>
stats() -> BackendStats
close() -> Unit
```

`BackendHit` 固定包含：

```text
recordId: String
key: String
distance: Float64
```

distance 统一为“越小越相似”；cosine similarity 必须转换为 `1 - similarity`。

## 4. Query 与 Ground Truth

每个 query 固定包含：

```text
queryId, sourceRecordId, trueKey, vector, keyVector
```

Ground truth 固定为：

```text
queryId -> ordered recordIds
```

成员 A 独立准备 news20、ohsumed、yahoo 的 query/GT；成员 B 独立准备 caltech、cub、coco
的 query/GT。两人都先用自己负责的数据集 sample 开发，再各自扩展到 full，不等待对方的
exporter 或算法实现。

## 5. Result JSON 最小字段

```text
schemaVersion
backend
datasetHash
configHash
queries[]: queryId, hits[]
metrics: recallAtK, ndcgAtK
timing: buildMs, meanMs, p50Ms, p95Ms
```

A、B 分别对自己的三个数据集产生符合契约的 Python、仓颉、Faiss/数据库 Result JSON。
通用 reporter 可使用 Fake Result 开发；最终汇总只读取两人的结果文件，不调用对方的
算法或数据流水线。
