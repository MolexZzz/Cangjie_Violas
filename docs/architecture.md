# 架构与设计

## 目标

Cangjie Violas 用统一的数据结构组织实体类别、微簇和实例向量，并在查询时联合考虑语义距离与
embedding 距离。实现重点是论文算法复现、可观察性和跨后端公平评测。

## 模块

| 模块 | 职责 |
| --- | --- |
| `storage/vectorgroup.cj` | 保存代表向量、实例向量及其元数据 |
| `storage/vectormap.cj` | 数据组织、对象维护、检索入口和 HDMG 生命周期 |
| `storage/clustering.cj` | 确定性 KMeans 与 Python/sklearn 行为对齐 |
| `storage/hdmg.cj` | HDMG 构建和搜索配置 |
| `storage/mixed_scoring.cj` | mixed score 与实体 key 处理 |
| `bench/paper_protocol.cj` | 冻结 90/10 输入的评测协议 |
| `tools/` | 数据生成、外部后端、结果汇总和校验 |

## 数据模型

`VectorMap` 按语义 key 管理 `VectorMapEntry`。每个 entry 包含一个或多个 `VectorGroup`，
每个 group 表示一个微簇并保存代表向量、实例向量和描述信息。稳定 `recordId` 用于对象插入、
更新和删除。

## Mixed Search

对候选实例使用下式排序：

```text
mixed_distance = (1 - β) × embedding_distance + β × semantic_distance
```

当查询没有语义向量时，搜索退化为 embedding 距离。β 被限制在 `[0, 1]`。

## HDMG

HDMG 以微簇为节点，组合 embedding 邻边和语义邻边。查询包含三个阶段：

1. 根据查询向量和语义向量选择入口；
2. 在图上遍历并收集候选微簇；
3. 对候选微簇中的实例计算 mixed score 并返回 Top-K。

构建和搜索参数由 `HdmgBuildConfig`、`HdmgSearchConfig` 管理。论文默认参数及其扫描结果记录在
[实验文档](experiments.md)。

## 索引生命周期

插入、更新、删除和 key vector 变化会增加数据版本并使索引失效。HDMG 查询检测到索引未构建
或版本过期时，会执行有明确统计标记的精确候选回退。

公开可变容器是当前兼容性设计的一部分。直接修改 `data`、`groups` 或 `vectors` 的调用方必须调用
`markDirty()`；新代码应优先使用受控 CRUD API。

## 扩展点

- 新距离函数应集中加入 `storage/utils.cj` 并补充单元测试；
- 新索引策略应拥有独立配置和生命周期，不应继续扩大 `VectorMap`；
- 新数据库后端应实现统一 Python 适配接口并读取同一冻结 artifact；
- 新实验协议必须使用独立的协议标识，避免与现有结果混用。
