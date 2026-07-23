# 架构与设计

## 设计范围

Cangjie Violas 用统一的数据结构表示语义类别、微簇和实例向量。查询阶段先从 HDMG 图中
取得候选微簇，再依据语义距离和 embedding 距离的加权结果返回近邻。本仓库同时保留了
精确检索路径，用于正确性校验和消融实验。

## 代码组织

| 模块 | 职责 |
| --- | --- |
| `storage/vectorgroup.cj` | 保存代表向量、实例向量及元数据 |
| `storage/vectormap.cj` | 数据组织、对象维护、检索入口和 HDMG 生命周期 |
| `storage/clustering.cj` | 确定性 KMeans 实现 |
| `storage/hdmg.cj` | HDMG 构建和查询配置 |
| `storage/mixed_scoring.cj` | 混合距离与实体 key 处理 |
| `bench/paper_protocol.cj` | 90/10 实验输入的评测协议 |
| `tools/` | 数据生成、外部后端调用、结果汇总与校验 |

## 数据模型

`VectorMap` 以语义 key 为单位管理 `VectorMapEntry`。一个 entry 可包含多个
`VectorGroup`；每个 group 对应一个微簇，保存代表向量、实例向量及其描述信息。
记录通过稳定的 `recordId` 完成插入、更新和删除。

## 距离函数

候选实例按下式排序：

```text
mixed_distance = (1 - β) × embedding_distance + β × semantic_distance
```

β 的取值范围为 `[0, 1]`。查询未提供语义向量时，只计算 embedding 距离。

## HDMG 查询

HDMG 以微簇为节点，同时建立 embedding 邻边和语义邻边。一次查询包括三个步骤：

1. 根据查询向量和语义向量确定入口；
2. 遍历图并收集候选微簇；
3. 计算候选实例的混合距离并返回 Top-K。

构建参数和查询参数分别由 `HdmgBuildConfig` 与 `HdmgSearchConfig` 保存，参数选择依据见
[实验设置](experiments.md)。

## 索引生命周期

插入、更新、删除以及 key vector 的变化都会递增数据版本并使现有索引失效。如果查询时
索引尚未建立或版本已经过期，系统改用精确候选路径，并在统计信息中标记本次回退。

为兼容已有调用方式，部分容器仍然公开可变。直接修改 `data`、`groups` 或 `vectors` 后
必须调用 `markDirty()`；一般情况下应使用现有 CRUD 接口。

## 扩展约束

新的距离函数应放在 `storage/utils.cj` 并配套单元测试。新的索引实现应独立管理配置和
生命周期，避免继续增加 `VectorMap` 的职责。外部数据库后端通过统一的 Python 适配接口
读取同一组实验输入；若改变数据划分或真值定义，应使用新的协议标识。
