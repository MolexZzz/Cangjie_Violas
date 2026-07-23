# 项目文档

本目录仅保留对使用者和评审者有直接价值的正式文档。根目录 README 提供快速入口。

## 核心文档

- [架构与设计](architecture.md)：模块边界、数据模型、索引生命周期与扩展点；
- [实验方法与结果](experiments.md)：数据、指标、基线、全量结果与参数扫描；
- [复现实验](reproducibility.md)：环境、数据准备、命令、输出和校验方法；
- [已知限制](limitations.md)：实现边界、性能风险和结果解释范围。

## 专项指南

- [仓颉工程说明](guides/cangjie-core.md)
- [数据与模型](guides/data-and-models.md)
- [冻结 90/10 协议](guides/python-paper-90-10-protocol.md)
- [外部数据库](guides/external-database-benchmark.md)
- [Faiss、维护实验与代码量](guides/faiss-maintenance-loc-benchmark.md)

历史计划和阶段审计不属于发布文档。发布结论以本页列出的核心文档和
`results-summary/` 中的冻结结果为准。
