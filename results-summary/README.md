# 冻结实验结果

本目录保存适合进入 Git 的精选结果。大型 artifact、逐查询日志和数据库数据仍保存在本地
`results/`，不提交到仓库。

## 发布结果

- [Table 2 三图像全量结果](table2-full-results.md)
- [Faiss 检索与代码量对比](faiss-comparison.md)
- [Table 3 数据与索引维护实验](table3-maintenance.md)
- [HDMG 准确率—延迟参数扫描](hdmg-parameter-scan.md)
- [最终指标 JSON](final-results.json)

`final-results.json` 固定了数据规模、β=0.5 的逐数据集结果、三数据集平均值、代码量对比、
性能优化结果和原始结果 SHA-256。

## 结果状态

- 全量图像实验：完成；
- 查询范围：三个数据集完整 10% 查询池；
- β 扫描：0.0–1.0，步长 0.1；
- 外部后端：Milvus、Qdrant、Chroma；
- Faiss：Flat、IVF、HNSW；
- 参数扫描：5 组 HDMG 配置，每数据集固定 200 个查询。

原始延迟来自同一工作站，但数据库运行在 Docker service 模式。准确率结果可用于算法对比，
绝对延迟仅用于说明当前环境下的表现。
