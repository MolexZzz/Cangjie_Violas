# 实验结果

本目录给出项目报告所采用的汇总数据：

- [三图像数据集完整结果](table2-full-results.md)
- [Violas 与 Faiss 的实现及检索对比](faiss-comparison.md)
- [HDMG 参数扫描](hdmg-parameter-scan.md)
- [四范式代码上下文检索案例（Markdown）](../docs/code-context-case-study.md)
- [四范式代码上下文检索案例（JSON）](code-context-case-study.json)
- [机器可读结果](final-results.json)

完整实验使用 Caltech-101、CUB-200-2011 和 COCO 的 10% 查询集，β 的取值范围为
0.0 至 1.0，步长为 0.1。参数扫描另取各数据集前 200 个查询。Milvus、Qdrant 和
Chroma 运行于本机 Docker 服务，因此表中延迟包含进程间通信开销。

`final-results.json` 记录 β=0.5 的逐数据集结果、代码量统计、运行提交和原始汇总文件哈希。
逐查询日志及中间文件保存在本地 `results/`，未纳入版本控制。
