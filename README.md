# Cangjie Violas

Violas 的仓颉实现，以及用于复现实验和横向对比的 Benchmark 工具。

当前仓库已经能够：

- 构建并查询仓颉 HDMG；
- 在 Caltech-101、CUB-200-2011 和 COCO 完整图像数据上运行；
- 输出 Mixed Recall@3、Mixed NDCG@3 和查询延迟；
- 对比 w/o HDMG、Faiss、Milvus、Qdrant 和 Chroma；
- 保存 JSON、Markdown、日志和断点续跑状态；
- 运行仓颉标准单元测试与核心集成回归。

数据集、模型、预处理 artifact、实验结果、Python 参考实现和构建产物均保留在本地，不提交 Git。

## 仓库结构

```text
.
├── cj_core/          # 可编译的仓颉工程
├── tools/            # 数据准备、Benchmark、校验与结果汇总脚本
├── manifests/        # 小规模固定测试输入与清单
├── photo-data/       # COCO 图片清单生成工具
├── docs/
│   ├── guides/       # 使用方法与实验协议
│   ├── reports/      # Review、实验对齐与阶段报告
│   └── plans/        # 科研实践计划与后续工程计划
├── dataset/          # 本地原始数据，Git 忽略
├── model/            # 本地模型缓存，Git 忽略
├── artifacts/        # 冻结实验输入，Git 忽略
└── results/          # 实验输出，Git 忽略
```

完整文档导航见 [docs/README.md](docs/README.md)。

## 快速验证

需要仓颉工具链 `1.0.4`。在 PowerShell 中运行：

```powershell
cd cj_core
cjpm build
cjpm test --no-color
"2" | cjpm run
```

三条命令分别验证工程构建、标准单元测试和核心集成回归。

## 图像全量实验

冻结 artifact 已准备好时，可按数据集运行完整查询与三个数据库：

```powershell
.\tools\run_image_full_suite.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/python-paper-90-10/full-run" `
  -MaxQueries 0 `
  -CangjieHeapSize 2GB `
  -DatabaseExecutionMode service `
  -LiveOutput `
  -Resume
```

`MaxQueries 0` 表示使用完整的 10% 查询池。Milvus、Qdrant 和 Chroma 需要提前启动本地服务。

## Faiss 与代码量

```powershell
.\tools\run_faiss_and_maintenance.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/faiss-and-maintenance" `
  -MaxQueries 0 `
  -MutationCount 200
```

Faiss 是纯 embedding 向量索引基线；Violas 是语义距离与 embedding 距离联合检索。报告时应同时
说明功能范围，不能只用代码行数或单项延迟直接评价两者。

## 本地文件

- 数据与模型位置：[docs/guides/data-and-models.md](docs/guides/data-and-models.md)
- 冻结实验协议：[docs/guides/python-paper-90-10-protocol.md](docs/guides/python-paper-90-10-protocol.md)
- 外部数据库：[docs/guides/external-database-benchmark.md](docs/guides/external-database-benchmark.md)
- Faiss、维护实验与 LOC：[docs/guides/faiss-maintenance-loc-benchmark.md](docs/guides/faiss-maintenance-loc-benchmark.md)

本地 `violas_python/` 只用于参考，不属于仓颉交付代码，也不会出现在当前 Git 主分支中。
