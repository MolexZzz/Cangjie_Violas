# Cangjie Violas

Violas 的仓颉实现，面向语义类别距离与图像向量距离的联合检索。项目实现了 VectorMap、
微簇、HDMG 图索引和 Mixed Search，并提供可复现的全量图像基准测试工具。

## 主要特性

- 使用仓颉实现向量存储、自动聚类、对象维护和检索；
- 使用 HDMG 在微簇图上获取候选并按 mixed score 重排；
- 支持 Caltech-101、CUB-200-2011 和 COCO 的冻结 90/10 实验协议；
- 提供 Faiss、Milvus、Qdrant 和 Chroma 对比工具；
- 输出 Recall@3、NDCG@3、查询延迟、构建时间和维护开销；
- 保存数据哈希、参数、Git commit 和运行环境等复现信息。

## 已验证结果

三套图像数据均已完成完整 10% 查询池评测。β=0.5 时：

| 数据集 | 训练向量 | 查询向量 | Recall@3 | NDCG@3 | 延迟（ms/query） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Caltech-101 | 7,766 | 911 | 0.998536 | 0.999994 | 1.086 |
| CUB-200-2011 | 10,597 | 1,191 | 1.000000 | 1.000000 | 1.421 |
| COCO | 8,967 | 1,033 | 0.996450 | 0.999944 | 0.904 |

完整的冻结结果、Faiss 对比和参数扫描见 [results-summary](results-summary/README.md)。

## 快速开始

要求仓颉工具链 `1.0.4`。

```powershell
cd cj_core
cjpm build
cjpm test --no-color
"2" | cjpm run
```

上述命令依次执行构建、标准单元测试和核心集成回归。

运行最小示例：

```powershell
cd cj_core
"0" | cjpm run
```

## 全量实验

原始数据、模型和冻结 embedding 不进入 Git。准备方法见
[复现实验说明](docs/reproducibility.md)。

```powershell
.\tools\run_image_full_suite.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/python-paper-90-10/reproduction" `
  -MaxQueries 0 `
  -CangjieHeapSize 2GB `
  -DatabaseExecutionMode service `
  -LiveOutput
```

准确率—延迟参数扫描：

```powershell
python tools\run_accuracy_parameter_scan.py --queries 200 --beta 0.5
```

## 仓库结构

```text
cj_core/          仓颉核心、Benchmark 和测试
tools/            数据准备、评测、校验与汇总工具
manifests/        小样例和冻结实验清单
results-summary/  可提交、可审阅的精选结果
docs/             架构、实验方法、复现说明和已知限制
```

数据、模型、完整 artifact、原始日志和构建产物保存在本地并由 Git 忽略。

## 文档

- [架构与设计](docs/architecture.md)
- [实验方法与结果](docs/experiments.md)
- [复现实验](docs/reproducibility.md)
- [已知限制](docs/limitations.md)
- [工具索引](tools/README.md)
- [参与贡献](CONTRIBUTING.md)

## 项目状态

本仓库是研究型实现，不以替代 Faiss 或通用向量数据库为目标。HDMG 是实际图索引；
rep/single 接口目前仍采用精确遍历快照。外部数据库通过 Python SDK 进程适配器接入。

## 许可

本项目采用 [MIT License](LICENSE)。
