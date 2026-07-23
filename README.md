# Cangjie Violas

本项目使用仓颉实现 Violas 的主要数据结构与检索流程，用于研究同时考虑语义距离和
向量距离的近邻检索。仓颉核心包括 `VectorMap`、微簇组织、HDMG 图索引和 Mixed Search；
Python 工具负责数据预处理、基准测试与结果校验。

项目目前在 Caltech-101、CUB-200-2011 和 COCO 三个图像数据集上完成了 90% 建库、
10% 查询的实验。图像和类别文本均使用 CLIP ViT-B/32 编码。β=0.5 时的结果如下：

| 数据集 | 训练向量 | 查询向量 | Recall@3 | NDCG@3 | 延迟（ms/query） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Caltech-101 | 7,766 | 911 | 0.998536 | 0.999994 | 1.086 |
| CUB-200-2011 | 10,597 | 1,191 | 1.000000 | 1.000000 | 1.421 |
| COCO | 8,967 | 1,033 | 0.996450 | 0.999944 | 0.904 |

完整结果及其适用范围见 [实验结果](results-summary/README.md)。这些数据只反映本仓库所采用的
三数据集实验协议，不等同于原论文的六数据集结果。

## 构建与测试

仓颉工具链版本为 `1.0.4`。

```powershell
cd cj_core
cjpm build
cjpm test --no-color
"2" | cjpm run
```

`cjpm test` 当前运行 9 项测试；输入 `2` 执行核心集成回归。最小示例可通过下列命令运行：

```powershell
cd cj_core
"0" | cjpm run
```

## 实验复现

原始数据、模型和中间向量文件体积较大，不随仓库发布。数据准备和文件校验方法见
[复现说明](docs/reproducibility.md)。完整查询池实验的调用方式为：

```powershell
.\tools\run_image_full_suite.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/python-paper-90-10/reproduction" `
  -MaxQueries 0 `
  -CangjieHeapSize 2GB `
  -DatabaseExecutionMode service `
  -LiveOutput
```

HDMG 参数扫描使用每个数据集固定的前 200 个查询：

```powershell
python tools\run_accuracy_parameter_scan.py --queries 200 --beta 0.5
```

## 目录

```text
cj_core/          仓颉实现与测试
tools/            数据处理和实验脚本
manifests/        实验输入清单
results-summary/  汇总结果
docs/             设计与复现文档
```

## 进一步阅读

- [架构与设计](docs/architecture.md)
- [实验设置](docs/experiments.md)
- [复现说明](docs/reproducibility.md)
- [已知限制](docs/limitations.md)
- [工具索引](tools/README.md)
- [贡献指南](CONTRIBUTING.md)

## 许可

本项目以 [MIT License](LICENSE) 发布。
