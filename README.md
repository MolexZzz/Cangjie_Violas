# Cangjie Violas

本项目使用仓颉实现 Violas 的主要数据结构与检索流程，用于研究同时考虑语义距离和
向量距离的近邻检索。仓颉核心包括 `VectorMap`、微簇组织、HDMG 图索引和 Mixed Search；
Python 工具负责数据预处理、基准测试与结果校验。

## 功能

- 以 `VectorMap` 和 `VectorGroup` 组织实体、成员向量、微簇及对象关系；
- 使用 HDMG 加速实体路由、微簇遍历和候选重排；
- 支持语义距离与 embedding 距离的 β 加权检索；
- 提供全量数据实验、Faiss 基线、参数扫描和维护基准；
- 提供 EAR、DDR、RER、CMP 四种实体中心代码上下文检索示例。

项目目前在 Caltech-101、CUB-200-2011 和 COCO 三个图像数据集上完成了 90% 建库、
10% 查询的实验。图像和类别文本均使用 CLIP ViT-B/32 编码。β=0.5 时的结果如下：

| 数据集 | 训练向量 | 查询向量 | Recall@3 | NDCG@3 | 延迟（ms/query） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Caltech-101 | 7,766 | 911 | 0.998536 | 0.999994 | 1.086 |
| CUB-200-2011 | 10,597 | 1,191 | 1.000000 | 1.000000 | 1.421 |
| COCO | 8,967 | 1,033 | 0.996450 | 0.999944 | 0.904 |

完整结果及其适用范围见 [实验结果](results-summary/README.md)。这些数据只反映本仓库所采用的
三数据集实验协议，不等同于原论文的六数据集结果。

代码上下文案例将仓颉源码、测试、文档、脚本和结果组织为 4 个项目实体、24 个成员和 17 条关系。
在小型案例中，EAR 的实体纯度与扁平向量检索持平，DDR、RER 和 CMP 分别补充了实现侧面、
依赖链和源码—文档配对。结果及限制见
[四范式代码上下文案例](docs/code-context-case-study.md)。

## 快速验证

### 1. 仓颉核心

仓颉工具链版本为 `1.0.4`。该步骤不需要数据集或 Python 环境。

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

### 2. 四范式代码上下文案例

该步骤需要 Python 和 MiniLM 模型；模型首次运行时由 `sentence-transformers` 下载并缓存。

```powershell
python -m pip install -r tools\requirements-benchmark.txt
python tools\run_code_context_case_study.py
```

冻结的机器可读结果位于
[`results-summary/code-context-case-study.json`](results-summary/code-context-case-study.json)。

### 3. 全量图像实验

原始数据、模型和中间向量文件体积较大，不随仓库发布。数据准备和文件校验方法见
[复现说明](docs/reproducibility.md)；建议使用 Python 3.11 复现原实验环境。完整查询池实验的
调用方式为：

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
tests/fixtures/   小型一致性测试数据
docs/             设计与复现文档
```

## 进一步阅读

- [架构与设计](docs/architecture.md)
- [实验设置](docs/experiments.md)
- [复现说明](docs/reproducibility.md)
- [已知限制](docs/limitations.md)
- [四范式代码上下文案例](docs/code-context-case-study.md)
- [工具索引](tools/README.md)
- [贡献指南](CONTRIBUTING.md)

## 许可

本项目自行编写的仓颉实现以 [MIT License](LICENSE) 发布。项目设计和一致性测试参考了采用
Apache License 2.0 的 [AgentCombo/Violas](https://github.com/AgentCombo/Violas)；
来源及第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
