# 复现说明

## 运行环境

实验使用 Windows 11、仓颉工具链 1.0.4 和 Python 3.11。运行 Milvus、Qdrant 或
Chroma 时还需要 Docker。完整数据集实验建议为仓颉进程设置 2 GB 堆内存。

安装 Python 依赖：

```powershell
python -m pip install -r tools\requirements-benchmark.txt
python -m pip install -r tools\requirements-external-db.txt
```

## 输入文件

项目约定以下本地目录：

```text
dataset/                         原始数据
model/                           CLIP 模型缓存
artifacts/python-paper-90-10/   实验输入
results/                         原始输出
```

这些目录不进入 Git。每个数据集的实验输入包含：

```text
<dataset>-full/
├── manifest.json
├── train.jsonl
├── queries.jsonl
├── ground_truth.json
└── cangjie_input.txt
```

本次结果所用 `cangjie_input.txt` 的 SHA-256 记录在
[`manifests/release-artifacts.json`](../manifests/release-artifacts.json)。

## 基础检查

```powershell
cd cj_core
cjpm build
cjpm test --no-color
"2" | cjpm run
```

分别检查三套实验输入：

```powershell
python tools\verify_shared_artifact.py artifacts\python-paper-90-10\caltech-full
python tools\verify_shared_artifact.py artifacts\python-paper-90-10\cub-full
python tools\verify_shared_artifact.py artifacts\python-paper-90-10\coco-full
```

## 完整查询集

`MaxQueries 0` 表示不截断查询集：

```powershell
.\tools\run_image_full_suite.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/python-paper-90-10/reproduction" `
  -MaxQueries 0 `
  -CangjieHeapSize 2GB `
  -DatabaseExecutionMode service `
  -LiveOutput
```

## 参数扫描

```powershell
python tools\run_accuracy_parameter_scan.py `
  --queries 200 `
  --beta 0.5 `
  --cangjie-heap-size 2GB
```

脚本先构建仓颉工程，再调用 release 可执行文件，并生成 JSON 与 Markdown。查询顺序由
输入文件确定；准确率可在相同输入上复核，延迟会随机器负载变化。

## Faiss 基线

```powershell
.\tools\run_faiss_and_maintenance.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/faiss-and-maintenance/reproduction" `
  -MaxQueries 0 `
  -MutationCount 200
```

该脚本同时运行仓库中保留的维护基准，但维护数据不列入 `results-summary/`。

## 汇总文件校验

```powershell
python tools\verify_release_bundle.py
```

校验器检查结果文件结构、数据集数量、查询数量、指标范围和输入文件哈希格式。
