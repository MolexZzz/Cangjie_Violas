# 复现实验

## 环境

- Windows 11 或等价的 PowerShell 环境；
- 仓颉工具链 1.0.4；
- Python 3.11 及以上；
- 全量外部数据库实验需要 Docker；
- 仓颉全量运行建议设置 `cjHeapSize=2GB`。

安装 Python 依赖：

```powershell
python -m pip install -r tools\requirements-benchmark.txt
python -m pip install -r tools\requirements-external-db.txt
```

## 数据目录

```text
dataset/                         原始数据
model/                           CLIP 模型缓存
artifacts/python-paper-90-10/   冻结实验输入
results/                         原始运行输出
```

这些目录不进入 Git。冻结输入应包含：

```text
<dataset>-full/
├── manifest.json
├── train.jsonl
├── queries.jsonl
├── ground_truth.json
└── cangjie_input.txt
```

发布实验使用的 `cangjie_input.txt` SHA-256 记录在
[`manifests/release-artifacts.json`](../manifests/release-artifacts.json)。

## 基础验证

```powershell
cd cj_core
cjpm build
cjpm test --no-color
"2" | cjpm run
```

校验冻结输入：

```powershell
python tools\verify_shared_artifact.py artifacts\python-paper-90-10\caltech-full
python tools\verify_shared_artifact.py artifacts\python-paper-90-10\cub-full
python tools\verify_shared_artifact.py artifacts\python-paper-90-10\coco-full
```

## 仓颉全量评测

`MaxQueries 0` 表示使用完整查询池：

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

扫描工具会先构建仓颉工程，再直接运行 release 可执行文件，输出 JSON 和 Markdown。查询顺序来自
冻结 artifact，因此同一输入上的准确率可重复；延迟仍会受机器负载影响。

## Faiss 与维护实验

```powershell
.\tools\run_faiss_and_maintenance.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/faiss-and-maintenance/reproduction" `
  -MaxQueries 0 `
  -MutationCount 200
```

## 发布结果校验

```powershell
python tools\verify_release_bundle.py
```

校验器检查冻结结果 schema、数据集数量、查询数量、指标范围和 artifact 哈希格式。
