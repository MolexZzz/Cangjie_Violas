# 复现说明

## 运行环境

正式实验使用 Windows 11、仓颉工具链 1.0.4 和 Python 3.11。GitHub Actions 使用
Python 3.13 编译检查工具脚本；为贴近原始运行环境，完整数据实验仍建议使用 Python 3.11。
运行 Milvus、Qdrant 或 Chroma 时还需要 Docker。完整数据集实验建议为仓颉进程设置
2 GB 堆内存。

常规 benchmark 与四范式案例安装：

```powershell
python -m pip install -r tools\requirements-benchmark.txt
```

重新下载图片并生成 CLIP artifact 时额外安装：

```powershell
python -m pip install -r tools\requirements-data-generation.txt
```

只有运行真实外部数据库适配器时才安装：

```powershell
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
仓库不托管这些约 100～140 MB 的单数据集输入，第三方需按下述命令重新生成，并使用清单哈希
核对是否与冻结实验边界一致。

## 四范式代码上下文案例

该案例不需要图像数据：

```powershell
python tools\run_code_context_case_study.py
```

脚本会生成 MiniLM 向量、调用仓颉 `codecase` 入口，并将原始结果写入被 Git 忽略的
`results/code-context-case-study/`。提交的冻结结果位于
[`results-summary/code-context-case-study.json`](../results-summary/code-context-case-study.json)。

## 下载数据

以下命令下载并校验 Caltech-101、CUB-200-2011，并准备固定的 COCO-10k 清单及图片：

```powershell
python tools\download_image_datasets.py caltech cub coco
```

下载结果写入 `dataset/`，不会进入 Git。COCO 图片来自固定 URL 表，少数源站链接若暂时失效，
脚本会记录失败项；重新执行相同命令只会重试缺失对象，不会替换样本。

## 生成冻结实验输入

Caltech-101：

```powershell
python tools\paper_artifact.py `
  --dataset caltech `
  --source-kind folder-images `
  --source dataset/caltech-101/101_ObjectCategories `
  --output-dir artifacts/python-paper-90-10/caltech-full `
  --key-vector-source clip-text `
  --model ViT-B/32 `
  --seed 42 `
  --test-size 0.1 `
  --max-queries 0 `
  --full-verified
```

CUB-200-2011：

```powershell
python tools\paper_artifact.py `
  --dataset cub `
  --source-kind folder-images `
  --source dataset/CUB_200_2011/images `
  --output-dir artifacts/python-paper-90-10/cub-full `
  --key-vector-source clip-text `
  --model ViT-B/32 `
  --seed 42 `
  --test-size 0.1 `
  --max-queries 0 `
  --full-verified
```

COCO-10k：

```powershell
python tools\paper_artifact.py `
  --dataset coco `
  --source-kind coco-json `
  --source dataset/coco/coco_dataset_10000.json `
  --image-root dataset/coco `
  --output-dir artifacts/python-paper-90-10/coco-full `
  --key-vector-source clip-text `
  --model ViT-B/32 `
  --seed 42 `
  --test-size 0.1 `
  --max-queries 0 `
  --full-verified
```

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
