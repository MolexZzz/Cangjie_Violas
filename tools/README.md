# 工具导航

## 实验入口

| 工具 | 用途 |
| --- | --- |
| `run_image_full_suite.ps1` | 分数据集运行仓颉 Violas 与外部数据库完整实验 |
| `run_faiss_and_maintenance.ps1` | 运行 Faiss、维护实验和 LOC 统计 |
| `run_cangjie_benchmark.py` | 从 Python 侧驱动仓颉 Benchmark |
| `run_accuracy_parameter_scan.py` | 在三图像数据上扫描 HDMG 准确率—延迟参数 |

## 数据与实验输入

| 工具 | 用途 |
| --- | --- |
| `download_image_datasets.py` | 下载/整理图像数据 |
| `paper_artifact.py` | 生成 `python-paper-90-10` 实验输入 |
| `precomputed_artifacts.py` | 读取旧版预计算输入 |
| `verify_shared_artifact.py` | 校验不同后端是否共享相同输入 |

## 基线与外部数据库

| 工具 | 用途 |
| --- | --- |
| `run_faiss_baseline.py` | Faiss Flat、IVF、HNSW 基线 |
| `external_db_benchmark.py` | Milvus、Qdrant、Chroma 统一适配器 |
| `run_maintenance_benchmark.py` | 200 条插入、更新、删除及索引维护 |

## 校验与诊断

| 工具 | 用途 |
| --- | --- |
| `compare_python_cangjie.py` | 使用本地 Violas Python 参考实现进行仓颉一致性检查 |
| `verify_release_bundle.py` | 校验 Git 跟踪的精选结果与 release artifact 清单 |

## 汇总

| 工具 | 用途 |
| --- | --- |
| `summarize_image_full_results.py` | 单数据集实验汇总 |
| `summarize_image_paper_average.py` | 三图像数据集平均表 |
| `summarize_cangjie_faiss_comparison.py` | 仓颉—Faiss 功能、LOC、性能对比 |
| `count_source_lines.py` | 基于 Git 已跟踪文件统计源码行数 |

实验输出应写入 `results/`，不要写回 `tools/` 或仓颉源码目录。
