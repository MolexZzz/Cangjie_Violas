# HDMG 准确率—延迟参数扫描

扫描使用 Caltech-101、CUB-200-2011 和 COCO 的既定实验输入。每个数据集取前 200 个查询，
β=0.5；Recall@3 和 NDCG@3 均以 exact mixed search 为真值。

| 配置 | 平均 Recall@3 | 最低 Recall@3 | 平均 NDCG@3 | 平均延迟（ms/query） | 平均候选池 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse-graph | 0.993889 | 0.985000 | 0.999810 | 1.049 | 74.9 |
| small-candidate-pool | 0.996667 | 0.991667 | 0.999967 | 0.894 | 56.1 |
| paper-default | 0.998333 | 0.996667 | 0.999978 | 1.108 | 92.1 |
| wide-candidate-pool | 0.999444 | 0.998333 | 0.999985 | 1.520 | 142.2 |
| dense-graph | 0.998333 | 0.996667 | 0.999978 | 1.211 | 111.8 |

## 逐数据集结果

| 数据集 | 配置 | Recall@3 | NDCG@3 | 延迟（ms/query） | 候选池 |
| --- | --- | ---: | ---: | ---: | ---: |
| caltech | sparse-graph | 0.985000 | 0.999483 | 1.053 | 75.8 |
| cub | sparse-graph | 1.000000 | 1.000000 | 1.211 | 64.1 |
| coco | sparse-graph | 0.996667 | 0.999947 | 0.884 | 84.7 |
| caltech | small-candidate-pool | 0.998333 | 0.999986 | 0.925 | 62.7 |
| cub | small-candidate-pool | 1.000000 | 1.000000 | 1.010 | 46.5 |
| coco | small-candidate-pool | 0.991667 | 0.999915 | 0.748 | 59.2 |
| caltech | paper-default | 0.998333 | 0.999986 | 1.128 | 98.0 |
| cub | paper-default | 1.000000 | 1.000000 | 1.229 | 78.2 |
| coco | paper-default | 0.996667 | 0.999947 | 0.966 | 100.0 |
| caltech | wide-candidate-pool | 1.000000 | 1.000000 | 1.554 | 144.9 |
| cub | wide-candidate-pool | 1.000000 | 1.000000 | 1.611 | 124.5 |
| coco | wide-candidate-pool | 0.998333 | 0.999955 | 1.395 | 157.2 |
| caltech | dense-graph | 0.998333 | 0.999986 | 1.264 | 122.4 |
| cub | dense-graph | 1.000000 | 1.000000 | 1.343 | 97.1 |
| coco | dense-graph | 0.996667 | 0.999947 | 1.025 | 116.0 |

`wide-candidate-pool` 的平均 Recall@3 为 0.999444，平均延迟比 `paper-default` 高约 37%。
`small-candidate-pool` 的平均延迟最低，但 COCO Recall@3 降至 0.991667。综合三套数据，
当前代码继续采用 `paper-default`。

## 配置定义

| 配置 | embeddingK | semanticIntraK | bridge keys | bridge/key | pool multiplier | top keys |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse-graph | 8 | 12 | 2 | 1 | 3 | 5 |
| small-candidate-pool | 12 | 20 | 2 | 1 | 2 | 3 |
| paper-default | 12 | 20 | 2 | 1 | 3 | 5 |
| wide-candidate-pool | 12 | 20 | 2 | 1 | 5 | 8 |
| dense-graph | 16 | 24 | 4 | 2 | 3 | 5 |

运行时间：2026-07-23 UTC。基础实现对应 commit
`44e14ef10a9380de9011d5b46b9bedfad9bb093f`；扫描程序为
`tools/run_accuracy_parameter_scan.py`，输入文件哈希见 `manifests/release-artifacts.json`。
