# 仓颉工程说明

`cj_core/` 是本仓库唯一的仓颉编译工程，包名为 `semanticvg_cj`。

## 代码结构

```text
cj_core/
├── cjpm.toml
└── src/
    ├── main.cj
    ├── storage/
    │   ├── vectormap.cj
    │   ├── vectorgroup.cj
    │   ├── hdmg.cj
    │   ├── clustering.cj
    │   ├── mixed_scoring.cj
    │   └── utils.cj
    ├── bench/
    │   ├── paper_protocol.cj
    │   ├── evaluations.cj
    │   ├── external_backends.cj
    │   ├── runner.cj
    │   └── types.cj
    └── examples/
        ├── core_regression.cj
        └── minimal_vectormap.cj
```

## 构建与测试

```powershell
cd cj_core
cjpm build
cjpm test --no-color
```

`storage_test.cj` 使用仓颉标准 `unittest`，覆盖 mixed score、向量数学、`VectorGroup` CRUD 与
输入校验、索引生命周期、HDMG fallback、稳定排序和非法配置。

核心集成回归：

```powershell
"2" | cjpm run
```

## 非交互入口

程序启动后的主要命令：

```text
bench <smoke|partial|full> <1..6|t|v|a>
parity <1..6> <maxQueries> <beta>
dbbench <mock|milvus|qdrant|chroma> <1..6> <smoke|partial|full>
paper <cangjie_input.txt> <maxQueries> <beta|all>
maintenance <cangjie_input.txt> <count> [summary.log]
```

正式图像实验通常不直接手输这些命令，而是使用根目录 `tools/run_image_full_suite.ps1`。

## 实现边界

- 当前存储与 HDMG 核心使用 `Float64`；
- HDMG 是实际图索引；
- rep/single 当前仍是 exact-scan snapshot，不能称为真实 ANN；
- 数据插入、更新或删除后，旧索引会失效，需重建后再作为当前索引使用。
