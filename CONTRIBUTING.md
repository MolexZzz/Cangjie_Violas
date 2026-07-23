# 贡献指南

仓库分为仓颉核心、实验工具、文档和汇总结果四部分。新增实验不应改变核心存储接口；
算法改动也不应直接写入数据准备脚本。

## 开发检查

```powershell
cd cj_core
cjpm build
cjpm test --no-color
"2" | cjpm run
cd ..
python -m compileall -q tools photo-data
python tools\verify_release_bundle.py
```

## 提交约定

- 算法改动需要对应的单元测试或固定输入回归测试；
- 性能优化需要记录数据集、参数、before/after 和准确率变化；
- 新实验结果应注明实验协议、查询范围、数据哈希和 Git commit；
- 不提交原始数据、模型、embedding、容器数据或大型日志；
- 文档只记录可以核验的事实，并区分正式结果与调试输出；
- 不把不同目标函数下的 Recall 或延迟直接作为算法排名。

## 实验结果

原始输出写入 `results/`，不纳入版本控制。需要提交的汇总结果放在 `results-summary/`，
并通过 `tools/verify_release_bundle.py` 检查。若实验输入发生变化，还需更新
`manifests/release-artifacts.json`。
