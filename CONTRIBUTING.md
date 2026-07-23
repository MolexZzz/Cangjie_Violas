# 参与贡献

感谢对 Cangjie Violas 的关注。提交修改前，请先确认改动属于仓颉核心、实验工具、文档或冻结结果
中的哪一层，避免将实验逻辑混入存储核心。

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

## 提交要求

- 算法改动需要对应的单元测试或固定输入回归测试；
- 性能优化需要记录数据集、参数、before/after 和准确率变化；
- 新实验结果必须包含协议标识、查询范围、数据哈希和 Git commit；
- 不提交原始数据、模型、embedding、容器数据或大型日志；
- 文档使用可验证的陈述，区分正式结果、调试结果和未来工作；
- 不把不同目标函数下的 Recall 或延迟直接作为算法排名。

## 结果更新

原始输出写入 `results/`。只有经过 `tools/verify_release_bundle.py` 校验且适合公开审阅的小型汇总
才能进入 `results-summary/`。更新冻结结果时，应同时更新 `manifests/release-artifacts.json`。
