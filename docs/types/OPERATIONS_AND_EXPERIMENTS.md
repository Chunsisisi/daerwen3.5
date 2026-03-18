# 运行与实验

## 范围

本文件汇总运行流程、实验协议与环境约束。

来源文档：

- `docs/MVS_RUNBOOK.md`
- `docs/notes/session_summary_gpu.md`
- `scripts/` 下运行脚本

## 标准运行路径

- 手动闭环烟测路径（`ManualDriver`）。
- Server + Web UI 演示路径。
- 控制器评测路径（`scripts/evaluate_controllers.py`）。

## 可复现性要求

- 必须记录配置、seed 范围、运行模式与干预日志。
- 报告必须区分 warmup 与 measured 区间。
- 对比实验必须保持 benchmark 配置一致。

## GPU/性能说明

- GPU 运行不可用时默认 CPU 回退属于正常行为。
- WSL/CUDA 路径受环境影响，建议每次会话重验。
- 当前主瓶颈仍是 Python 层逐粒子循环。

## CONFLICT: 演示便利性 vs 实验纯净性

观点A：

- demo/server 流程希望保留自动扰动与交互便利特性。

观点B：

- 实验级可比性要求严格受控干预。

处理准则：

- 演示便利特性与 baseline benchmark 模式必须隔离。

## 运行安全基线

- 优先使用通过校验的 `ExternalInput` 路径。
- 运行脚本避免直接状态突变。
- 所有非 baseline 干预必须保留日志。
