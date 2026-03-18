# 工程规范与护栏

## 范围

本文件汇总运行时行为与接口边界的工程契约。

来源文档：

- `docs/CORE_OVERVIEW.md`
- `docs/API_REFERENCE.md`
- `docs/V2_HARDENING_CHECKLIST.md`

## 运行时契约

- 统一接口：`apply_external_input()` 与 `get_system_output()`。
- 控制器与脚本应只通过该边界交互。
- 生产路径中的状态变更必须可审计。

## 模式契约

- `baseline`：服务层不允许隐藏式自主干预。
- `experiment`：允许控制器干预。
- `demo`：允许人工交互与可视化便利特性。

## 输入安全契约

- 未知输入类型必须拒绝并记录。
- `parameter_adjust` 必须 allowlist + range-check。
- 结构性字段禁止运行时热修改。

## 指标有效性契约

- warmup 阶段前的涌现指标视为低有效性。
- 报告必须明确区分 warmup 与 measured 区间。

## CONFLICT: API文档与硬化清单

观点A：

- `docs/API_REFERENCE.md` 现描述 `parameter_adjust` 可修改任意 `Ecology2DConfig` 字段。

观点B：

- `docs/V2_HARDENING_CHECKLIST.md` 要求严格 allowlist + 结构字段锁定。

处理准则：

- 运行时以硬化清单为准；API 文档需同步更新。

## CONFLICT: 概览文档与基线纯净性

观点A：

- `docs/CORE_OVERVIEW.md` 记录了 server 僵化触发扰动能力。

观点B：

- 硬化契约要求 baseline 模式不能有隐藏式自主干预。

处理准则：

- 保留功能，但仅在 experiment/demo 显式开启；baseline 禁用。

## 必守边界规则

生产控制器不得直接修改粒子内部状态或化学场内部状态。
