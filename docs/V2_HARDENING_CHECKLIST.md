# DAERWEN3.5 v2 硬化清单（冻结版）

## 1. 目的

本文件用于在 v2 实施前冻结不可谈判的硬化要求。

目标：

- 保护第一性原理涌现基线。
- 消除隐藏干预路径与不安全运行时突变。
- 保证控制器动作可通过 `ExternalInput` 审计。

范围：

- `engine/core.py`
- `engine/server.py`
- `controllers/manual_driver.py`
- `controllers/predictive_controller.py`
- `scripts/evaluate_controllers.py`
- `tests/comprehensive_suite.py`

---

## 2. 运行模式（必须显式声明）

以下三种模式为强制项，必须写入实验元数据：

- `baseline`：世界演化中不允许服务层自主干预。
- `experiment`：允许控制器注入 `ExternalInput`。
- `demo`：允许交互式运行、可视化和显式人工输入。

未声明模式的实验报告视为无效。

---

## 3. 不可谈判项

### 条目1 - 模式隔离

要求：

- `baseline` 模式禁止服务层自主扰动逻辑。
- 任何自主干预必须受模式开关约束，并在输出元数据/日志中可见。

代码锚点：

- `engine/server.py:190` (`_handle_stagnation`)
- `engine/server.py:204` (`trigger_disturbance_event` call)

通过标准：

- baseline 运行中不存在服务触发扰动事件（除非显式开启）。

---

### 条目2 - ExternalInput 边界严格性

要求：

- 未知 `input_type` 必须拒绝并记录。
- 未知输入不得导致任何世界状态突变。

代码锚点：

- `engine/core.py:1156` (`apply_external_input`)
- `engine/core.py:1172` (current unknown-input fallback to random disturbance)

通过标准：

- 无效/未知输入走拒绝路径，且不触发任何扰动副作用。

---

### 条目3 - parameter_adjust 安全契约

要求：

- `parameter_adjust` 必须采用显式 allowlist + 按键类型/范围校验。
- 结构字段运行时不可变，更改必须通过 reset/reinit 流程。

结构字段（最小集合）：

- `world_size`
- `n_particles`
- `genome_length`
- `n_chemical_species`

代码锚点：

- `engine/core.py:1216` (`_apply_parameter_adjust_input`)
- `engine/core.py:1219` (current `hasattr` + `setattr` pattern)

通过标准：

- 超范围与非 allowlist 键全部拒绝。
- 结构字段在运行时被拒绝，并返回明确原因。

---

### 条目4 - Warmup 指标有效性

要求：

- step<1000 期间被门控的涌现指标不得按“完全有效信号”驱动控制策略。
- 评测报告必须拆分 warmup 与 measured 阶段。

代码锚点：

- `engine/core.py:1397` (pre-1000 emergence gating)
- `controllers/manual_driver.py:31` (feedback cadence every 50 steps)
- `controllers/predictive_controller.py:93` (`_score_vector` includes emergence signal)
- `scripts/evaluate_controllers.py:38` (`run_sequence`)
- `scripts/evaluate_controllers.py:49` (`prewarm`: only `system.step()`)
- `scripts/evaluate_controllers.py:58` (`measured`: controller `run(total_steps=run_steps)`)

通过标准：

- 控制器/评测流程显式处理 warmup 有效性，并单独报告 measured 窗口。
- `evaluate_controllers` 输出必须同时包含 prewarm/measured 指标与 gain，报告结论以 measured 窗口为主。

---

### 条目5 - 控制器角色边界

要求：

- 控制器只能通过 `ExternalInput` 影响世界。
- 控制器不得直接修改粒子内部、化学场内部，或直接调用内部扰动原语。

代码锚点：

- `controllers/manual_driver.py:74` (`ExternalInput('chemical_pulse', ...)`)
- `controllers/manual_driver.py:78` (`ExternalInput('parameter_adjust', ...)`)
- `controllers/predictive_controller.py:146` (`apply_external_input`)
- `tests/comprehensive_suite.py:178` (direct particle mutation path to be kept test-only and excluded from production control semantics)

通过标准：

- 生产控制器路径严格为 `SystemOutput -> policy -> ExternalInput`。

---

## 4. v2 硬化完成定义

以下条件必须全部满足：

1. 5/5 不可谈判项全部通过。
2. 每个运行产物都声明 baseline/experiment/demo 模式。
3. 输入审计链可以解释每一次状态变更干预。
4. baseline 模式不存在隐藏服务层突变。

---

## 5. 复核闸门

本清单已冻结，用于实施规划。

任何例外必须提供：

- 明确理由，
- 代码锚点，
- 对第一性完整性与设计师陷阱风险的影响评估。
