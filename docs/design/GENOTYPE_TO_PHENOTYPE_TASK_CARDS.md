# 基因到显性改造任务卡（代码落地版）

## 目标

将当前 `genome -> behavior parameters` 主链，逐步替换为：

`genome -> regulatory_state -> reaction_coefficients -> growth/structure -> behavior`

约束：

- 不新增上层目标塑形。
- 不新增人工行为闸门（年龄门槛、复制比例上限等）。
- 每次只改一个底层环节，改后必须跑固定回归协议。

---

## 回归协议（每张卡都必须执行）

- 固定实验配置与 seed 区间。
- 最低烟测：20-seed。
- 关键里程碑：50-seed。
- 核心指标：
  - `extinction_rate`
  - `genetic_diversity`（`div_mean`）
  - `max_generation`（`max_gen_mean`）

---

## 卡1：复制通道去直连（第一刀）

### 目的

把复制判断从“直接读 `replication_threshold`”改成“读调控态 + 局部反应可达性”。

### 现锚点

- `_express_genes` 直接生成 `replication_threshold`：`engine/core.py:190`
- 复制直接读取表型阈值：`engine/core.py:1020`

### 改造动作（最小）

1. 在 `Particle2D` 增加 `regulatory_state`（连续变量字典）。
2. 在每步代谢后更新 `regulatory_state`（输入仅来自局部化学场+能量）。
3. `_particle_replication` 改为读取 `regulatory_state['replication_drive']`，不直接读 `phenotype['replication_threshold']`。

### 验收

- 代码可编译、可运行。
- 20-seed 指标不劣化到不可用（至少不过度崩溃）。
- 输入边界与模式隔离不回退。

---

## 卡2：代谢通道去直连

### 目的

将 `conversion_threshold` 从行为阈值改成“反应系数影响代谢方程”。

### 现锚点

- `conversion_threshold` 直接来自 `_express_genes`：`engine/core.py:214`
- 代谢直接按阈值分支：`engine/core.py:762`

### 改造动作（最小）

1. 以 `regulatory_state` 生成 `reaction_coefficients`（如 waste->ATP 转化速率）。
2. `_particle_metabolism` 用速率计算替代阈值 if-branch。
3. 保留原参数作为回退开关，便于 A/B。

### 验收

- 20-seed 与回退版本可对比。
- 记录 `reaction_coefficients` 统计，确保可审计。

---

## 卡3：交互通道去直连

### 目的

将 `interaction_mode` 从直接行为基线，进一步下沉为“调控态驱动的反应倾向”。

### 现锚点

- 交互模式基线读取：`engine/core.py:1001`
- 交互规则核心：`engine/core.py:863`

### 改造动作（最小）

1. 新增 `regulatory_state['interaction_drive']` 更新。
2. `_effective_interaction_mode` 改为优先读调控态，不直接用固定基因基线。
3. 交互强度仅由局部场、能量、调控态共同决定。

### 验收

- 20-seed 不出现系统性立即灭绝。
- 交互事件统计稳定（`interaction_events` 不异常爆炸）。

---

## 卡4：生长结构显式化（从“表达”迈向“生长”）

### 目的

引入最小“生长态”变量，让结构变化可积累，而不是每步瞬时决策。

### 推荐实现

- 新增 `morph_state`（如体积/表面积代理、局部黏附潜势）。
- 每步由反应系数更新 `morph_state`。
- 行为函数（移动/复制/交互）读取 `morph_state`，不直接读取基因段。

### 验收

- 能观察到跨时间累积效应（非瞬时噪声）。
- 指标不因引入状态变量而完全失真。

---

## 卡5：最小环境共进化接口（不是先做）

### 前置条件

- 卡1-4 至少完成两张且通过 50-seed 回归。

### 目的

将“扰动脚本”升级为“环境参数族群”，避免把扰动当主复杂性引擎。

### 最小动作

- 把扰动参数封装为 `EnvironmentGenome`。
- 记录环境参数与种群结果的配对轨迹。
- 先做离线筛选，不直接在线闭环。

---

## 禁止项（红线）

- 禁止通过人工行为闸门直接“修稳定”。
- 禁止把高层目标分数写进底层更新规则。
- 禁止绕开 `ExternalInput` / `SystemOutput` 边界做控制层直改。

---

## 执行顺序建议

1. 卡1（复制去直连）
2. 卡2（代谢去直连）
3. 卡3（交互去直连）
4. 卡4（显式生长态）
5. 卡5（环境共进化接口）

说明：当前最关键是“先把表达链路拉长”，不是先做世界共进化。
