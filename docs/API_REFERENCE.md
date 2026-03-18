# ExternalInput / SystemOutput API

文档状态：`ACTIVE`

本文件记录现阶段引擎的统一输入输出格式，供控制器、测试或外部系统调用。

---

## 1. ExternalInput

所有输入事件都以 `ExternalInput` 结构表示：

```json
{
  "input_type": "chemical_pulse",
  "params": { ... },
  "metadata": { ... }   # 可选
}
```

### 支持的 `input_type`

| 类型 | 说明 | 典型参数 |
|------|------|----------|
| `chemical_pulse` | 在指定区域注入化学物质/能量 | `x`, `y`, `radius`, `intensity`, `chemical_index` |
| `gradient_field` | 重新构建化学梯度 | `axis` (`x`/`y`), `start_value`, `end_value`, `chemical_index` |
| `catastrophe` | 触发灾难事件（内部映射到 `trigger_disturbance_event`） | `event_type` (`energy_fluctuation`/`mass_extinction`/`mutation_burst`) |
| `parameter_adjust` | 动态修改配置（受 allowlist 与范围限制） | 仅允许运行时安全字段，如 `mutation_rate`, `solar_energy_rate` |

示例：

```python
from engine import core

config = core.Ecology2DConfig()
system = core.Ecology2DSystem(config)

inp = core.ExternalInput(
    input_type='chemical_pulse',
    params={'x': 100, 'y': 120, 'radius': 12, 'intensity': 1.5}
)
system.apply_external_input(inp)
```

---

## 2. SystemOutput

引擎通过 `get_system_output()` 返回结构化状态：

```json
{
  "time_step": 1234,
  "timestamp": 1730000000.123,
  "visualization": { ... },  # 包含粒子/化学场
  "stats": { ... },
  "emergence": { ... },
  "recent_inputs": [ ... ],
  "metadata": { ... }  # 可选
}
```

### 关键字段

- `visualization.particles`：粒子坐标、能量、世代、基因等数组。  
- `visualization.chemical_field.atp`：ATP 浓度矩阵。  
- `stats`：存活数量、总能量、事件计数等。  
- `emergence`：涌现得分、多样性、空间方差、最大世代等。  
- `recent_inputs`：最近应用的 ExternalInput 记录（默认保留 5 条）。

外部控制器可以按如下方式消费状态：

```python
output = system.get_system_output()
score = output.emergence['emergence_score']
alive = output.stats['alive_particles']

if score < 0.4:
    # 做一些调节
```

---

## 3. WebSocket 消息

`engine/server.py` 将 `SystemOutput` 转为 JSON 推送给前端；同时支持以下动作：

```json
{ "action": "pause" | "resume" | "reset" | "export" }
{ "action": "input", "input_type": "...", "payload": {...} }
{ "action": "snapshot" }
```

返回消息：

- **实时广播**：直接发送 `SystemOutput` JSON。  
- **输入确认**：`{"type": "input_ack", "input": {...}, "time_step": ...}`  
- **快照**：`{"type": "snapshot", "data": {SystemOutput}}`  
- **错误**：`{"type": "error", "message": "..."}`  

---

## 4. 最佳实践

1. 所有外部脚本务必通过 `ExternalInput`/`SystemOutput` 交互，避免直接修改 `particles`。  
2. 使用 `metadata` 标记输入来源，方便追踪实验。  
3. 控制器可根据 `recent_inputs` 防止在短时间内重复同一事件。  
4. 当需要批量输入时，建议通过 WebSocket/脚本对接，而不是直接调用内部函数。

未来版本会在此基础上扩展更多输入类型和输出指标（例如多化学层、任务得分、记忆向量等），保持兼容性。
