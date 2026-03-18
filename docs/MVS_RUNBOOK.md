# DAERWEN3.5 最小可行系统（MVS）运行手册

文档状态：`ACTIVE`

本手册记录在当前环境下复现 2D 生态引擎闭环实验的步骤，确保任何人都能快速启动系统并观察涌现指标。

---

## 1. 环境准备

1. 确保使用 Python 3.12（仓库已自带 `.venv`，也可新建）：
   ```powershell
   cd F:\avalanche-持续学习\daerwen3.5
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. 安装依赖：
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install numpy websockets orjson
   ```
   > `numpy`：核心数值计算；`websockets`：实时服务器；`orjson`：高性能序列化。若需要更多依赖，同样按需安装。

3. 若离线环境，准备 wheel 放入 `wheels\` 后离线安装：
   ```powershell
   python -m pip install --no-index --find-links .\wheels numpy websockets orjson
   ```

---

## 2. 手动驱动闭环（ManualDriver）

用于验证 `ExternalInput/SystemOutput` 闭环：

```powershell
@'
from controllers.manual_driver import ManualDriver
from engine import core
cfg = core.Ecology2DConfig(world_size=32, n_particles=200)
driver = ManualDriver(cfg, step_interval=0.0)
driver.run(total_steps=500)
'@ | python -
```

运行期望日志：
- `⚠️ GPU不可用，使用CPU模式 (NumPy)`：自动回退 CPU。  
- `[Step ...] Emergence=... Diversity=...`：每 50 步输出一次涌现、基因多样性、存活数、总能量。  
- 当多样性 < `target_diversity` 或能量 < `target_energy` 时，脚本通过 `chemical_pulse` / `parameter_adjust` 注入能量。

此步骤验证：引擎步进→状态聚合→反馈输入的闭环链路可用。

---

## 3. 启动实时服务器 + Web UI

```powershell
python scripts\start_engine.py
```

流程说明：
1. 服务器监听 `ws://localhost:8765`。  
2. 打开 `engine/web.html`（本地文件）即可连入实时可视化面板。  
3. 使用外部输入面板触发 `chemical_pulse` / `catastrophe` / `parameter_adjust`。  
4. “最近输入”“涌现指标”区域会实时反馈；若长时间僵化，服务器会自动触发扰动。  
5. `Ctrl+C` 退出；若需要批量导出状态，可在 Web UI 里执行 `snapshot/export`。

常见依赖：`websockets`、`orjson`。若未安装会在启动时提示，可按第 1 节命令补齐。

---

## 4. 测试套件（可选）

`tests/comprehensive_suite.py` 利用 `SimpleStateAggregator` 评估一致性/区分度：
```powershell
python tests\comprehensive_suite.py
```
运行时间较长（需多轮 step），建议在确认闭环稳定后再执行。测试脚本会自动禁用 GPU (`CUDA_VISIBLE_DEVICES=-1`)，并在内部生成输入扰动、应用选择压力。

### 4.1 控制器评测（推荐）

`scripts/evaluate_controllers.py` 已采用“warmup 与 measured 分离”口径：

- `warmup` 阶段：仅世界自由演化（只执行 `system.step()`），控制器不注入动作。
- `measured` 阶段：控制器正式介入，评测指标以 measured 窗口为主。
- 输出同时包含 prewarm/measured/gain，避免把预热期门控指标误当有效策略反馈。

示例：

```powershell
python scripts\evaluate_controllers.py --world-size 80 --particles 1200 --prewarm 1000 --run 500 --controllers manual predictive --step-interval 0.0
```

建议：

- 所有对比实验保持相同 `prewarm/run` 配置。
- 记录 `mode`、`seed`、`prewarm`、`run` 与事件日志（若开启 `--log-events`）。

---

## 5. 下一步工作对齐

- **记录实验轨迹**：将 `ManualDriver` / `start_engine` 的关键输出（配置、输入事件、涌现指标）保存到 `data_archive/`，为持续学习控制器提供训练数据。  
- **生态优化优先项**（参照 `docs/AGI_VISION.md` §6）：  
  1. 扩展化学反应与能量循环，确保 `SystemOutput.emergence` 反映空间熵、能量梯度等指标。  
  2. 让 DNA 表达路径决定分子/代谢策略，为“潜意识”记忆提供物理载体。  
  3. 在 `StateAggregator` 中追加多尺度网格与历史缓冲，确保控制器能读取生态慢变量。  
- **控制器迭代**：在 `controllers/` 新增主动推理/记忆模块，利用聚合后的状态 + 历史输入训练预测模型，再通过 `ExternalInput` 驱动生态演化。

该 runbook 将随环境或依赖更新同步维护，确保“最小可行路径”始终清晰。
