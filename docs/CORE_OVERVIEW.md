DAERWEN3.5 核心概览（敏感细节已适度隐藏）
========================================

文档状态：`REFERENCE`

> 本文档用于快速了解 DAERWEN3.5 的核心组成、关键流程与测试框架。为避免触发设计师陷阱，部分实现细节与参数以 `[已隐藏]` 标记处理，仅保留必要的结构信息。

1. 引擎层（`engine/`)
---------------------

- **核心系统**：`engine/core.py` 定义 `Ecology2DSystem` 与 `Ecology2DConfig`。系统将粒子（基因、能量、位置）与化学场整合，主循环 `step()` 顺序执行扩散、太阳能输入、新陈代谢、交互、移动、复制与死亡等过程。  
  - 化学物质：ATP、废物等由 `ChemicalField2D` 管理，扩散方程采用 5 点差分模板。  
  - 粒子行为：`Particle2D` 将基因八段表达为场交互、运动响应、复制阈值等特征，复制阶段包含点突变 + 低概率结构变异。  
  - 涌现指标：`get_emergence_metrics()` 在 1000 步后计算世代多样性、基因多样性、能量方差等，合成 `emergence_score`。部分具体阈值已用 `[已隐藏]` 代替。

- **实时服务器**：`engine/server.py` 提供 `Ecology2DServer`，通过 WebSocket 向 `engine/web.html` 推送 `SystemOutput`。  
  - 支持外部指令：暂停、恢复、重置、输入事件、数据导出。  
  - 动态扰动：检测种群僵化后在冷却期外触发 `energy_fluctuation / mass_extinction / mutation_burst`。  
  - JSON 序列化默认使用 `orjson`；广播逻辑由 `websockets` 完成。

2. 控制与聚合层（`controllers/`)
-------------------------------

- **状态聚合器**：`controllers/state_aggregator.py` 的 `SimpleStateAggregator` 将 `SystemOutput` 下采样为多尺度栅格，构建 10 维状态向量（包含存活比例、平均能量、基因长度统计等）。  
  - 历史缓存：`history_length` 默认 32，用于上层控制器进行时间建模。  
  - 由于一致性测试敏感，建议在外层对 `vector` 做规范化或裁剪（当前实现未处理）。

3. 测试体系（`tests/comprehensive_suite.py`)
-----------------------------------------

- **总览**：`ComprehensiveTestSuite` 集成 5 类标准评估：基准训练、泛化、鲁棒性、重复性、消融。  
  1. *基准训练*：输入序列 A/B/C，每次注入扰动（能量/灭绝/突变）后采集状态，按一致性/区分度施加选择压力。  
  2. *泛化*：对未见输入 D/E 计算响应稳定性，并与训练输入最近 5 次平均向量求相似度。  
  3. *鲁棒性*：向输入 A 添加不同噪声或设置极端事件，衡量与原响应差异及恢复率。  
  4. *重复性*：重复 10 次同一输入，计算均值、标准差与 CV。  
  5. *消融*：在记录基线后移除选择压力，观察一致性与区分度的变化。  
- **一致性问题提示**：由于每次输入前都会触发强扰动并仅演化 15 步，状态方差剧烈，导致 `consistency ≈ 0`。若需要更稳定的数据，请引入更长的冷却期 `[已隐藏]` 或限制 `mass_extinction` 频率。

4. 脚本与文档
------------

- `scripts/start_engine.py` / `start_engine_cpu.py`：一键启动 WebSocket 服务器；浏览器打开 `engine/web.html` 可实时观察。  
- `docs/API_REFERENCE.md`：列出工程 API 的调用约定；由于保密策略，关键端点参数与默认值在文档中以 `[已隐藏]` 处理。  
- `requirements_performance.txt`：性能依赖（如 `orjson`, `numba`）按需安装；GPU 路径可选 CuPy，默认读 `CUDA_VISIBLE_DEVICES`（测试中常设为 `-1` 强制 CPU）。

5. 建议的安全操作
---------------

- 运行综合测试前，务必执行 `_reset_system()`，当前实现默认预热 200 步；如需更长预热，可在 `ComprehensiveTestSuite` 初始化后额外调用。  
- 若希望在 Web 端观察测试过程，可将 `ComprehensiveTestSuite` 替换为对 `Ecology2DServer` 的控制脚本，通过 WebSocket 注入输入并订阅状态。  
- 任何修改需保持“无预设行为”原则：避免硬编码目标策略，仅通过扰动与选择引导演化；涉及具体阈值的部分已在文档中以 `[已隐藏]` 占位，实际数值请留在代码中控制。

附录
----

- **日志**：`logs/` 中存储测试与手动注入事件（如 `test_manual.jsonl`）。  
- **数据归档**：`data_archive/` 保存历史实验数据；敏感部分未在此文档中展开。  
- **哲学讨论**：`哲学讨论记录-从设计师陷阱到宇宙本质.md` 收录设计反思，与技术实现分离。

> 如需更细粒度的实现细节，请直接查阅对应源码文件，并确保遵循“隐蔽关键部分”的安全约束。
