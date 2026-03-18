"""
Manual Driver - 示例控制器
---------------------------------

使用统一的 ExternalInput/SystemOutput 接口，构建一个最小闭环：
    1. 连接引擎（通过直接导入 engine.core）
    2. 周期性读取 SystemOutput
    3. 根据涌现得分/能量等指标生成 ExternalInput

该示例用于演示如何快速编写新的控制脚本，
后续可以替换为更复杂的主动推理或世界模型。
"""
import time
import numpy as np
from engine import core
from .state_aggregator import SimpleStateAggregator, AggregatedState


class ManualDriver:
    def __init__(self, config: core.Ecology2DConfig, step_interval: float = 0.0):
        self.system = core.Ecology2DSystem(config)
        self.step_interval = step_interval
        self.aggregator = SimpleStateAggregator(config, grid_size=32)
        self.target_diversity = 0.3
        self.target_energy = 1500

    def run(self, total_steps: int = 2000):
        for step in range(total_steps):
            self.system.step()

            if step % 50 == 0:
                output = self.system.get_system_output(metadata={'source': 'manual_driver'})
                agg_state = self.aggregator.aggregate(output)
                self._log_status(output, agg_state)
                self._apply_feedback(output, agg_state)

            if self.step_interval:
                time.sleep(self.step_interval)

    def _log_status(self, output: core.SystemOutput, agg_state: AggregatedState):
        emergence = output.emergence or {}
        stats = output.stats or {}
        emergence_score = float(emergence.get('emergence_score', 0.0))
        diversity = float(emergence.get('genetic_diversity', 0.0))
        alive = int(stats.get('alive_particles', 0))
        energy = float(stats.get('total_energy', 0.0))
        print(
            f"[Step {output.time_step:5d}] "
            f"Emergence={emergence_score:.3f} "
            f"Diversity={diversity:.3f} "
            f"Alive={alive} "
            f"Energy={energy:.1f}"
        )

    def _apply_feedback(self, output: core.SystemOutput, agg_state: AggregatedState):
        emergence = output.emergence or {}
        stats = output.stats or {}
        diversity = float(emergence.get('genetic_diversity', 0.0))
        alive = int(stats.get('alive_particles', 0))
        total_energy = float(stats.get('total_energy', 0.0))

        if diversity < self.target_diversity:
            hotspot = self._select_injection_point(agg_state)
            hx, hy = hotspot if hotspot else (
                int(alive % self.system.config.world_size),
                int((alive // 2) % self.system.config.world_size),
            )
            params = {
                'x': hx,
                'y': hy,
                'radius': 10,
                'intensity': 1.2,
            }
            self.system.apply_external_input(core.ExternalInput('chemical_pulse', params))

        if total_energy < self.target_energy:
            adjust = {'solar_energy_rate': self.system.config.solar_energy_rate * 1.05}
            self.system.apply_external_input(core.ExternalInput('parameter_adjust', adjust))

    def _select_injection_point(self, agg_state: AggregatedState):
        grids = agg_state.multi_scale_grids
        if not grids:
            return None
        smallest = min(grids.keys())
        grid = grids[smallest]
        if grid.size == 0:
            return None
        flat_idx = int(np.argmax(grid))
        gx, gy = divmod(flat_idx, grid.shape[1])
        scale = self.system.config.world_size / max(1, grid.shape[0])
        x = int(gx * scale)
        y = int(gy * scale)
        return x % self.system.config.world_size, y % self.system.config.world_size


if __name__ == "__main__":
    cfg = core.Ecology2DConfig(world_size=150, n_particles=1200)
    driver = ManualDriver(cfg, step_interval=0.0)
    driver.run(total_steps=1000)
