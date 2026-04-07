"""
State Aggregator
----------------

将 SystemOutput 转换为更易于上层模型消费的表征，包括：
1. 下采样后的化学场栅格
2. 快速计算的状态向量（用于测试/控制策略）
3. 激素向量（HormoneLayer）：将生态统计量映射为5维生理信号

后续可以在此模块扩展更复杂的图结构或流形表示。
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from engine.core import Ecology2DConfig, SystemOutput


@dataclass
class AggregatedState:
    grid: np.ndarray
    stats: Dict
    emergence: Dict
    vector: np.ndarray
    multi_scale_grids: Dict[int, np.ndarray]
    history_vectors: np.ndarray


class SimpleStateAggregator:
    def __init__(
        self,
        config: Ecology2DConfig,
        grid_size: int = 64,
        grid_sizes: Optional[List[int]] = None,
        history_length: int = 32,
    ):
        self.config = config
        self.grid_sizes = grid_sizes or self._default_grid_sizes(grid_size)
        self.history = deque(maxlen=max(1, history_length))

    def aggregate(self, output: SystemOutput) -> AggregatedState:
        grids = {
            size: self._downsample_field(output.visualization, size)
            for size in self.grid_sizes
        }
        primary_grid = grids[self.grid_sizes[0]]
        vector = self._to_vector(output, primary_grid)
        self.history.append(vector.copy())
        history_array = (
            np.stack(tuple(self.history), axis=0)
            if self.history
            else np.zeros((0, vector.shape[0]), dtype=np.float32)
        )
        return AggregatedState(
            grid=primary_grid,
            stats=output.stats,
            emergence=output.emergence,
            vector=vector,
            multi_scale_grids=grids,
            history_vectors=history_array,
        )

    def _default_grid_sizes(self, base_size: int) -> List[int]:
        sizes = [max(4, int(base_size))]
        if base_size >= 16:
            sizes.append(base_size // 2)
        if base_size >= 32:
            sizes.append(base_size // 4)
        # 去重且按大→小排列，保证首个为主栅格
        unique = []
        for size in sorted(set(sizes), reverse=True):
            if size not in unique:
                unique.append(size)
        return unique or [max(4, base_size)]

    def _downsample_field(self, visualization: Dict, target_size: int) -> np.ndarray:
        field = visualization.get('chemical_field', {}).get('atp')
        if field is None:
            return np.zeros((target_size, target_size), dtype=np.float32)

        arr = np.array(field, dtype=np.float32)
        if arr.ndim != 2:
            return np.zeros((target_size, target_size), dtype=np.float32)

        sx = max(1, arr.shape[0] // target_size)
        sy = max(1, arr.shape[1] // target_size)
        down = arr[::sx, ::sy]

        target = np.zeros((target_size, target_size), dtype=np.float32)
        target[:down.shape[0], :down.shape[1]] = down[: target_size, : target_size]
        return target

    def _to_vector(self, output: SystemOutput, grid: np.ndarray) -> np.ndarray:
        stats = output.stats
        emergence = output.emergence

        alive_ratio = (
            stats['alive_particles'] / max(1, self.config.n_particles)
        )
        avg_energy = (
            stats['total_energy'] / max(1, stats['alive_particles'])
            if stats['alive_particles'] > 0 else 0.0
        )
        genome_stats = stats.get('genome_lengths', {})
        avg_genome = genome_stats.get('avg', 0.0)
        max_genome = genome_stats.get('max', 0.0)
        min_genome = genome_stats.get('min', 0.0)
        avg_atp = float(grid.mean()) if grid.size else 0.0

        vector = np.array([
            alive_ratio,
            avg_energy / 5.0,
            emergent := emergence.get('genetic_diversity', 0.0),
            emergence.get('energy_variance', 0.0),
            emergence.get('spatial_variance', 0.0),
            avg_genome / 50.0,
            max_genome / 60.0,
            min_genome / 40.0,
            emergence.get('emergence_score', 0.0),
            avg_atp,
        ], dtype=np.float32)
        return vector


# ---------------------------------------------------------------------------
# 激素层（下丘脑模块）
# ---------------------------------------------------------------------------

@dataclass
class HormoneVector:
    """
    5维激素向量——生态系统的生存状态信号。

    每个维度 [0, 1]，0 = 完全静默，1 = 最大激活。
    可直接作为 LLM system-prompt 注入或 steering vector 偏移量使用。
    """
    cortisol: float        # 皮质醇：压力/危机
    dopamine: float        # 多巴胺：驱动/奖励
    norepinephrine: float  # 去甲肾上腺素：警觉/竞争
    serotonin: float       # 血清素：稳态/安全
    oxytocin: float        # 催产素：聚集/合作

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.cortisol, self.dopamine, self.norepinephrine,
             self.serotonin, self.oxytocin],
            dtype=np.float32,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            'cortisol':        self.cortisol,
            'dopamine':        self.dopamine,
            'norepinephrine':  self.norepinephrine,
            'serotonin':       self.serotonin,
            'oxytocin':        self.oxytocin,
        }

    def __repr__(self) -> str:
        d = self.to_dict()
        parts = [f"{k[:3].upper()}={v:.3f}" for k, v in d.items()]
        return f"HormoneVector({', '.join(parts)})"


class HormoneLayer:
    """
    下丘脑模块：将生态引擎统计量映射为激素向量。

    调用流程：
        aggregated_state = aggregator.aggregate(system_output)
        hormone_vec      = hormone_layer.compute(aggregated_state)

    不依赖、不修改任何现有接口，是纯粹的 AggregatedState → HormoneVector 映射。

    生物类比：
        生态引擎原始信号 → 下丘脑整合 → 激素向量 → （未来）LLM 调制
    """

    # history_vectors 中各列的含义（与 _to_vector 保持一致）
    _COL_ALIVE_RATIO = 0

    def __init__(self, config: Ecology2DConfig, smoothing: float = 0.3):
        """
        Args:
            config:    生态系统配置，用于归一化粒子数。
            smoothing: EMA 平滑系数 (0, 1)。越大跟踪越灵敏，越小越平稳。
        """
        self.config = config
        self.smoothing = smoothing

        # 累计事件上一步的值（用于计算增量速率）
        self._prev_replication: int = 0
        self._prev_death: int = 0

        # EMA 平滑后的激素值，初始化为生理静息态
        # [cortisol, dopamine, norepinephrine, serotonin, oxytocin]
        self._ema = np.array([0.1, 0.3, 0.1, 0.5, 0.4], dtype=np.float32)

        self._step: int = 0

    def compute(self, state: AggregatedState) -> HormoneVector:
        """
        输入当前聚合状态，输出平滑后的激素向量。
        每次调用后自动推进内部状态（事件计数、EMA）。
        """
        stats    = state.stats
        emergence = state.emergence
        history  = state.history_vectors  # shape (T, 10)

        n_alive      = max(1, stats.get('alive_particles', 1))
        alive_ratio  = n_alive / max(1, self.config.n_particles)
        avg_energy   = stats.get('total_energy', 0.0) / n_alive

        # 事件增量（累计值 → 本步新增）
        cur_rep   = stats.get('replication_events', 0)
        cur_death = stats.get('death_events', 0)
        rep_delta   = max(0, cur_rep   - self._prev_replication)
        death_delta = max(0, cur_death - self._prev_death)
        self._prev_replication = cur_rep
        self._prev_death       = cur_death

        # 存活率近期趋势（正=增长，负=崩溃）
        alive_trend = self._alive_trend(history)

        # ── 皮质醇（压力/危机） ──────────────────────────────
        inhibitor = emergence.get('inhibitor_pressure', 0.0)
        cortisol  = (
            (1.0 - alive_ratio)              * 0.35 +
            min(death_delta / 50.0, 1.0)     * 0.35 +
            min(inhibitor * 2.0,    1.0)     * 0.20 +
            min(max(-alive_trend, 0) * 10.0, 1.0) * 0.10
        )

        # ── 多巴胺（驱动/奖励） ──────────────────────────────
        emergence_score = emergence.get('emergence_score', 0.0)
        dopamine = (
            min(rep_delta   / 30.0, 1.0) * 0.40 +
            min(avg_energy  / 5.0,  1.0) * 0.30 +
            emergence_score              * 0.30
        )

        # ── 去甲肾上腺素（警觉/竞争） ────────────────────────
        energy_var     = emergence.get('energy_variance', 0.0)
        norepinephrine = (
            min(energy_var  / 10.0, 1.0) * 0.50 +
            min(death_delta / 30.0, 1.0) * 0.30 +
            (1.0 - min(alive_ratio * 2, 1.0)) * 0.20
        )

        # ── 血清素（稳态/安全） ──────────────────────────────
        alive_stability = 1.0 - min(abs(alive_trend) * 20.0, 1.0)
        serotonin = (
            alive_stability              * 0.40 +
            emergence_score              * 0.30 +
            min(alive_ratio * 1.2, 1.0) * 0.30
        )

        # ── 催产素（聚集/合作） ──────────────────────────────
        spatial_var      = emergence.get('spatial_variance', 300.0)
        genetic_div      = emergence.get('genetic_diversity', 1.0)
        spatial_cohesion = 1.0 - min(spatial_var / 300.0, 1.0)
        # 适中遗传多样性（非克隆但有亲缘）催产素最高，用倒抛物线建模
        kinship  = max(0.0, 1.0 - abs(genetic_div - 0.4) / 0.6)
        oxytocin = spatial_cohesion * 0.60 + kinship * 0.40

        # 裁剪到 [0, 1]
        raw = np.clip(
            [cortisol, dopamine, norepinephrine, serotonin, oxytocin],
            0.0, 1.0,
        ).astype(np.float32)

        # EMA 平滑
        self._ema = self.smoothing * raw + (1.0 - self.smoothing) * self._ema
        self._step += 1

        return HormoneVector(
            cortisol=float(self._ema[0]),
            dopamine=float(self._ema[1]),
            norepinephrine=float(self._ema[2]),
            serotonin=float(self._ema[3]),
            oxytocin=float(self._ema[4]),
        )

    def _alive_trend(self, history: np.ndarray) -> float:
        """
        用 history_vectors[:, 0]（alive_ratio列）计算近期斜率。
        正值 = 种群增长，负值 = 种群下降。
        """
        if history.shape[0] < 4:
            return 0.0
        recent = history[-min(8, history.shape[0]):, self._COL_ALIVE_RATIO]
        half   = max(1, len(recent) // 2)
        return float(recent[half:].mean() - recent[:half].mean())

    def reset(self) -> None:
        """重置内部状态，新实验开始时调用。"""
        self._prev_replication = 0
        self._prev_death       = 0
        self._ema = np.array([0.1, 0.3, 0.1, 0.5, 0.4], dtype=np.float32)
        self._step = 0
