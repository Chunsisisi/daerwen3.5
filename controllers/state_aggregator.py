"""
State Aggregator
----------------

将 SystemOutput 转换为更易于上层模型消费的表征，包括：
1. 下采样后的化学场栅格
2. 快速计算的状态向量（用于测试/控制策略）

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
