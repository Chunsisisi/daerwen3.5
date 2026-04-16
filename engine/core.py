"""
DAERWEN3 - 2D自持生态网络核心系统
基于物理-化学-基因三层整合的涌现AGI平台

设计哲学：
1. 使用2D空间（计算效率提升10倍+）
2. 只定义最小物理规则，让生态网络自发涌现
3. 物理层-化学层-基因层完全整合
4. 无预设生态关系（捕食、共生等应涌现，非设计）
"""
import os
import builtins
import sys
import numpy as np


def _safe_print(*args, **kwargs):
    """Best-effort console print that won't crash on narrow encodings (e.g. gbk)."""
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        fallback_args = []
        for arg in args:
            text = str(arg)
            safe_text = text.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore")
            fallback_args.append(safe_text)
        builtins.print(*fallback_args, **kwargs)


# Keep all module prints safe on Windows terminals with non-UTF8 encodings.
print = _safe_print  # type: ignore

# GPU优先使用 CuPy；可通过 CUDA_VISIBLE_DEVICES=-1 或 CUPY_DISABLE_CUDA=1 强制 CPU
enable_cupy = os.environ.get("DAERWEN_ENABLE_CUPY") == "1"
force_cpu = (
    not enable_cupy
    or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    or os.environ.get("CUPY_DISABLE_CUDA") == "1"
    or os.environ.get("DAERWEN_FORCE_CPU") == "1"
)
GPU_AVAILABLE = False
cp = None

if not force_cpu:
    try:
        import cupy as cp  # type: ignore
        try:
            # 触发一次轻量GPU操作，确保设备可用
            _probe = cp.zeros((1,), dtype=cp.float32)
            _probe.sum().item()
            GPU_AVAILABLE = True
            print("✅ GPU加速已启用 (CuPy)")
        except Exception:
            cp = None
            GPU_AVAILABLE = False
            print("⚠️ GPU不可用，使用CPU模式 (NumPy)")
    except Exception:
        print("⚠️ GPU不可用，使用CPU模式 (NumPy)")
else:
    print("⚠️ 强制使用CPU模式 (NumPy)")

import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import math

xp = cp if GPU_AVAILABLE else np

PHENOTYPE_KEYS = [
    # 原始 8 个表型（由基因组前 8 段编码）
    'field_interaction',
    'movement_response',
    'interaction_mode',
    'replication_threshold',
    'conversion_threshold',
    'interaction_threshold',
    'cooperation_threshold',
    'aging_resistance',
    # 第三层新增 4 个表型（由基因组扩展段编码，基因组 < 36 碱基时使用 Config 默认值）
    'inhibitor_sensitivity',      # 抑制剂伤害系数 [0, 0.1]
    'chemotaxis_gene_strength',   # 趋化力强度 [0, 0.15]
    'replication_energy_split',   # 子代能量比例 [0.3, 0.7]
    'atp_absorption_rate',        # ATP 吸收系数 [0.05, 0.25]
]

def _to_xp(array: Any, dtype: Optional[Any] = None):
    if GPU_AVAILABLE:
        return xp.asarray(array, dtype=dtype)
    return np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)

def _to_numpy(array: Any):
    if GPU_AVAILABLE and isinstance(array, xp.ndarray):  # type: ignore
        return cp.asnumpy(array)  # type: ignore[arg-type]
    return array

@dataclass
class Ecology2DConfig:
    """2D生态系统配置"""
    # 空间参数
    world_size: int = 200           # 2D正方形世界
    n_particles: int = 5000         # 粒子数量
    
    # 物理参数
    dt: float = 0.1                 # 时间步长
    diffusion_rate: float = 0.1     # 扩散率
    
    # 基因参数
    genome_length: int = 48         # 基因组长度（4碱基编码；32原始段 + 16扩展段）
    mutation_rate: float = 0.01
    
    # 化学参数
    n_chemical_species: int = 12    # 化学物质种类

    # 能量参数
    solar_energy_rate: float = 1.0
    metabolic_cost: float = 0.01
    atp_decay_rate: float = 5e-4
    waste_recovery_rate: float = 2e-4
    nutrient_decay_rate: float = 3e-4
    inhibitor_decay_rate: float = 2e-4
    spatial_gradient_strength: float = 0.2
    
    # 交互参数
    interaction_radius: float = 3.0 # 交互半径
    
    # 概率/交互调节
    max_interaction_samples: int = 300
    gene_exchange_scale: float = 0.1
    structural_variation_rate: float = 0.001
    adhesion_strength: float = 0.05
    cohesion_strength: float = 0.08
    repulsion_strength: float = 0.15
    gradient_drift_rate: float = 0.01

    # === 第一层：物理单位参数（原硬编码，现显式化）===
    solar_energy_scale: float = 0.001   # 太阳能实际注入系数（原匿名 * 0.001）
    solar_saturation: float = 1.0       # ATP 饱和阈值（修复 allowlist 孤儿）
    velocity_damping: float = 0.9       # 速度阻尼系数（原匿名 0.9）
    brownian_strength: float = 0.1      # 布朗运动强度（原匿名 0.1）
    chemotaxis_strength: float = 0.05   # 趋化力系数（原匿名 0.05；L3 中变为新表型默认值）
    atp_absorption_scale: float = 0.1   # ATP 吸收系数（原匿名 0.1）
    inhibitor_damage_coeff: float = 0.02 # 抑制剂伤害系数（原匿名 0.02；L3 中变为新表型默认值）
    interaction_interval: int = 10       # 交互计算间隔步数（原匿名 % 10）

    # === 第二层：守恒参数（原隐式，现显式化）===
    death_waste_release: float = 0.5    # 死亡时能量转废物比例（修复守恒 bug）
    predation_heat_loss: float = 0.6    # 捕食能量热损失率（原隐式 = 1 - 0.4）



@dataclass
class ExternalInput:
    """外部输入事件，用于驱动生态系统"""
    input_type: str
    params: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_type': self.input_type,
            'params': self.params,
            'metadata': self.metadata or {}
        }


@dataclass
class SystemOutput:
    """系统输出快照，供外部系统消费"""
    time_step: int
    timestamp: float
    visualization: Dict[str, Any]
    emergence: Dict[str, Any]
    stats: Dict[str, Any]
    recent_inputs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        particles = self.visualization.get('particles', {})
        chemical_field = self.visualization.get('chemical_field', {})
        base = {
            'time_step': self.time_step,
            'timestamp': self.timestamp,
            'visualization': self.visualization,
            'particles': particles,
            'chemical_field': chemical_field,
            'stats': self.stats,
            'emergence': self.emergence,
            'recent_inputs': self.recent_inputs,
        }
        if self.metadata:
            base['metadata'] = self.metadata
        return base


class Particle2D:
    """2D粒子 - 最小生命单元"""
    
    def __init__(self, particle_id: int, position: np.ndarray, genome: np.ndarray,
                 system_ref: Optional["Ecology2DSystem"] = None):
        self.id = particle_id
        self.position = position  # (x, y)
        self.velocity = np.zeros(2)
        self.genome = genome      # 四碱基序列 [0,1,2,3] = [A,T,G,C]
        self._system_ref = system_ref
        
        # 生物状态
        self.energy = 1.0
        self.age = 0
        self.generation = 0
        self.alive = True
        
        # 基因表达产物（表型）
        self.phenotype = self._express_genes()

    def _express_genes(self) -> Dict[str, float]:
        """Gene expression: base composition → 12 phenotype parameters.

        Uses a single uniform principle (chem_sim-inspired):
            phenotype_i = lerp(min_i, max_i, clip(2 * freq_of_base_i, 0, 1))

        This replaced the earlier 12-distinct-formula approach (preserved in
        `engine/_legacy_gene_expression.py`) because that version produced
        extreme boom-or-bust population dynamics (σ of final population ~373
        across seeds, vs ~225 here), making experiments non-reproducible.

        Behavior is further modulated by local chemical environment
        (atp/waste/nutrient gradient) via env_factor.
        """
        from engine.chem_sim_genes import express_phenotypes_from_composition
        phenotype = express_phenotypes_from_composition(self.genome)

        # Environmental modulation (unchanged from before)
        if self._system_ref is not None:
            local_field = self._system_ref.chemical_field.get_local_concentration(self.position)
            atp_level = float(local_field[self._system_ref.chemical_field.ATP_index])
            waste_level = float(local_field[self._system_ref.chemical_field.waste_index])
            nutrient_level = float(local_field[self._system_ref.chemical_field.nutrient_index])
            env_pressure = np.clip(atp_level - waste_level + nutrient_level, -1.0, 1.0)
            env_factor = 1.0 + 0.5 * env_pressure
            for key in phenotype.keys():
                if key not in {'aging_resistance', 'movement_response', 'interaction_mode'}:
                    phenotype[key] *= env_factor
        return phenotype
    
    
    def can_replicate(self, threshold_override: Optional[float] = None) -> bool:
        """判断是否能复制"""
        threshold = threshold_override if threshold_override is not None else self.phenotype['replication_threshold']
        return self.energy >= threshold and self.alive


class ChemicalField2D:
    """2D化学场 - 管理所有化学物质的浓度分布"""
    
    def __init__(
        self,
        world_size: int,
        n_species: int,
        rng: Optional[np.random.Generator] = None,
        gradient_drift_rate: float = 0.0,
        initial_atp_level: float = 0.2,
        initial_nutrient_level: float = 0.05,
        initial_inhibitor_level: float = 0.01,
        initial_waste_level: float = 0.0,
        initial_random_scale: float = 0.02,
    ):
        self.world_size = world_size
        self.n_species = n_species
        self._rng = rng or np.random.default_rng()
        self.gradient_drift_rate = gradient_drift_rate
        
        # 为每种化学物质命名
        base_names = ["ATP", "Nutrient", "Inhibitor"]
        self.species_names = [
            base_names[i] if i < len(base_names) else f"Chem_{i}"
            for i in range(n_species)
        ]
        # 特殊物质索引
        self.ATP_index = 0  # ATP = 能量货币
        self.nutrient_index = min(1, n_species - 1)
        self.inhibitor_index = min(2, n_species - 1)
        self.waste_index = n_species - 1

        # 化学浓度场 [world_size, world_size, n_species]
        base = np.zeros((world_size, world_size, n_species), dtype=np.float32)
        base[:, :, self.ATP_index] += float(initial_atp_level)
        base[:, :, self.nutrient_index] += float(initial_nutrient_level)
        base[:, :, self.inhibitor_index] += float(initial_inhibitor_level)
        base[:, :, self.waste_index] += float(initial_waste_level)
        if initial_random_scale and initial_random_scale > 0:
            base += self._rng.uniform(
                0, float(initial_random_scale), (world_size, world_size, n_species)
            ).astype(np.float32)
        base = np.clip(base, 0.0, None)
        self.concentrations = _to_xp(base)

        self._gradient_phase = self._rng.random(2) * 2 * np.pi
        self._gradient_drift_scale = self._rng.uniform(0.5, 1.5, size=2)

        coords = np.linspace(0, 2 * np.pi, world_size, endpoint=False, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(coords, coords, indexing='ij')
        self._grid_x = _to_xp(grid_x)
        self._grid_y = _to_xp(grid_y)
        self._cached_gradient_step = -1
        self._cached_gradient = (None, None)

    def invalidate_gradient_cache(self):
        self._cached_gradient_step = -1
        self._cached_gradient = (None, None)
    
    def diffuse(self, dt: float, diffusion_rate: float):
        """扩散方程：∂C/∂t = D∇²C（周期边界）"""
        if diffusion_rate <= 0 or dt <= 0:
            return
        field = self.concentrations
        laplacian = (
            xp.roll(field, 1, axis=0) + xp.roll(field, -1, axis=0) +
            xp.roll(field, 1, axis=1) + xp.roll(field, -1, axis=1) -
            4.0 * field
        )
        field += diffusion_rate * laplacian * dt
        xp.maximum(field, 0, out=field)
        self.invalidate_gradient_cache()
    
    def add_solar_energy(self, rate: float, saturation: float = 1.0):
        """太阳能持续输入；按平均浓度限流，避免能量无限累积"""
        atp_layer = self.concentrations[:, :, self.ATP_index]
        current_mean = xp.mean(atp_layer)

        if saturation > 0:
            if current_mean >= saturation:
                return
            scale = float(max(0.0, 1.0 - current_mean / saturation))
        else:
            scale = 1.0

        atp_layer += rate * scale
        atp_layer[:] = xp.minimum(atp_layer, 2.0)  # 局部封顶
        self.invalidate_gradient_cache()

    def apply_energy_cycle(
        self,
        atp_decay_rate: float,
        waste_recovery_rate: float,
        nutrient_decay_rate: float = 0.0,
        inhibitor_decay_rate: float = 0.0,
    ):
        """
        基础能量循环：
        1. ATP 会自然衰减为废物（模拟热散失）
        2. 废物以较慢速度被生态重新利用，生成一部分ATP
        这样能量不会无限累积，同时提供缓慢的资源再生。
        """
        if (
            atp_decay_rate <= 0
            and waste_recovery_rate <= 0
            and nutrient_decay_rate <= 0
            and inhibitor_decay_rate <= 0
        ):
            return

        atp_layer = self.concentrations[:, :, self.ATP_index]
        waste_layer = self.concentrations[:, :, self.waste_index]
        nutrient_layer = self.concentrations[:, :, self.nutrient_index]
        inhibitor_layer = self.concentrations[:, :, self.inhibitor_index]

        if atp_decay_rate > 0:
            decay_amount = xp.minimum(atp_layer, atp_layer * atp_decay_rate)
            atp_layer -= decay_amount
            waste_layer += decay_amount

        if waste_recovery_rate > 0:
            recovered = waste_layer * waste_recovery_rate
            waste_layer -= recovered
            efficiency = 0.5 + 0.4 * xp.clip(nutrient_layer, 0.0, 1.0)
            self.concentrations[:, :, self.ATP_index] += recovered * efficiency

        # 营养缓慢衰减，抑制物持续压制
        if nutrient_decay_rate > 0:
            nutrient_layer[:] = xp.maximum(nutrient_layer - nutrient_decay_rate, 0.0)
        if inhibitor_decay_rate > 0:
            inhibitor_layer[:] = xp.maximum(inhibitor_layer - inhibitor_decay_rate, 0.0)
        self.invalidate_gradient_cache()

    def apply_environmental_gradients(self, strength: float, time_step: int):
        """在营养/抑制物上施加空间梯度，形成可供观测的慢变量"""
        if strength <= 0:
            return

        grid_x = self._grid_x
        grid_y = self._grid_y
        drift_x = self.gradient_drift_rate * self._gradient_drift_scale[0]
        drift_y = self.gradient_drift_rate * self._gradient_drift_scale[1]
        phase_x = self._gradient_phase[0] + time_step * drift_x
        phase_y = self._gradient_phase[1] + time_step * drift_y

        nutrient_pattern = 0.5 + 0.5 * (
            xp.sin(grid_x + phase_x) + xp.cos(grid_y + phase_y)
        ) / 2.0
        inhibitor_pattern = 0.5 + 0.5 * (
            xp.cos(grid_x * 1.5 + phase_y) * xp.sin(grid_y * 0.75 + phase_x)
        )

        nutrient_layer = self.concentrations[:, :, self.nutrient_index]
        inhibitor_layer = self.concentrations[:, :, self.inhibitor_index]
        nutrient_layer += strength * (nutrient_pattern - 0.5)
        inhibitor_layer += strength * (inhibitor_pattern - 0.5)

        # 约束范围
        xp.clip(nutrient_layer, 0.0, 1.5, out=nutrient_layer)
        xp.clip(inhibitor_layer, 0.0, 1.0, out=inhibitor_layer)
        self.invalidate_gradient_cache()
    
    def get_local_concentration(self, position: np.ndarray) -> np.ndarray:
        """获取指定位置的化学浓度（双线性插值 + 周期边界）"""
        x = float(position[0]) % self.world_size
        y = float(position[1]) % self.world_size
        
        x0 = int(math.floor(x)) % self.world_size
        y0 = int(math.floor(y)) % self.world_size
        x1 = (x0 + 1) % self.world_size
        y1 = (y0 + 1) % self.world_size
        
        tx = x - math.floor(x)
        ty = y - math.floor(y)
        
        field = self.concentrations
        c00 = field[x0, y0, :]
        c10 = field[x1, y0, :]
        c01 = field[x0, y1, :]
        c11 = field[x1, y1, :]
        
        top = c00 * (1 - tx) + c10 * tx
        bottom = c01 * (1 - tx) + c11 * tx
        return _to_numpy(top * (1 - ty) + bottom * ty)

    def sample_concentrations_batch(self, positions: np.ndarray) -> np.ndarray:
        """批量采样多个位置的化学浓度，返回 [N, n_species]"""
        if positions.size == 0:
            return np.zeros((0, self.n_species), dtype=float)

        pos = np.asarray(positions, dtype=float)
        x = np.mod(pos[:, 0], self.world_size)
        y = np.mod(pos[:, 1], self.world_size)

        fx = np.floor(x)
        fy = np.floor(y)
        tx = xp.asarray((x - fx)[:, np.newaxis])
        ty = xp.asarray((y - fy)[:, np.newaxis])

        x0 = xp.asarray(fx.astype(int) % self.world_size)
        y0 = xp.asarray(fy.astype(int) % self.world_size)
        x1 = (x0 + 1) % self.world_size
        y1 = (y0 + 1) % self.world_size

        field = self.concentrations
        c00 = field[x0, y0, :]
        c10 = field[x1, y0, :]
        c01 = field[x0, y1, :]
        c11 = field[x1, y1, :]

        top = c00 * (1.0 - tx) + c10 * tx
        bottom = c01 * (1.0 - tx) + c11 * tx
        return _to_numpy(top * (1.0 - ty) + bottom * ty)

    def _sample_scalar_field(self, field: np.ndarray, position: np.ndarray) -> float:
        """双线性插值采样单通道场"""
        x = float(position[0]) % self.world_size
        y = float(position[1]) % self.world_size
        x0 = int(math.floor(x)) % self.world_size
        y0 = int(math.floor(y)) % self.world_size
        x1 = (x0 + 1) % self.world_size
        y1 = (y0 + 1) % self.world_size
        tx = x - math.floor(x)
        ty = y - math.floor(y)
        c00 = field[x0, y0]
        c10 = field[x1, y0]
        c01 = field[x0, y1]
        c11 = field[x1, y1]
        top = c00 * (1 - tx) + c10 * tx
        bottom = c01 * (1 - tx) + c11 * tx
        result = top * (1 - ty) + bottom * ty
        return float(_to_numpy(result))

    def _sample_scalar_field_batch(self, field: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """批量采样单通道场"""
        if positions.size == 0:
            return np.zeros((0,), dtype=float)
        pos = np.asarray(positions, dtype=float)
        x = np.mod(pos[:, 0], self.world_size)
        y = np.mod(pos[:, 1], self.world_size)
        fx = np.floor(x)
        fy = np.floor(y)
        tx = x - fx
        ty = y - fy
        x0 = fx.astype(int) % self.world_size
        y0 = fy.astype(int) % self.world_size
        x1 = (x0 + 1) % self.world_size
        y1 = (y0 + 1) % self.world_size
        c00 = field[x0, y0]
        c10 = field[x1, y0]
        c01 = field[x0, y1]
        c11 = field[x1, y1]
        top = c00 * (1 - tx)
        top += c10 * tx
        bottom = c01 * (1 - tx)
        bottom += c11 * tx
        return _to_numpy(top * (1 - ty) + bottom * ty)

    def get_atp_gradient(self, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取ATP场梯度（缓存到时间步级别）"""
        if self._cached_gradient_step == time_step:
            grad_x, grad_y = self._cached_gradient
            if grad_x is not None and grad_y is not None:
                return grad_x, grad_y
        atp_layer = self.concentrations[:, :, self.ATP_index]
        grad_x = 0.5 * (xp.roll(atp_layer, -1, axis=0) - xp.roll(atp_layer, 1, axis=0))
        grad_y = 0.5 * (xp.roll(atp_layer, -1, axis=1) - xp.roll(atp_layer, 1, axis=1))
        self._cached_gradient_step = time_step
        self._cached_gradient = (grad_x, grad_y)
        return grad_x, grad_y

    def sample_gradient(self, position: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        return np.array([
            self._sample_scalar_field(grad_x, position),
            self._sample_scalar_field(grad_y, position)
        ], dtype=float)

    def sample_gradient_batch(self, positions: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """批量采样梯度"""
        if positions.size == 0:
            return np.zeros((0, 2), dtype=float)
        gx = self._sample_scalar_field_batch(grad_x, positions)
        gy = self._sample_scalar_field_batch(grad_y, positions)
        return np.stack([gx, gy], axis=1)
    
    def consume(self, position: np.ndarray, species_index: int, amount: float):
        """粒子消耗化学物质"""
        x, y = int(position[0]), int(position[1])
        x = max(0, min(x, self.world_size - 1))
        y = max(0, min(y, self.world_size - 1))
        
        available = float(self.concentrations[x, y, species_index])
        consumed = min(available, amount)
        self.concentrations[x, y, species_index] -= consumed
        self.invalidate_gradient_cache()
        return consumed
    
    def produce(self, position: np.ndarray, species_index: int, amount: float):
        """粒子产生化学物质"""
        x, y = int(position[0]), int(position[1])
        x = max(0, min(x, self.world_size - 1))
        y = max(0, min(y, self.world_size - 1))
        
        self.concentrations[x, y, species_index] += amount
        self.invalidate_gradient_cache()


class Ecology2DSystem:
    """2D生态系统 - 核心整合器"""
    
    def __init__(self, config: Ecology2DConfig):
        self.config = config
        self.time_step = 0
        self.rng = np.random.default_rng()
        
        # 化学场
        self.chemical_field = ChemicalField2D(
            config.world_size,
            config.n_chemical_species,
            rng=self.rng,
            gradient_drift_rate=config.gradient_drift_rate,
        )
        
        # 粒子列表
        self.particles: List[Particle2D] = []
        self.next_particle_id = 0  # 全局递增ID计数器，防止ID冲突
        self._initialize_particles()
        
        # 统计数据
        history_window = max(1200, config.world_size * 4)
        self.stats = {
            'total_particles': deque(maxlen=history_window),
            'alive_particles': deque(maxlen=history_window),
            'total_energy': deque(maxlen=history_window),
            'replication_events': 0,
            'death_events': 0,
            'interaction_events': 0,
        }
        self.input_history: List[Dict[str, Any]] = []

        print(f"✅ 2D生态系统初始化完成")
        print(f"   世界大小: {config.world_size}×{config.world_size}")
        print(f"   初始粒子: {len(self.particles)}")
        print(f"   化学物质种类: {config.n_chemical_species}")
    
    @property
    def fields(self) -> Dict[str, np.ndarray]:
        """Access chemical fields by name (ATP, Nutrient, Inhibitor, etc.)"""
        result = {}
        for i, name in enumerate(self.chemical_field.species_names):
            result[name.lower()] = self.chemical_field.concentrations[:, :, i]
        return result
    
    def _initialize_particles(self):
        """初始化粒子种群"""
        for i in range(self.config.n_particles):
            # 随机位置
            position = self.rng.uniform(
                0, self.config.world_size, size=2
            )
            
            # 随机基因组（四碱基）
            genome = self.rng.integers(0, 4, size=self.config.genome_length)
            
            particle = Particle2D(self.next_particle_id, position, genome, self)
            self.particles.append(particle)
            self.next_particle_id += 1
    
    def _sample_particles(self, particles: List[Particle2D], sample_size: int) -> List[Particle2D]:
        """从粒子列表中无放回采样；在数量不足或为空时安全返回"""
        if sample_size <= 0 or not particles:
            return []
        if sample_size >= len(particles):
            return list(particles)
        indices = self.rng.choice(len(particles), size=sample_size, replace=False)
        if np.isscalar(indices):
            indices = [int(indices)]
        return [particles[int(idx)] for idx in indices]
    
    def _build_particle_state(self, particles: List[Particle2D]) -> Dict[str, Any]:
        count = len(particles)
        positions = np.zeros((count, 2), dtype=np.float32)
        velocities = np.zeros((count, 2), dtype=np.float32)
        energies = np.zeros(count, dtype=np.float32)
        ages = np.zeros(count, dtype=np.float32)
        phenotypes = {
            key: np.zeros(count, dtype=np.float32)
            for key in PHENOTYPE_KEYS
        }
        id_to_index = {}
        
        for idx, particle in enumerate(particles):
            positions[idx] = particle.position
            velocities[idx] = particle.velocity
            energies[idx] = particle.energy
            ages[idx] = particle.age
            id_to_index[particle.id] = idx
            for key in PHENOTYPE_KEYS:
                phenotypes[key][idx] = float(particle.phenotype.get(key, 0.0))
        
        return {
            'count': count,
            'positions': positions,
            'velocities': velocities,
            'energies': energies,
            'ages': ages,
            'phenotypes': phenotypes,
            'id_index': id_to_index,
        }
    
    def step(self):
        """执行一个时间步 - 这是整个系统运行的核心"""
        dt = self.config.dt
        
        # 1. 化学场扩散（每5步执行一次，降低计算量）
        if self.time_step % 5 == 0:
            self.chemical_field.diffuse(dt * 5, self.config.diffusion_rate)
        
        # 2. 太阳能输入（持续能量源）
        self.chemical_field.add_solar_energy(
            self.config.solar_energy_rate * self.config.solar_energy_scale,
            self.config.solar_saturation,
        )
        self.chemical_field.apply_environmental_gradients(
            self.config.spatial_gradient_strength,
            self.time_step
        )

        # 2.5 基础能量循环（避免能量无限累积）
        self.chemical_field.apply_energy_cycle(
            self.config.atp_decay_rate,
            self.config.waste_recovery_rate,
            self.config.nutrient_decay_rate,
            self.config.inhibitor_decay_rate,
        )

        alive_particles = [p for p in self.particles if p.alive]
        state = self._build_particle_state(alive_particles)

        # 3. 粒子新陈代谢（核心：能量-物质转化）
        self._particle_metabolism(alive_particles, state)
        
        # 4. 粒子间相互作用（让生态关系涌现）
        self._particle_interactions(alive_particles, state)
        
        # 5. 粒子移动
        self._particle_movement(alive_particles, state, dt)
        
        # 6. 复制（自我繁殖）
        self._particle_replication(alive_particles, state)
        
        # 7. 死亡和清理
        alive_particles = [p for p in self.particles if p.alive]
        state = self._build_particle_state(alive_particles)
        self._particle_death(alive_particles, state)
        
        # 8. 统计
        alive_particles = [p for p in self.particles if p.alive]
        state = self._build_particle_state(alive_particles)
        self._update_statistics(alive_particles, state)
        
        self.time_step += 1
    
    def _particle_metabolism(self, alive_particles: List[Particle2D], state: Dict[str, Any]):
        """粒子与场的能量交换"""
        if not alive_particles:
            return

        positions = state['positions']
        local_fields = self.chemical_field.sample_concentrations_batch(positions)
        field_interactions = state['phenotypes']['field_interaction']
        conversion_thresholds = state['phenotypes']['conversion_threshold']
        movement_responses = state['phenotypes']['movement_response']
        energies = state['energies']
        ages = state['ages']

        for idx, (particle, local_field) in enumerate(zip(alive_particles, local_fields)):
            absorption_rate = field_interactions[idx]

            if absorption_rate > 0:
                absorbed = self.chemical_field.consume(
                    particle.position,
                    self.chemical_field.ATP_index,
                    abs(absorption_rate) * particle.phenotype.get('atp_absorption_rate', self.config.atp_absorption_scale)
                )
                particle.energy += absorbed
            else:
                released = min(particle.energy * 0.05, particle.energy * 0.5)
                if released > 0:
                    self.chemical_field.produce(
                        particle.position,
                        self.chemical_field.ATP_index,
                        released
                    )
                    particle.energy -= released

            conversion_threshold = conversion_thresholds[idx]
            if conversion_threshold > 0.3:
                waste = local_field[self.chemical_field.waste_index]
                if waste > conversion_threshold * 0.5:
                    converted = self.chemical_field.consume(
                        particle.position,
                        self.chemical_field.waste_index,
                        0.05
                    )
                    self.chemical_field.produce(
                        particle.position,
                        self.chemical_field.ATP_index,
                        converted * conversion_threshold
                    )

            temp_cost = self.config.metabolic_cost
            particle.energy -= temp_cost

            nutrient_available = local_field[self.chemical_field.nutrient_index]
            if nutrient_available > 0.01:
                nutrient_gain = self.chemical_field.consume(
                    particle.position,
                    self.chemical_field.nutrient_index,
                    min(0.05, nutrient_available * 0.1)
                )
                particle.energy += nutrient_gain * (0.4 + movement_responses[idx] * 0.02)

            inhibitor_level = local_field[self.chemical_field.inhibitor_index]
            if inhibitor_level > 0.01:
                penalty = inhibitor_level * particle.phenotype.get('inhibitor_sensitivity', self.config.inhibitor_damage_coeff)
                particle.energy -= penalty
            
            particle.age += 1
            energies[idx] = particle.energy
            ages[idx] = particle.age
    
    def _particle_interactions(self, alive_particles: List[Particle2D], state: Dict[str, Any]):
        """粒子间相互作用 - 让捕食、共生等关系涌现"""
        # 每 interaction_interval 步执行一次交互检测（性能优化）
        if self.time_step % self.config.interaction_interval != 0:
            return
            
        alive_count = len(alive_particles)
        if alive_count < 2:
            return

        positions = state['positions']
        energies = state['energies']
        radius_sq = float(self.config.interaction_radius ** 2)
        sample_size = min(self.config.max_interaction_samples, alive_count)
        if sample_size <= 0:
            return
        sample_indices = self.rng.choice(alive_count, size=sample_size, replace=False)
        if np.isscalar(sample_indices):
            sample_indices = [int(sample_indices)]
        max_neighbors = 64

        # 构建空间哈希以减少候选
        spatial_hash = defaultdict(list)
        grid_size = self.config.interaction_radius
        for idx, particle in enumerate(alive_particles):
            grid_x = int(particle.position[0] / grid_size)
            grid_y = int(particle.position[1] / grid_size)
            spatial_hash[(grid_x, grid_y)].append(idx)

        for base_idx in sample_indices:
            base_idx = int(base_idx)
            particle_i = alive_particles[base_idx]
            grid_x = int(particle_i.position[0] / grid_size)
            grid_y = int(particle_i.position[1] / grid_size)
            neighbor_idx_list: List[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor_idx_list.extend(spatial_hash.get((grid_x + dx, grid_y + dy), []))

            if not neighbor_idx_list:
                continue

            neighbor_indices = np.array(neighbor_idx_list, dtype=int)
            base_pos = positions[base_idx]
            neighbor_positions = positions[neighbor_indices]
            offsets = neighbor_positions - base_pos
            dist_sq = np.sum(offsets * offsets, axis=1)
            valid_idx = np.where((dist_sq > 0.0) & (dist_sq < radius_sq))[0]
            if valid_idx.size == 0:
                continue

            if valid_idx.size > max_neighbors:
                valid_idx = self.rng.choice(valid_idx, size=max_neighbors, replace=False)

            distances = np.sqrt(dist_sq[valid_idx])
            for idx_in_list, distance in zip(valid_idx, distances):
                neighbor_idx = neighbor_indices[int(idx_in_list)]
                particle_j = alive_particles[neighbor_idx]
                if particle_i.id >= particle_j.id:
                    continue
                self._interact(particle_i, particle_j, float(distance))
                energies[base_idx] = particle_i.energy
                energies[neighbor_idx] = particle_j.energy
                self.stats['interaction_events'] += 1
    
    def _interact(self, particle_i: Particle2D, particle_j: Particle2D, distance: float):
        """两个粒子的交互 - 纯物理化学规则，无预设生态关系"""
        
        # 计算基因相似性（处理不同长度的基因）
        len_i = len(particle_i.genome)
        len_j = len(particle_j.genome)
        min_len = min(len_i, len_j)
        
        # 只比较共同长度的部分
        genome_diff = float(np.sum(np.abs(particle_i.genome[:min_len] - particle_j.genome[:min_len])))
        # 长度差异也算作差异
        length_penalty = abs(len_i - len_j)
        total_diff = genome_diff + length_penalty
        
        genome_similarity = 1.0 / (1.0 + total_diff / max(len_i, len_j))
        
        # 交互模式由基线倾向 + 局部环境共同决定（降低“基因直接控制行为”）
        mode_i = self._effective_interaction_mode(particle_i)
        mode_j = self._effective_interaction_mode(particle_j)
        
        # 交互强度：基因差异越大，交互越强烈
        interaction_strength = (1.0 - genome_similarity) * abs(mode_i - mode_j)
        base_distance = max(distance, 1e-3)
        
        # 使用基因编码的阈值
        threshold_i = particle_i.phenotype['interaction_threshold']
        threshold_j = particle_j.phenotype['interaction_threshold']
        avg_threshold = (threshold_i + threshold_j) / 2.0
        
        if interaction_strength > avg_threshold:
            # 能量转移方向由interaction_mode决定
            mode_diff = mode_i - mode_j
            
            if abs(mode_diff) > avg_threshold:  # 显著差异（阈值由基因决定）
                # mode高的从mode低的获取能量
                # 第二层：predation_heat_loss 显式化捕食热损失（原隐式 1 - 0.4 = 0.6）
                if mode_diff > 0:
                    transfer_amount = 0.1 * particle_j.energy * interaction_strength
                    particle_i.energy += transfer_amount * (1.0 - self.config.predation_heat_loss)
                    particle_j.energy -= transfer_amount
                else:
                    transfer_amount = 0.1 * particle_i.energy * interaction_strength
                    particle_j.energy += transfer_amount * (1.0 - self.config.predation_heat_loss)
                    particle_i.energy -= transfer_amount
        
        # 基因交换（相似性和合作倾向都由基因决定）
        coop_threshold_i = particle_i.phenotype['cooperation_threshold']
        coop_threshold_j = particle_j.phenotype['cooperation_threshold']
        avg_coop = (coop_threshold_i + coop_threshold_j) / 2.0
        
        if genome_similarity > avg_coop and mode_i > 0 and mode_j > 0:
            exchange_prob = min(
                1.0,
                self.config.gene_exchange_scale * avg_coop * genome_similarity * (1.0 + interaction_strength)
            )
            if self.rng.random() < exchange_prob:
                min_len = min(len(particle_i.genome), len(particle_j.genome))
                if min_len > 0:
                    exchange_point = int(self.rng.integers(min_len))
                    particle_i.genome[exchange_point], particle_j.genome[exchange_point] = \
                        particle_j.genome[exchange_point], particle_i.genome[exchange_point]
                    
                    # 重新表达基因
                    particle_i.phenotype = particle_i._express_genes()
                    particle_j.phenotype = particle_j._express_genes()

        # 物理势能：距离很近且合作倾向较高时产生粘附与能量共享
        adhesion_factor = self.config.adhesion_strength * avg_coop
        if base_distance < self.config.interaction_radius * 0.5 and adhesion_factor > 0.0:
            overlap = max(0.0, self.config.interaction_radius * 0.5 - base_distance)
            adhesion_force = adhesion_factor * (overlap / (self.config.interaction_radius * 0.5))

            # 能量向平衡移动（类似共生体共享资源）
            energy_diff = particle_i.energy - particle_j.energy
            transfer = energy_diff * 0.15 * adhesion_force
            particle_i.energy -= transfer
            particle_j.energy += transfer

            # 缓慢让它们靠近（为后续突变提供聚集机会）
            direction = (particle_j.position - particle_i.position) / base_distance
            adhesion_velocity = direction * adhesion_force * 0.05
            particle_i.velocity += adhesion_velocity
            particle_j.velocity -= adhesion_velocity

        # 统一短程势能（类似柔性 Lennard-Jones）
        potential_range = self.config.interaction_radius
        if base_distance < potential_range:
            norm_dir = (particle_j.position - particle_i.position) / base_distance
            normalized = max(0.0, (potential_range - base_distance) / potential_range)
            cohesion_force = self.config.cohesion_strength * genome_similarity * normalized
            repulsion_force = self.config.repulsion_strength / (base_distance + 0.5)
            net_force = cohesion_force - repulsion_force
            particle_i.velocity += norm_dir * net_force
            particle_j.velocity -= norm_dir * net_force
        
        # 物理排斥力（防止重叠）
        if float(distance) < 1.0:
            repulsion_direction = (particle_i.position - particle_j.position) / (distance + 0.01)
            particle_i.velocity += repulsion_direction * 0.1
            particle_j.velocity -= repulsion_direction * 0.1
    
    def _particle_movement(self, alive_particles: List[Particle2D], state: Dict[str, Any], dt: float):
        """粒子移动 - 布朗运动 + 场梯度响应"""
        grad_x, grad_y = self.chemical_field.get_atp_gradient(self.time_step)
        if not alive_particles:
            return
        positions = state['positions']
        velocities = state['velocities']
        gradients = self.chemical_field.sample_gradient_batch(positions, grad_x, grad_y)
        local_fields = self.chemical_field.sample_concentrations_batch(positions)
        field_interactions = state['phenotypes']['field_interaction']
        movement_responses = state['phenotypes']['movement_response']

        for idx, (particle, gradient) in enumerate(zip(alive_particles, gradients)):
            movement_resp = self._effective_movement_response(movement_responses[idx], local_fields[idx])
            if movement_resp <= 0:
                continue

            random_force = self.rng.normal(0, self.config.brownian_strength, size=2)
            field_interaction = field_interactions[idx]
            gradient_force = gradient * field_interaction * movement_resp * particle.phenotype.get('chemotaxis_gene_strength', self.config.chemotaxis_strength)

            particle.velocity += (random_force + gradient_force) * dt
            particle.velocity *= self.config.velocity_damping
            particle.position += particle.velocity * dt
            particle.position = np.mod(particle.position, self.config.world_size)
            velocities[idx] = particle.velocity
            positions[idx] = particle.position

    def _effective_movement_response(self, base_response: float, local_field: np.ndarray) -> float:
        """将基因编码的移动敏感度映射为局部环境下的有效移动响应。"""
        atp = float(local_field[self.chemical_field.ATP_index])
        nutrient = float(local_field[self.chemical_field.nutrient_index])
        inhibitor = float(local_field[self.chemical_field.inhibitor_index])
        field_drive = np.clip(0.4 + 0.5 * atp + 0.3 * nutrient - 0.7 * inhibitor, 0.0, 2.5)
        return float(np.clip(base_response * field_drive, 0.0, 3.0))

    def _effective_interaction_mode(self, particle: Particle2D) -> float:
        """交互模式 = 基因基线倾向 + 局部化学/能量状态调制。"""
        base_mode = float(particle.phenotype['interaction_mode'])
        local_field = self.chemical_field.get_local_concentration(particle.position)
        nutrient = float(local_field[self.chemical_field.nutrient_index])
        inhibitor = float(local_field[self.chemical_field.inhibitor_index])
        atp = float(local_field[self.chemical_field.ATP_index])
        energy_signal = np.tanh((particle.energy - 1.0) * 0.8)
        chemistry_signal = np.tanh((nutrient + 0.5 * atp - inhibitor) * 1.2)
        return float(np.tanh(base_mode + 0.25 * energy_signal + 0.35 * chemistry_signal))
    
    def _particle_replication(self, alive_particles: List[Particle2D], state: Dict[str, Any]):
        """粒子复制 - 能量足够时自我繁殖"""
        new_particles = []
        if not alive_particles:
            return

        positions = state['positions']
        local_fields = self.chemical_field.sample_concentrations_batch(positions)

        base_thresholds = np.array([p.phenotype['replication_threshold'] for p in alive_particles], dtype=float)
        energies = state['energies']
        can_replicate_mask = energies >= base_thresholds
        replicator_indices = np.nonzero(can_replicate_mask)[0]

        for idx in replicator_indices:
            particle = alive_particles[int(idx)]

            offspring_position = particle.position + self.rng.normal(0, 1, size=2)
            offspring_position = np.mod(offspring_position, self.config.world_size)

            offspring_genome = particle.genome.copy()

            for i in range(len(offspring_genome)):
                if self.rng.random() < self.config.mutation_rate:
                    offspring_genome[i] = int(self.rng.integers(0, 4))

            variation_prob = min(
                0.5,
                self.config.structural_variation_rate * (1.0 + max(particle.energy, 0.0))
            )
            if self.rng.random() < variation_prob:
                variation_type = int(self.rng.integers(0, 3))
                
                if variation_type == 0 and len(offspring_genome) > 5:
                    start = int(self.rng.integers(0, len(offspring_genome) - 3))
                    length = int(self.rng.integers(2, min(8, len(offspring_genome) - start)))
                    segment = offspring_genome[start:start+length].copy()
                    insert_pos = int(self.rng.integers(0, len(offspring_genome)))
                    offspring_genome = np.concatenate([
                        offspring_genome[:insert_pos],
                        segment,
                        offspring_genome[insert_pos:]
                    ])
                
                elif variation_type == 1:
                    insert_length = int(self.rng.integers(1, 4))
                    new_bases = self.rng.integers(0, 4, size=insert_length)
                    insert_pos = int(self.rng.integers(0, len(offspring_genome)))
                    offspring_genome = np.concatenate([
                        offspring_genome[:insert_pos],
                        new_bases,
                        offspring_genome[insert_pos:]
                    ])
                
                elif variation_type == 2 and len(offspring_genome) > 15:
                    delete_length = int(self.rng.integers(1, min(6, len(offspring_genome) - 10)))
                    delete_pos = int(self.rng.integers(0, len(offspring_genome) - delete_length))
                    offspring_genome = np.concatenate([
                        offspring_genome[:delete_pos],
                        offspring_genome[delete_pos + delete_length:]
                    ])
            
            offspring = Particle2D(
                self.next_particle_id,
                offspring_position,
                offspring_genome,
                self
            )
            self.next_particle_id += 1
            offspring.generation = particle.generation + 1
            _split = particle.phenotype.get('replication_energy_split', 0.5)
            offspring.energy = particle.energy * _split

            particle.energy *= (1.0 - _split)
            energies[idx] = particle.energy
            
            new_particles.append(offspring)
            self.stats['replication_events'] += 1
        
        # 添加新粒子
        self.particles.extend(new_particles)
    
    def _particle_death(self, alive_particles: List[Particle2D], state: Dict[str, Any]):
        """粒子死亡 - 能量耗尽或衰老损伤"""
        if not alive_particles:
            return

        energies = state['energies']
        ages = state['ages']
        aging_resistance = state['phenotypes']['aging_resistance']

        aging_denominator = np.maximum(100.0, aging_resistance * 500.0)
        aging_damage = ages / aging_denominator

        energy_depleted = energies <= 0.0
        aging_failed = aging_damage > energies
        death_mask = energy_depleted | aging_failed

        if not np.any(death_mask):
            if self.time_step % 100 == 0:
                self.particles = [p for p in self.particles if p.alive]
            return

        indices = np.nonzero(death_mask)[0]
        for idx in indices:
            particle = alive_particles[int(idx)]
            if not particle.alive:
                continue
            particle.alive = False
            self.stats['death_events'] += 1

            # 第二层：守恒修复 —— 统一用 death_waste_release 比例
            # energy_depleted 时 energy ≤ 0，max(0, energy) = 0，waste = 0（守恒）
            # aging_death 时 energy > 0，waste = energy * release（与原逻辑一致）
            waste = max(0.0, particle.energy) * self.config.death_waste_release
            self.chemical_field.produce(
                particle.position,
                self.chemical_field.waste_index,
                waste
            )
        
        # 清理死亡粒子（保留一段时间后再删除）
        if self.time_step % 100 == 0:
            self.particles = [p for p in self.particles if p.alive]
    
    def _update_statistics(self, alive_particles: List[Particle2D], state: Dict[str, Any]):
        """更新统计数据"""
        alive_count = len(alive_particles)
        total_energy = float(np.sum(state['energies'])) if alive_count > 0 else 0.0
        
        self.stats['total_particles'].append(len(self.particles))
        self.stats['alive_particles'].append(alive_count)
        self.stats['total_energy'].append(total_energy)

    def apply_external_input(self, external_input: ExternalInput) -> bool:
        """应用来自外部的输入事件。返回是否成功应用。"""
        input_type = (external_input.input_type or '').lower()
        params = external_input.params or {}
        status = 'applied'
        reason = 'ok'

        if input_type == 'chemical_pulse':
            self._apply_chemical_pulse_input(params)
        elif input_type == 'gradient_field':
            self._apply_gradient_field_input(params)
        elif input_type == 'catastrophe':
            allowed_events = {'energy_fluctuation', 'mass_extinction', 'mutation_burst', 'random'}
            event = (params.get('event_type', 'energy_fluctuation') or 'energy_fluctuation').lower()
            if event not in allowed_events:
                status = 'rejected'
                reason = f'unsupported catastrophe event_type: {event}'
            else:
                self.trigger_disturbance_event(event)
        elif input_type == 'parameter_adjust':
            adjust_result = self._apply_parameter_adjust_input(params)
            applied_keys = adjust_result['applied']
            rejected_keys = adjust_result['rejected']
            if not applied_keys:
                status = 'rejected'
                if rejected_keys:
                    reason = f"no valid parameter_adjust keys: {', '.join(sorted(rejected_keys.keys()))}"
                else:
                    reason = 'empty parameter_adjust payload'
            elif rejected_keys:
                reason = f"partial parameter_adjust apply; rejected keys: {', '.join(sorted(rejected_keys.keys()))}"
        else:
            status = 'rejected'
            reason = f'unsupported input_type: {input_type or "<empty>"}'

        record = external_input.to_dict()
        record['applied_at'] = self.time_step
        record['status'] = status
        record['reason'] = reason
        self.input_history.append(record)
        self.input_history = self.input_history[-20:]
        return status == 'applied'

    def _apply_chemical_pulse_input(self, params: Dict[str, Any]):
        """在指定位置注入化学脉冲"""
        world = self.config.world_size
        x = int(params.get('x', world // 2)) % world
        y = int(params.get('y', world // 2)) % world
        radius = max(1, int(params.get('radius', 10)))
        intensity = float(params.get('intensity', 1.0))
        chemical_index = int(params.get('chemical_index', self.chemical_field.ATP_index))

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px = (x + dx) % world
                    py = (y + dy) % world
                    self.chemical_field.concentrations[px, py, chemical_index] += intensity
        self.chemical_field.invalidate_gradient_cache()

    def _apply_gradient_field_input(self, params: Dict[str, Any]):
        """创建或更新一个化学梯度场"""
        axis = params.get('axis', 'x')
        start_value = float(params.get('start_value', 0.1))
        end_value = float(params.get('end_value', 1.0))
        chemical_index = int(params.get('chemical_index', self.chemical_field.ATP_index))
        world_size = self.config.world_size

        if axis == 'y':
            for y in range(world_size):
                mix = y / max(1, world_size - 1)
                value = start_value * (1 - mix) + end_value * mix
                self.chemical_field.concentrations[:, y, chemical_index] = value
        else:
            for x in range(world_size):
                mix = x / max(1, world_size - 1)
                value = start_value * (1 - mix) + end_value * mix
                self.chemical_field.concentrations[x, :, chemical_index] = value
        self.chemical_field.invalidate_gradient_cache()

    def _apply_parameter_adjust_input(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """调整系统参数（受 allowlist 与范围限制）。"""
        allow_rules: Dict[str, Dict[str, Any]] = {
            'mutation_rate':              {'min': 0.0,   'max': 0.2,   'cast': float},
            'structural_variation_rate':  {'min': 0.0,   'max': 0.05,  'cast': float},
            'solar_energy_rate':          {'min': 0.0,   'max': 3.0,   'cast': float},  # 修复：原 max=0.2 是 bug
            'solar_energy_scale':         {'min': 1e-5,  'max': 0.01,  'cast': float},  # 新增
            'solar_saturation':           {'min': 0.1,   'max': 5.0,   'cast': float},  # 修复：现与 Config 同步
            'metabolic_cost':             {'min': 0.0,   'max': 1.0,   'cast': float},
            'atp_decay_rate':             {'min': 0.0,   'max': 0.1,   'cast': float},
            'waste_recovery_rate':        {'min': 0.0,   'max': 0.05,  'cast': float},
            'nutrient_decay_rate':        {'min': 0.0,   'max': 0.1,   'cast': float},
            'inhibitor_decay_rate':       {'min': 0.0,   'max': 0.1,   'cast': float},
            'diffusion_rate':             {'min': 0.0,   'max': 1.0,   'cast': float},
            'spatial_gradient_strength':  {'min': 0.0,   'max': 2.0,   'cast': float},
            'interaction_radius':         {'min': 0.5,   'max': 20.0,  'cast': float},
            'max_interaction_samples':    {'min': 10,    'max': 2000,  'cast': int},
            'gene_exchange_scale':        {'min': 0.0,   'max': 2.0,   'cast': float},
            'adhesion_strength':          {'min': 0.0,   'max': 2.0,   'cast': float},
            'cohesion_strength':          {'min': 0.0,   'max': 2.0,   'cast': float},
            'repulsion_strength':         {'min': 0.0,   'max': 3.0,   'cast': float},
            'gradient_drift_rate':        {'min': 0.0,   'max': 1.0,   'cast': float},
            # 第一层新增
            'velocity_damping':           {'min': 0.5,   'max': 0.99,  'cast': float},
            'brownian_strength':          {'min': 0.0,   'max': 1.0,   'cast': float},
            'chemotaxis_strength':        {'min': 0.0,   'max': 0.5,   'cast': float},
            'atp_absorption_scale':       {'min': 0.0,   'max': 1.0,   'cast': float},
            'inhibitor_damage_coeff':     {'min': 0.0,   'max': 0.2,   'cast': float},
            'interaction_interval':       {'min': 1,     'max': 100,   'cast': int},
            # 第二层新增
            'death_waste_release':        {'min': 0.0,   'max': 1.0,   'cast': float},
            'predation_heat_loss':        {'min': 0.0,   'max': 1.0,   'cast': float},
        }

        result: Dict[str, Dict[str, Any]] = {'applied': {}, 'rejected': {}}
        for key, value in params.items():
            rule = allow_rules.get(key)
            if rule is None:
                result['rejected'][key] = 'not allowlisted for runtime adjust'
                continue

            caster = rule['cast']
            try:
                cast_value = caster(value)
            except (TypeError, ValueError):
                result['rejected'][key] = f'invalid type: {type(value).__name__}'
                continue

            if cast_value < rule['min'] or cast_value > rule['max']:
                result['rejected'][key] = f'out of range [{rule["min"]}, {rule["max"]}]'
                continue

            setattr(self.config, key, cast_value)
            result['applied'][key] = cast_value

        return result

    def _get_recent_inputs(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.input_history[-limit:]

    def get_system_output(self, metadata: Optional[Dict[str, Any]] = None) -> SystemOutput:
        """获取结构化系统输出"""
        visualization = self.get_visualization_data()
        emergence = self.get_emergence_metrics()
        stats = visualization.get('stats', {})
        return SystemOutput(
            time_step=int(self.time_step),
            timestamp=time.time(),
            visualization=visualization,
            emergence=emergence,
            stats=stats,
            recent_inputs=self._get_recent_inputs(),
            metadata=metadata
        )
    
    def get_visualization_data(self) -> Dict:
        """获取可视化数据（发送给Web前端）- 包含hover所需的详细信息"""
        alive_particles = [p for p in self.particles if p.alive]
        
        if len(alive_particles) == 0:
            return {
                'time_step': int(self.time_step),
                'particles': {
                    'positions': [], 'energies': [], 'generations': [],
                    'ids': [], 'ages': [], 'genomes': [], 'phenotypes': [], 'count': 0
                },
                'chemical_field': {'atp': [[]], 'shape': [0, 0]},
                'stats': {'alive_particles': 0, 'total_energy': 0.0, 
                          'replication_events': 0, 'death_events': 0, 
                          'interaction_events': 0, 'max_generation': 0}
            }
        
        # 基础数据
        positions = np.array([p.position for p in alive_particles])
        energies = np.array([p.energy for p in alive_particles])
        generations = np.array([p.generation for p in alive_particles])
        
        # 详细信息（用于hover）
        ids = [p.id for p in alive_particles]
        ages = [p.age for p in alive_particles]
        genomes = [p.genome.tolist() if hasattr(p.genome, 'tolist') else list(p.genome) 
                   for p in alive_particles]
        phenotypes = [p.phenotype for p in alive_particles]
        
        # 化学场可视化（取ATP浓度）
        atp_field = self.chemical_field.concentrations[:, :, self.chemical_field.ATP_index]
        nutrient_field = self.chemical_field.concentrations[:, :, self.chemical_field.nutrient_index]
        inhibitor_field = self.chemical_field.concentrations[:, :, self.chemical_field.inhibitor_index]
        
        # GPU → CPU转换（如果使用CuPy）
        if GPU_AVAILABLE:
            positions = cp.asnumpy(positions)
            energies = cp.asnumpy(energies)
            generations = cp.asnumpy(generations)
            atp_field = cp.asnumpy(atp_field)
            nutrient_field = cp.asnumpy(nutrient_field)
            inhibitor_field = cp.asnumpy(inhibitor_field)
        
        return {
            'time_step': int(self.time_step),
            'particles': {
                'positions': positions.tolist(),
                'energies': energies.tolist(),
                'generations': generations.tolist(),
                # 详细信息
                'ids': ids,
                'ages': ages,
                'genomes': genomes,
                'phenotypes': phenotypes,
                'count': int(len(alive_particles))
            },
            'chemical_field': {
                'atp': atp_field.tolist(),
                'nutrient': nutrient_field.tolist(),
                'inhibitor': inhibitor_field.tolist(),
                'shape': list(atp_field.shape)
            },
            'stats': {
                'alive_particles': int(len(alive_particles)),
                'total_energy': float(sum(energies)),
                'replication_events': int(self.stats['replication_events']),
                'death_events': int(self.stats['death_events']),
                'interaction_events': int(self.stats['interaction_events']),
                'max_generation': int(max(generations)),
                # 基因组统计
                'genome_lengths': {
                    'max': int(max(len(p.genome) for p in alive_particles)),
                    'min': int(min(len(p.genome) for p in alive_particles)),
                    'avg': float(np.mean([len(p.genome) for p in alive_particles])),
                },
                # 第二层：场能量统计（用于守恒监控）
                'field_atp_total': float(np.sum(atp_field)),
                'system_total_energy': float(sum(energies)) + float(np.sum(atp_field)),
            }
        }
    
    def trigger_disturbance_event(self, event_type='random'):
        """触发环境扰动事件 - 打破系统僵化，催化新涌现"""
        print(f"\n💥 环境扰动事件触发！类型: {event_type}")
        
        if event_type == 'random' or event_type == 'energy_fluctuation':
            # 能量波动：随机区域能量剧增或骤降
            x_center = int(self.rng.integers(0, self.config.world_size))
            y_center = int(self.rng.integers(0, self.config.world_size))
            radius = 20
            
            for x in range(max(0, x_center - radius), min(self.config.world_size, x_center + radius)):
                for y in range(max(0, y_center - radius), min(self.config.world_size, y_center + radius)):
                    dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)
                    if dist < radius:
                        # ATP浓度剧烈变化
                        if self.rng.random() < 0.5:
                            # 能量爆发
                            self.chemical_field.concentrations[x, y, self.chemical_field.ATP_index] += 2.0
                        else:
                            # 能量枯竭
                            self.chemical_field.concentrations[x, y, self.chemical_field.ATP_index] *= 0.3
            
            print(f"   区域: ({x_center}, {y_center}), 半径: {radius}")
            self.chemical_field.invalidate_gradient_cache()
        
        elif event_type == 'mass_extinction':
            # 大灭绝：随机杀死一定比例的粒子
            alive_particles = [p for p in self.particles if p.alive]
            if not alive_particles:
                return
            population_ratio = len(alive_particles) / max(1, self.config.n_particles)
            kill_ratio = 0.15 + 0.15 * np.clip(population_ratio - 1.0, -0.5, 1.0)
            kill_ratio = float(np.clip(kill_ratio, 0.05, 0.3))
            n_to_kill = int(len(alive_particles) * kill_ratio)
            victims = self._sample_particles(alive_particles, n_to_kill)
            
            for particle in victims:
                particle.alive = False
                # 释放能量
                x, y = int(particle.position[0]), int(particle.position[1])
                x = max(0, min(x, self.config.world_size - 1))
                y = max(0, min(y, self.config.world_size - 1))
                self.chemical_field.produce(particle.position, self.chemical_field.waste_index, particle.energy)
            
            print(f"   灭绝比例: {kill_ratio*100}%, 死亡: {n_to_kill}")
        
        elif event_type == 'mutation_burst':
            # 突变爆发：大幅提高变异率
            alive_particles = [p for p in self.particles if p.alive]
            burst_count = max(1, int(len(alive_particles) * 0.05))
            burst_particles = self._sample_particles(alive_particles, burst_count)
            for particle in burst_particles:
                # 随机突变多个基因位点
                n_mutations = int(self.rng.integers(2, 6))
                mutation_points = self.rng.choice(
                    len(particle.genome),
                    size=min(n_mutations, len(particle.genome)),
                    replace=False
                )
                for point in mutation_points:
                    particle.genome[int(point)] = int(self.rng.integers(0, 4))
                # 重新表达
                particle.phenotype = particle._express_genes()
            
            print(f"   突变个体数: {len(burst_particles)}")
        
        self.stats['disturbance_events'] = self.stats.get('disturbance_events', 0) + 1
    
    def get_emergence_metrics(self) -> Dict:
        """计算涌现性指标"""
        alive_particles = [p for p in self.particles if p.alive]
        
        if len(alive_particles) == 0:
            return {'emergence_detected': False, 'emergence_score': 0.0}
        
        # 真正的涌现需要时间发展：至少1000步后才可能出现
        if self.time_step < 1000:
            return {
                'emergence_detected': False,
                'emergence_score': 0.0,
                'genetic_diversity': 0.0,
                'energy_variance': 0.0,
                'spatial_variance': 0.0,
                'generation_diversity': 0,
                'max_generation': 0,
                'world_diversity': 0.0,
            }
        
        # 1. 世代演化（核心指标）
        generations = np.array([p.generation for p in alive_particles])
        max_generation = int(np.max(generations))
        if GPU_AVAILABLE:
            generation_list = cp.asnumpy(generations).tolist()
        else:
            generation_list = generations.tolist()
        generation_diversity = len(set(generation_list))
        
        # 2. 基因变异度（相对于初始种群）
        genomes = [p.genome for p in alive_particles]
        if GPU_AVAILABLE:
            genome_tuples = [tuple(cp.asnumpy(g).tolist()) for g in genomes]
        else:
            genome_tuples = [tuple(g.tolist()) for g in genomes]
        unique_genomes = len(set(genome_tuples))
        genetic_diversity = unique_genomes / len(alive_particles)
        
        # 3. 能量分布（是否有能量循环和分层）
        energies = np.array([p.energy for p in alive_particles])
        energy_variance = np.var(energies)
        
        # 4. 空间聚集度（是否形成生态位）
        positions = np.array([p.position for p in alive_particles])
        spatial_variance = np.var(positions, axis=0).mean()
        nutrient_field = self.chemical_field.concentrations[:, :, self.chemical_field.nutrient_index]
        inhibitor_field = self.chemical_field.concentrations[:, :, self.chemical_field.inhibitor_index]
        if GPU_AVAILABLE:
            nutrient_field = cp.asnumpy(nutrient_field)
            inhibitor_field = cp.asnumpy(inhibitor_field)
        nutrient_variance = float(np.var(nutrient_field))
        inhibitor_pressure = float(np.mean(inhibitor_field))
        
        # 涌现判断（更严格）
        # 必须满足：世代>5，基因多样性>0.3，能量有分化
        generation_score = min(max_generation / 20.0, 1.0)  # 至少20代
        diversity_score = genetic_diversity  # 至少30%不同
        energy_score = min(energy_variance / 5.0, 1.0)  # 能量要有分化
        spatial_score = min(spatial_variance / 200.0, 1.0)  # 空间要有分布
        environment_score = min(nutrient_variance / 2.0, 1.0)
        
        emergence_score = (
            generation_score * 0.4 +  # 世代是最重要的
            diversity_score * 0.3 +
            energy_score * 0.15 +
            spatial_score * 0.1 +
            environment_score * 0.05
        )
        
        return {
            'emergence_detected': bool(emergence_score > 0.5),
            'emergence_score': float(emergence_score),
            'genetic_diversity': float(genetic_diversity),
            'energy_variance': float(energy_variance),
            'spatial_variance': float(spatial_variance),
            'nutrient_variance': nutrient_variance,
            'inhibitor_pressure': inhibitor_pressure,
            'max_generation': int(max_generation),
            'generation_diversity': int(generation_diversity),
            'population_size': int(len(alive_particles)),
        }


def run_ecology_2d_simulation():
    """运行2D生态模拟"""
    config = Ecology2DConfig(
        world_size=200,
        n_particles=3000,
        genome_length=32,
        mutation_rate=0.01
    )
    
    system = Ecology2DSystem(config)
    
    print("\n🌍 开始2D生态模拟...\n")
    
    for step in range(1000):
        system.step()
        
        if step % 100 == 0:
            emergence = system.get_emergence_metrics()
            print(f"步骤 {step}:")
            print(f"  存活粒子: {emergence['population_size']}")
            print(f"  最大世代: {emergence['max_generation']}")
            print(f"  涌现得分: {emergence['emergence_score']:.3f}")
            print(f"  复制事件: {system.stats['replication_events']}")
            print()


if __name__ == "__main__":
    run_ecology_2d_simulation()





