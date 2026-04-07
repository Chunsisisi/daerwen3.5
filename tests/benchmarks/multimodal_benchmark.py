"""
Multi-Modal Sensory Integration Benchmark v2
=============================================
核心问题：多模态感官信号（视觉指示 + 能量奖励）是否比单模态更能
         引导种群找到并利用能量区（"插座"）？

宏观模型：
  "插座" = 目标区域，每步注入固定 ATP 预算（能量上限防爆炸）
  "看见插座" = 目标区域有 Nutrient 指示信号，粒子可从远处感知
  "充电驱动" = 粒子能量低时趋化力增强，自然产生"饥饿→寻找能量"的行为

能量平衡设计：
  - 每步向目标区域注入固定预算（OUTLET_BUDGET = 0.5 ATP/步）
  - 目标区域 ATP 上限 = OUTLET_ATP_CAP（防止堆积）
  - 粒子在区域内正常吸收 ATP，全局能量守恒
  - 净效果：目标区域能量密度更高，其他区域相对贫瘠

三组对照：
  Baseline：均匀环境，ATP 靠太阳能自然补充
  Single  ：插座（目标区域固定 ATP 预算），无指示信号
  Multi   ：插座 + Nutrient 视觉/嗅觉指示信号

假设：Multi > Single > Baseline（空间专化度、种群稳定性）
     多模态指示信号让粒子更快找到能量区，适应更稳定
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _root)

import numpy as np

try:
    from engine.core import Ecology2DSystem, Ecology2DConfig
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ── 全局配置 ───────────────────────────────────────────────────────────────────
WORLD_SIZE   = 80
N_PARTICLES  = 300
N_CHEMICALS  = 12
WARMUP_STEPS = 400
TRAIN_STEPS  = 500
TEST_STEPS   = 200

# 目标区域（"插座"）：左上角 1/4
TARGET_X = WORLD_SIZE // 4   # 20
TARGET_Y = WORLD_SIZE // 4   # 20
TARGET_AREA_FRAC = (TARGET_X * TARGET_Y) / (WORLD_SIZE ** 2)  # 6.25%

# 插座能量参数
OUTLET_BUDGET  = 0.5   # 每步注入 ATP 总量（固定预算，粒子多时每个分到少）
OUTLET_ATP_CAP = 1.8   # 目标区域 ATP 上限，防止无限积累


# ── 系统工厂 ───────────────────────────────────────────────────────────────────

def make_system() -> Ecology2DSystem:
    cfg = Ecology2DConfig(
        world_size=WORLD_SIZE,
        n_particles=N_PARTICLES,
        n_chemical_species=N_CHEMICALS,
    )
    return Ecology2DSystem(cfg)


# ── 测量函数 ───────────────────────────────────────────────────────────────────

def count_alive(system: Ecology2DSystem) -> int:
    return sum(1 for p in system.particles if p.energy > 0)


def outlet_density(system: Ecology2DSystem) -> float:
    """目标区域粒子密度比（1.0 = 随机，>1 = 聚集）"""
    alive = [p for p in system.particles if p.energy > 0]
    if not alive:
        return 0.0
    in_zone = sum(
        1 for p in alive
        if p.position[0] < TARGET_X and p.position[1] < TARGET_Y
    )
    return (in_zone / len(alive)) / TARGET_AREA_FRAC


def genome_diversity(system: Ecology2DSystem) -> float:
    """表型 Shannon 熵，归一化到 [0,1]"""
    alive = [p for p in system.particles if p.energy > 0]
    if len(alive) < 2:
        return 0.0
    vals = [p.phenotype.get('replication_threshold', 1.0) for p in alive]
    counts, _ = np.histogram(vals, bins=10, range=(0.5, 4.5))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)) / np.log(10))


def mean_energy(system: Ecology2DSystem) -> float:
    alive = [p for p in system.particles if p.energy > 0]
    return float(np.mean([p.energy for p in alive])) if alive else 0.0


# ── 环境设置 ───────────────────────────────────────────────────────────────────

def setup_outlet_signal(system: Ecology2DSystem):
    """
    多模态指示信号（视觉 + 嗅觉）：
    在目标区域叠加 Nutrient 信号，让粒子从远处就能感知"插座"位置。
    不替换原有 Nutrient 场，只叠加增量。
    """
    w  = WORLD_SIZE
    cf = system.chemical_field
    ni = cf.nutrient_index

    xx, yy = np.meshgrid(np.arange(w), np.arange(w), indexing='ij')

    # 视觉信号：目标区域内叠加均匀增量
    in_target = (xx < TARGET_X) & (yy < TARGET_Y)
    visual = np.where(in_target, 0.6, 0.0).astype(np.float32)

    # 嗅觉信号：距离目标区域越近，Nutrient 越高（引导方向）
    dist = np.sqrt(
        np.maximum(xx.astype(float) - TARGET_X, 0)**2 +
        np.maximum(yy.astype(float) - TARGET_Y, 0)**2
    )
    olfactory = (0.4 * np.exp(-dist / (w * 0.2))).astype(np.float32)

    cf.concentrations[:, :, ni] = np.clip(
        cf.concentrations[:, :, ni] + visual + olfactory, 0.0, 2.5
    )
    cf.invalidate_gradient_cache()


def inject_outlet_atp(system: Ecology2DSystem):
    """
    每步向目标区域注入固定 ATP 预算，但不超过上限。
    固定预算 = 插座"功率"固定；上限 = 防止能量无限堆积。
    """
    cf  = system.chemical_field
    zone = cf.concentrations[:TARGET_X, :TARGET_Y, cf.ATP_index]

    # 只给未满的格子补充能量
    deficit = np.clip(OUTLET_ATP_CAP - zone, 0.0, None)
    total_deficit = float(deficit.sum())
    if total_deficit > 0:
        fill = np.minimum(deficit, deficit * (OUTLET_BUDGET / total_deficit))
        cf.concentrations[:TARGET_X, :TARGET_Y, cf.ATP_index] += fill.astype(np.float32)
        cf.invalidate_gradient_cache()


# ── 核心实验 ───────────────────────────────────────────────────────────────────

def run_condition(name: str, mode: str, seed: int = 42, verbose: bool = True) -> dict:
    """
    mode: 'baseline' | 'single' | 'multi'
    baseline : 均匀环境，无插座
    single   : 插座（固定 ATP 预算），无指示信号
    multi    : 插座 + Nutrient 指示信号
    """
    rng_backup = np.random.default_rng(seed)
    system = make_system()

    if verbose:
        print(f"  [{name}] 预热中...")

    # 预热
    for _ in range(WARMUP_STEPS):
        system.step()

    initial_pop = count_alive(system)
    if verbose:
        print(f"  [{name}] 预热完成，初始种群: {initial_pop}")
    if initial_pop < 10:
        return {'mode': mode, 'error': 'extinct_after_warmup',
                'population_stability': 0.0, 'specialization': 0.0,
                'genome_diversity': 0.0, 'mean_energy': 0.0}

    # 施加初始环境
    if mode in ('single', 'multi'):
        inject_outlet_atp(system)       # 先充一次初始能量
    if mode == 'multi':
        setup_outlet_signal(system)     # 叠加视觉/嗅觉指示信号

    # 训练阶段
    train_stability = []
    for _ in range(TRAIN_STEPS):
        if mode in ('single', 'multi'):
            inject_outlet_atp(system)
        system.step()
        train_stability.append(min(1.0, count_alive(system) / max(initial_pop, 1)))

    # 测试阶段（环境保持，测量适应结果）
    spec_scores, div_scores, pop_ratios, energy_scores = [], [], [], []
    for _ in range(TEST_STEPS):
        if mode in ('single', 'multi'):
            inject_outlet_atp(system)
        system.step()
        spec_scores.append(outlet_density(system))
        div_scores.append(genome_diversity(system))
        pop_ratios.append(min(1.0, count_alive(system) / max(initial_pop, 1)))
        energy_scores.append(mean_energy(system))

    result = {
        'mode': mode,
        'population_stability': float(np.mean(pop_ratios)),
        'specialization':       float(np.mean(spec_scores)),
        'genome_diversity':     float(np.mean(div_scores)),
        'mean_energy':          float(np.mean(energy_scores)),
        'final_population':     count_alive(system),
        'initial_population':   initial_pop,
    }

    if verbose:
        print(f"  [{name}] 完成 | "
              f"稳定性: {result['population_stability']:.3f} | "
              f"专化度: {result['specialization']:.3f} | "
              f"多样性: {result['genome_diversity']:.3f} | "
              f"均能量: {result['mean_energy']:.3f} | "
              f"末种群: {result['final_population']}")

    return result


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("MULTI-MODAL SENSORY INTEGRATION BENCHMARK v2")
    print("插座实验：固定能量预算 + 多模态指示信号")
    print("=" * 62)
    print(f"世界: {WORLD_SIZE}x{WORLD_SIZE}  粒子: {N_PARTICLES}  化学: {N_CHEMICALS}")
    print(f"插座区域: {TARGET_X}x{TARGET_Y} ({TARGET_AREA_FRAC*100:.1f}%)  "
          f"预算: {OUTLET_BUDGET}/步  上限: {OUTLET_ATP_CAP}")
    print()

    t0 = time.time()
    N_RUNS = 3
    results = {'baseline': [], 'single': [], 'multi': []}
    conditions = [
        ('baseline', 'Baseline'),
        ('single',   '单模态'),
        ('multi',    '多模态'),
    ]

    for run_i in range(N_RUNS):
        print(f"── 第 {run_i+1}/{N_RUNS} 轮 {'─'*40}")
        seed = 42 + run_i * 100
        for mode, name in conditions:
            r = run_condition(name, mode, seed=seed, verbose=True)
            results[mode].append(r)
        print()

    # 汇总
    def avg(mode, key):
        vals = [r[key] for r in results[mode] if key in r and 'error' not in r]
        return float(np.mean(vals)) if vals else 0.0

    summary = {m: {k: avg(m, k) for k in
                   ['population_stability','specialization','genome_diversity','mean_energy']}
               for m in ['baseline','single','multi']}

    synergy = {k: summary['multi'][k] - summary['single'][k]
               for k in summary['multi']}

    print("=" * 62)
    print("实验结果")
    print("=" * 62)
    header = f"{'指标':<16} {'Baseline':>10} {'单模态':>10} {'多模态':>10} {'协同(M-S)':>10}"
    print(header)
    print("-" * 62)
    labels = {
        'population_stability': '种群稳定性',
        'specialization':       '空间专化度',
        'genome_diversity':     '基因多样性',
        'mean_energy':          '平均能量',
    }
    for k, label in labels.items():
        print(f"{label:<16} "
              f"{summary['baseline'][k]:>10.3f} "
              f"{summary['single'][k]:>10.3f} "
              f"{summary['multi'][k]:>10.3f} "
              f"{synergy[k]:>+10.3f}")

    # 假设验证：Multi 在空间专化度或种群稳定性上显著优于 Single
    hyp = synergy['specialization'] > 0.3 or synergy['population_stability'] > 0.05
    print()
    print("假设：多模态(M) 在空间专化度或种群稳定性上显著优于单模态(S)")
    print(f"结果：{'[支持]' if hyp else '[不支持]'}")
    print(f"  空间专化协同效应: {synergy['specialization']:+.3f} (阈值 >0.3)")
    print(f"  稳定性协同效应:   {synergy['population_stability']:+.3f} (阈值 >0.05)")

    elapsed = time.time() - t0
    print(f"\n耗时: {elapsed/60:.1f} 分钟")

    # 保存
    out_dir = Path(_root) / 'benchmark_results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"multimodal_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        'summary': summary, 'synergy': synergy,
        'hypothesis_supported': hyp,
        'config': {
            'outlet_budget': OUTLET_BUDGET,
            'outlet_atp_cap': OUTLET_ATP_CAP,
            'target_area_frac': TARGET_AREA_FRAC,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {out_path}")
    return output


if __name__ == '__main__':
    main()
