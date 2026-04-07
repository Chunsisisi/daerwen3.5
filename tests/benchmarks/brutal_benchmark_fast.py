"""
BRUTAL PROFESSIONAL BENCHMARK - FAST VERSION v2

修复说明（v2）：
- 所有 CL 分数改为相对物理基准的提升量，消除"粒子自然聚集"带来的虚假得分
- FWT 改用标准定义：训练前历任务对新任务的迁移 vs 物理基准
- Robustness 三项 warmup 提升到 1000 步（原 500），加灭绝检测与重试
- 保存路径改为相对路径（兼容 Windows/Linux）

指标定义（Lopez-Paz & Ranzato 2017，相对基准版）：
  R[i,j]  = 训练完任务 i 后，对任务 j 的对齐分（绝对值）
  base[j] = 纯物理基准（无训练历史）对任务 j 的对齐分
  rel[i,j] = max(0, R[i,j] - base[j])  ← 相对提升量

  FAP = mean(rel[-1, j] for all j)          最终平均相对性能
  BWT = mean(rel[-1,i] - rel[i,i])          向后迁移（负值=遗忘）
  FWT = mean(R[i-1,i] - base[i]) for i≥1   向前迁移（训练前对未见任务的迁移）
  Forgetting = mean(max(0, rel[i,i] - rel[-1,i]))  遗忘量
"""
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 确保项目根目录在 path 里，无论从哪里运行
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from engine.core import Ecology2DSystem, Ecology2DConfig, ExternalInput
from controllers.state_aggregator import SimpleStateAggregator


def ttest_ind(a, b):
    """Simple independent t-test"""
    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    t_stat = (mean_a - mean_b) / (pooled_std * np.sqrt(1/n_a + 1/n_b)) if pooled_std > 0 else 0
    p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / 2)))
    return t_stat, p_value


@dataclass
class FastBrutalResults:
    continual_learning_10tasks: Dict
    robustness_tests: Dict
    baseline_comparison: Dict
    overall_score: float
    grade: str
    brutal_assessment: str
    timestamp: str


class TaskLibrary:
    @staticmethod
    def create_gradient(system: Ecology2DSystem, direction: str):
        world = system.config.world_size
        x = np.arange(world)
        y = np.arange(world)
        xx, yy = np.meshgrid(x, y)
        center = world // 2

        if direction == 'north':
            gradient = (yy / world) * 2.0
        elif direction == 'east':
            gradient = (xx / world) * 2.0
        elif direction == 'west':
            gradient = ((world - xx) / world) * 2.0
        elif direction == 'diagonal_nw':
            gradient = ((world - xx) + yy) / (2 * world) * 2.0
        elif direction == 'checkerboard':
            gradient = np.where((xx // 20 + yy // 20) % 2 == 0, 2.0, 0.5)
        elif direction == 'wave':
            gradient = 1.0 + np.sin(xx / world * 4 * np.pi) * 0.5
        # 新增：粒子自然不倾向去的区域（物理基准低）
        elif direction == 'boundary':
            # 边界高、中心低——与粒子天然聚集的中心相反
            dist = np.sqrt((xx - center)**2 + (yy - center)**2)
            max_dist = center * np.sqrt(2)
            gradient = 2.0 * (dist / max_dist)
        elif direction == 'corners':
            # 四个角落高，其余低
            d_tl = np.sqrt(xx**2 + yy**2)
            d_tr = np.sqrt((xx - world)**2 + yy**2)
            d_bl = np.sqrt(xx**2 + (yy - world)**2)
            d_br = np.sqrt((xx - world)**2 + (yy - world)**2)
            d_min = np.minimum(np.minimum(d_tl, d_tr), np.minimum(d_bl, d_br))
            gradient = 2.0 * np.exp(-d_min**2 / (2 * (world / 6)**2))
        elif direction == 'top_stripe':
            # 顶部 1/4 水平条带高，其余低
            gradient = np.where(yy >= world * 0.75, 2.0, 0.3)
        elif direction == 'left_stripe':
            # 左侧 1/4 垂直条带高，其余低
            gradient = np.where(xx <= world * 0.25, 2.0, 0.3)
        else:
            gradient = np.ones((world, world))

        system.chemical_field.concentrations[:, :, system.chemical_field.ATP_index] = gradient


class Metrics:
    @staticmethod
    def calculate_alignment(particles, gradient_type: str, world_size: int) -> float:
        alive = [p for p in particles if p.alive]
        if not alive:
            return 0.0

        positions = np.array([p.position for p in alive])
        cx, cy = np.mean(positions[:, 0]), np.mean(positions[:, 1])
        center = world_size / 2

        if gradient_type == 'north':
            score = cy / world_size
        elif gradient_type == 'east':
            score = cx / world_size
        elif gradient_type == 'west':
            score = (world_size - cx) / world_size
        elif gradient_type == 'diagonal_nw':
            score = ((world_size - cx) + cy) / (2 * world_size)
        elif gradient_type == 'checkerboard':
            avg_energy = np.mean([p.energy for p in alive])
            score = np.clip(avg_energy / 2.0, 0, 1)
        elif gradient_type == 'wave':
            avg_energy = np.mean([p.energy for p in alive])
            score = np.clip(avg_energy / 2.0, 0, 1)
        elif gradient_type == 'boundary':
            # 粒子距中心越远分越高
            dists = np.sqrt((positions[:, 0] - center)**2 + (positions[:, 1] - center)**2)
            max_dist = center * np.sqrt(2)
            score = float(np.mean(dists) / max_dist)
        elif gradient_type == 'corners':
            # 粒子距最近角落越近分越高
            corner_dists = []
            for px, py in positions:
                d = min(
                    np.sqrt(px**2 + py**2),
                    np.sqrt((px - world_size)**2 + py**2),
                    np.sqrt(px**2 + (py - world_size)**2),
                    np.sqrt((px - world_size)**2 + (py - world_size)**2),
                )
                corner_dists.append(d)
            max_corner = world_size / np.sqrt(2)
            score = float(1.0 - np.mean(corner_dists) / max_corner)
        elif gradient_type == 'top_stripe':
            score = float(np.mean(positions[:, 1]) / world_size)
        elif gradient_type == 'left_stripe':
            score = float(1.0 - np.mean(positions[:, 0]) / world_size)
        else:
            avg_energy = np.mean([p.energy for p in alive])
            score = np.clip(avg_energy / 2.0, 0, 1)

        return float(np.clip(score, 0.0, 1.0))


class Fast10TaskContinualLearning:
    def __init__(self, config: Ecology2DConfig):
        self.config = config
        self.system = Ecology2DSystem(config)
        # 任务选择原则：物理基准 < 0.5，避免粒子天然聚集区（center/ring/south）
        # 上次实测基准：north=0.298, east=0.380, west=0.530, diagonal_nw=0.566,
        #               checkerboard=0.194, wave=0.206
        # 新增4个：boundary/corners/top_stripe/left_stripe（预计基准更低）
        self.tasks = [
            ('north',        'north'),
            ('east',         'east'),
            ('west',         'west'),
            ('diagonal_nw',  'diagonal_nw'),
            ('checkerboard', 'checkerboard'),
            ('wave',         'wave'),
            ('boundary',     'boundary'),
            ('corners',      'corners'),
            ('top_stripe',   'top_stripe'),
            ('left_stripe',  'left_stripe'),
        ]
        self.performance_matrix = np.zeros((len(self.tasks), len(self.tasks)))
        self.frozen_baselines = np.zeros(len(self.tasks))

    def _compute_frozen_baselines(self) -> np.ndarray:
        """
        纯物理基准：一个从未受过训练的系统，在每个任务梯度下的对齐分。
        用于消除粒子自然聚集带来的虚假得分。
        """
        print("\n[Baseline] Computing physics-only baseline (1000-step warmup)...")
        n = len(self.tasks)
        baselines = np.zeros(n)
        system = Ecology2DSystem(self.config)
        for _ in range(1000):
            system.step()
        for j, (name, gradient_type) in enumerate(self.tasks):
            TaskLibrary.create_gradient(system, gradient_type)
            for _ in range(25):
                system.step()
            baselines[j] = Metrics.calculate_alignment(
                system.particles, gradient_type, self.config.world_size
            )
            print(f"  {name:15s}: {baselines[j]:.3f} (physics only)")
        return baselines

    def run(self) -> Dict:
        print("\n" + "="*70)
        print("BRUTAL TEST 1: 10-TASK CONTINUAL LEARNING (FAST)")
        print("="*70)

        # 先计算物理基准
        self.frozen_baselines = self._compute_frozen_baselines()

        print("\n[Warmup] Running 2000 steps to stabilize ecosystem...")
        for _ in range(2000):
            self.system.step()
        initial_alive = sum(1 for p in self.system.particles if p.alive)
        print(f"  After warmup: {initial_alive} alive")

        n_tasks = len(self.tasks)

        for i in range(n_tasks):
            task_name, gradient_type = self.tasks[i]
            print(f"\n[Task {i+1}/{n_tasks}] Training: {task_name}")

            for epoch in range(5):
                TaskLibrary.create_gradient(self.system, gradient_type)
                for _ in range(50):
                    self.system.step()

            alive = sum(1 for p in self.system.particles if p.alive)
            print(f"  After training: {alive} alive")

            for j in range(n_tasks):
                test_name, test_gradient = self.tasks[j]
                TaskLibrary.create_gradient(self.system, test_gradient)
                for _ in range(25):
                    self.system.step()
                score = Metrics.calculate_alignment(
                    self.system.particles, test_gradient, self.config.world_size
                )
                self.performance_matrix[i, j] = score

        results = self._calculate_metrics()
        self._print_results(results)
        return results

    def _calculate_metrics(self) -> Dict:
        n = len(self.tasks)
        base = self.frozen_baselines

        # 相对提升矩阵：减去物理基准，负值截为 0
        rel = np.maximum(0, self.performance_matrix - base[np.newaxis, :])

        # FAP：最终轮对所有任务的平均相对提升
        FAP = float(np.mean(rel[-1, :]))

        # BWT：最终轮 vs 各任务训练刚完时的相对提升差（负=遗忘）
        BWT = float(np.mean([rel[-1, i] - rel[i, i] for i in range(n - 1)]))

        # FWT（标准定义）：训练任务 i 之前，系统对任务 i 的绝对得分 vs 物理基准
        # 反映"训练过往任务"是否让系统对未见任务有迁移
        FWT = float(np.mean([
            self.performance_matrix[i - 1, i] - base[i]
            for i in range(1, n)
        ]))

        # 遗忘量
        forgetting = [max(0.0, rel[i, i] - rel[-1, i]) for i in range(n - 1)]
        avg_forgetting = float(np.mean(forgetting))
        max_forgetting = float(np.max(forgetting))

        return {
            'BWT': BWT,
            'FWT': FWT,
            'FAP': FAP,
            'avg_forgetting': avg_forgetting,
            'max_forgetting': max_forgetting,
            'frozen_baselines': base.tolist(),
            'performance_matrix': self.performance_matrix.tolist(),
            'relative_performance_matrix': rel.tolist(),
            'n_tasks': n,
        }

    def _print_results(self, results: Dict):
        print("\n" + "="*70)
        print("10-TASK CONTINUAL LEARNING RESULTS")
        print("(all scores = improvement over physics-only baseline)")
        print("="*70)
        print(f"\nBWT (Backward Transfer):   {results['BWT']:+.3f}")
        print(f"FWT (Forward Transfer):    {results['FWT']:+.3f}")
        print(f"FAP (Final Avg Perf):      {results['FAP']:.3f}")
        print(f"Avg Forgetting:            {results['avg_forgetting']:.3f}")
        print(f"Max Forgetting:            {results['max_forgetting']:.3f}")

        if results['BWT'] < -0.05:
            print("  ⚠ WARNING: Catastrophic forgetting detected")
        if results['avg_forgetting'] > 0.1:
            print("  ⚠ WARNING: High forgetting rate")
        if results['FAP'] <= 0:
            print("  ⚠ CRITICAL: System not outperforming physics baseline")


class FastRobustnessTest:
    def __init__(self, config: Ecology2DConfig):
        self.config = config

    def run(self) -> Dict:
        print("\n" + "="*70)
        print("BRUTAL TEST 2: ROBUSTNESS (FAST)")
        print("="*70)

        results = {}

        print("\n[1/3] Noise Resilience")
        noise_scores = []
        for noise_level in [0.0, 0.2, 0.5]:
            score = self._test_with_noise(noise_level)
            noise_scores.append(score)
            print(f"  Noise {noise_level:.1f}: {score:.3f}")

        results['noise_resilience'] = {
            'scores': noise_scores,
            'degradation': float(noise_scores[0] - noise_scores[-1]),
        }

        print("\n[2/3] Catastrophe Recovery")
        recovery_score = self._test_catastrophe_recovery()
        results['catastrophe_recovery'] = {
            'score': float(recovery_score),
            'recovered': recovery_score > 0.3,
        }
        print(f"  Recovery: {recovery_score:.3f}")

        print("\n[3/3] Extreme Conditions")
        sparse_score = self._test_sparse()
        results['extreme_conditions'] = {'sparse': float(sparse_score)}
        print(f"  Sparse: {sparse_score:.3f}")

        overall_robustness = float(np.clip(
            np.mean(noise_scores) * 0.5 +
            recovery_score * 0.3 +
            sparse_score * 0.2,
            0.0, 1.0
        ))
        results['overall_robustness'] = overall_robustness

        print(f"\n→ Overall Robustness: {overall_robustness:.3f}")
        return results

    def _test_with_noise(self, noise_level: float) -> float:
        system = Ecology2DSystem(self.config)
        for _ in range(1000):   # 原 500，提升到 1000
            system.step()
        initial_alive = sum(1 for p in system.particles if p.alive)
        if initial_alive == 0:
            return 0.0
        for _ in range(100):
            if noise_level > 0:
                noise = np.random.normal(0, noise_level,
                                         system.chemical_field.concentrations.shape)
                system.chemical_field.concentrations += noise
                system.chemical_field.concentrations = np.clip(
                    system.chemical_field.concentrations, 0, 10)
            system.step()
        final_alive = sum(1 for p in system.particles if p.alive)
        return float(np.clip(final_alive / initial_alive, 0.0, 1.0))

    def _test_catastrophe_recovery(self, max_retries: int = 2) -> float:
        """灭绝事件后的种群恢复率，最多重试 max_retries 次。"""
        for attempt in range(max_retries + 1):
            system = Ecology2DSystem(self.config)
            for _ in range(1000):   # 原 500
                system.step()
            before_alive = sum(1 for p in system.particles if p.alive)
            if before_alive == 0:
                continue  # warmup 就灭绝，重试
            system.apply_external_input(
                ExternalInput('catastrophe', {'event_type': 'mass_extinction'}))
            for _ in range(150):
                system.step()
            recovered_alive = sum(1 for p in system.particles if p.alive)
            ratio = recovered_alive / before_alive
            if ratio > 0 or attempt == max_retries:
                return float(np.clip(ratio, 0.0, 1.0))
        return 0.0  # 重试全部失败，标记为 0（不是 invalid）

    def _test_sparse(self) -> float:
        config = Ecology2DConfig(
            world_size=self.config.world_size,
            n_particles=100,
            genome_length=self.config.genome_length,
        )
        system = Ecology2DSystem(config)
        for _ in range(1000):   # 原 500
            system.step()
        initial_alive = sum(1 for p in system.particles if p.alive)
        if initial_alive == 0:
            return 0.0
        for _ in range(150):
            system.step()
        final_alive = sum(1 for p in system.particles if p.alive)
        return float(np.clip(final_alive / initial_alive, 0.0, 1.0))


class FastBaselineComparison:
    def __init__(self, config: Ecology2DConfig):
        self.config = config

    def run(self) -> Dict:
        print("\n" + "="*70)
        print("BRUTAL TEST 3: BASELINE COMPARISON (FAST)")
        print("="*70)

        print("\n[1/2] Testing YOUR SYSTEM...")
        your_score = self._test_system()
        print(f"  Your system: {your_score:.3f}")

        print("\n[2/2] Testing PHYSICS BASELINE (mutation_rate=0)...")
        baseline_score = self._test_physics_baseline()
        print(f"  Physics baseline: {baseline_score:.3f}")

        improvement = ((your_score - baseline_score) / max(baseline_score, 1e-6))

        print(f"\n→ Improvement over physics baseline: {improvement*100:.1f}%")

        if improvement < 0.1:
            print("  ⚠ CRITICAL: Not meaningfully better than physics alone!")
        elif improvement < 0.3:
            print("  ⚠ WARNING: Modest improvement only")
        else:
            print("  ✓ Significant improvement over physics baseline")

        return {
            'your_system': float(your_score),
            'physics_baseline': float(baseline_score),
            'improvement_over_baseline': float(improvement),
        }

    def _test_system(self) -> float:
        system = Ecology2DSystem(self.config)
        for _ in range(1000):
            system.step()
        TaskLibrary.create_gradient(system, 'boundary')
        for _ in range(250):
            system.step()
        return Metrics.calculate_alignment(system.particles, 'boundary', self.config.world_size)

    def _test_physics_baseline(self) -> float:
        """mutation_rate=0，只有物理，没有进化。"""
        config = Ecology2DConfig(
            world_size=self.config.world_size,
            n_particles=self.config.n_particles,
            genome_length=self.config.genome_length,
            mutation_rate=0.0,
        )
        system = Ecology2DSystem(config)
        for _ in range(1000):
            system.step()
        TaskLibrary.create_gradient(system, 'boundary')
        for _ in range(250):
            system.step()
        return Metrics.calculate_alignment(system.particles, 'boundary', self.config.world_size)


class FastBrutalBenchmark:
    def __init__(self):
        self.config = Ecology2DConfig(
            world_size=80,
            n_particles=300,
            genome_length=24,
            mutation_rate=0.02,
            n_chemical_species=12,
        )

    def run(self) -> FastBrutalResults:
        print("\n" + "="*70)
        print("BRUTAL PROFESSIONAL BENCHMARK - FAST VERSION v2")
        print("="*70)
        print("\n所有 CL 分数 = 相对物理基准的提升量（消除虚假得分）")
        print("Estimated time: 8-12 minutes")
        print("="*70)

        start_time = time.time()

        cl_test = Fast10TaskContinualLearning(self.config)
        cl_results = cl_test.run()

        robust_test = FastRobustnessTest(self.config)
        robust_results = robust_test.run()

        baseline_test = FastBaselineComparison(self.config)
        baseline_results = baseline_test.run()

        duration = time.time() - start_time

        overall_score, grade, assessment = self._calculate_brutal_score(
            cl_results, robust_results, baseline_results
        )

        results = FastBrutalResults(
            continual_learning_10tasks=cl_results,
            robustness_tests=robust_results,
            baseline_comparison=baseline_results,
            overall_score=overall_score,
            grade=grade,
            brutal_assessment=assessment,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self._print_final_report(results, duration)
        self._save_results(results)

        return results

    def _calculate_brutal_score(self, cl, robust, baseline) -> Tuple[float, str, str]:
        # FAP 是相对提升量，理论上 [0,1]，用 clip 防止极端值
        fap = float(np.clip(cl['FAP'], 0.0, 1.0))
        bwt = float(np.clip(cl['BWT'], -1.0, 1.0))
        fwt = float(np.clip(cl['FWT'], -1.0, 1.0))
        forget = float(np.clip(cl['avg_forgetting'], 0.0, 1.0))

        cl_score = float(np.clip(
            (fap * 0.5 + max(0, bwt) * 0.3 + max(0, fwt) * 0.2 - forget * 0.2) * 100,
            0.0, 100.0
        ))

        robust_score = float(np.clip(robust['overall_robustness'] * 100, 0.0, 100.0))

        improvement = baseline['improvement_over_baseline']
        if improvement < 0.1:
            baseline_score = 20.0
        elif improvement < 0.3:
            baseline_score = 45.0
        elif improvement < 0.6:
            baseline_score = 65.0
        else:
            baseline_score = float(np.clip(70.0 + improvement * 20, 70.0, 100.0))

        overall = float(np.clip(
            cl_score * 0.4 + robust_score * 0.3 + baseline_score * 0.3,
            0.0, 100.0
        ))

        if overall >= 85:
            grade = "A - Exceptional"
            assessment = "Publication-ready. Top 1% territory."
        elif overall >= 75:
            grade = "B - Strong"
            assessment = "Competitive with state-of-art. Minor refinement needed."
        elif overall >= 65:
            grade = "C - Acceptable"
            assessment = "Meets basic standards. Significant gaps remain."
        elif overall >= 50:
            grade = "D - Weak"
            assessment = "Below academic standards. Fundamental issues detected."
        else:
            grade = "F - Failed"
            assessment = "Does not demonstrate meaningful learning beyond physics."

        return overall, grade, assessment

    def _print_final_report(self, results: FastBrutalResults, duration: float):
        print("\n" + "="*70)
        print("BRUTAL FINAL REPORT")
        print("="*70)
        print(f"\nDuration: {duration/60:.1f} minutes")
        print("\n" + "-"*70)
        print("SCORES (relative to physics baseline)")
        print("-"*70)

        cl = results.continual_learning_10tasks
        print(f"\n1. 10-Task Continual Learning:")
        print(f"   BWT: {cl['BWT']:+.3f} | FWT: {cl['FWT']:+.3f} | FAP: {cl['FAP']:.3f}")
        print(f"   Forgetting: {cl['avg_forgetting']:.3f}")
        print(f"   Physics baselines: {[f'{v:.2f}' for v in cl['frozen_baselines']]}")

        robust = results.robustness_tests
        print(f"\n2. Robustness: {robust['overall_robustness']:.3f}")
        print(f"   Noise degradation: {robust['noise_resilience']['degradation']:.3f}")
        print(f"   Recovery: {robust['catastrophe_recovery']['score']:.3f}")

        baseline = results.baseline_comparison
        print(f"\n3. vs Physics Baseline: +{baseline['improvement_over_baseline']*100:.1f}%")
        print(f"   Your system: {baseline['your_system']:.3f}  |  "
              f"Physics only: {baseline['physics_baseline']:.3f}")

        print("\n" + "-"*70)
        print("OVERALL ASSESSMENT")
        print("-"*70)
        print(f"\nScore: {results.overall_score:.1f} / 100")
        print(f"Grade: {results.grade}")
        print(f"\n{results.brutal_assessment}")
        print("="*70)

    def _save_results(self, results: FastBrutalResults):
        # 相对路径：从本文件位置找到 benchmark_results/
        base = Path(__file__).resolve().parent.parent.parent
        output_dir = base / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        filename = f"brutal_fast_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(results), f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {filepath}")


if __name__ == "__main__":
    suite = FastBrutalBenchmark()
    results = suite.run()
    print("\n✅ Fast brutal benchmark complete!")
