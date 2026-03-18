"""
BRUTAL PROFESSIONAL BENCHMARK - FAST VERSION

Same harsh testing, but faster:
- 10 tasks (still 3x more than quick)
- 5 epochs per task (vs 20) = 4x faster
- Shorter evaluation (25 steps vs 50) = 2x faster
- Total time: ~5-8 minutes (vs 20-40 minutes)

Still includes:
- Extended continual learning (10 tasks)
- Robustness testing
- Baseline comparison
- Harsh grading

This is the "get punched faster" version.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from engine.core import Ecology2DSystem, Ecology2DConfig, ExternalInput
from controllers.state_aggregator import SimpleStateAggregator


def ttest_ind(a, b):
    """Simple independent t-test"""
    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    n_a, n_b = len(a), len(b)
    
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    t_stat = (mean_a - mean_b) / (pooled_std * np.sqrt(1/n_a + 1/n_b)) if pooled_std > 0 else 0
    df = n_a + n_b - 2
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
        elif direction == 'south':
            gradient = ((world - yy) / world) * 2.0
        elif direction == 'east':
            gradient = (xx / world) * 2.0
        elif direction == 'west':
            gradient = ((world - xx) / world) * 2.0
        elif direction == 'center':
            dist = np.sqrt((xx - center)**2 + (yy - center)**2)
            gradient = 2.0 * np.exp(-dist**2 / (2 * (world/4)**2))
        elif direction == 'ring':
            dist = np.sqrt((xx - center)**2 + (yy - center)**2)
            optimal_radius = world / 3
            gradient = 2.0 * np.exp(-(dist - optimal_radius)**2 / (2 * (world/8)**2))
        elif direction == 'diagonal_ne':
            gradient = (xx + yy) / (2 * world) * 2.0
        elif direction == 'diagonal_nw':
            gradient = ((world - xx) + yy) / (2 * world) * 2.0
        elif direction == 'checkerboard':
            gradient = np.where((xx // 20 + yy // 20) % 2 == 0, 2.0, 0.5)
        elif direction == 'wave':
            gradient = 1.0 + np.sin(xx / world * 4 * np.pi) * 0.5
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
        elif gradient_type == 'south':
            score = (world_size - cy) / world_size
        elif gradient_type == 'east':
            score = cx / world_size
        elif gradient_type == 'west':
            score = (world_size - cx) / world_size
        elif gradient_type == 'center':
            dist = np.sqrt((cx - center)**2 + (cy - center)**2)
            score = 1.0 - (dist / (world_size * np.sqrt(2) / 2))
        elif gradient_type == 'ring':
            dists = [np.sqrt((p.position[0] - center)**2 + (p.position[1] - center)**2) for p in alive]
            optimal_radius = world_size / 3
            ring_alignment = np.mean([1.0 - abs(d - optimal_radius) / optimal_radius for d in dists])
            score = np.clip(ring_alignment, 0, 1)
        elif gradient_type in ['diagonal_ne', 'diagonal_nw']:
            score = (cx + cy) / (2 * world_size)
        else:
            avg_energy = np.mean([p.energy for p in alive])
            score = np.clip(avg_energy / 2.0, 0, 1)
        
        return float(np.clip(score, 0.0, 1.0))


class Fast10TaskContinualLearning:
    def __init__(self, config: Ecology2DConfig):
        self.config = config
        self.system = Ecology2DSystem(config)
        self.tasks = [
            ('north', 'north'),
            ('south', 'south'),
            ('east', 'east'),
            ('west', 'west'),
            ('center', 'center'),
            ('ring', 'ring'),
            ('diagonal_ne', 'diagonal_ne'),
            ('diagonal_nw', 'diagonal_nw'),
            ('checkerboard', 'checkerboard'),
            ('wave', 'wave'),
        ]
        self.performance_matrix = np.zeros((len(self.tasks), len(self.tasks)))
    
    def run(self) -> Dict:
        print("\n" + "="*70)
        print("BRUTAL TEST 1: 10-TASK CONTINUAL LEARNING (FAST)")
        print("="*70)
        
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
        
        BWT = 0.0
        for i in range(n-1):
            BWT += (self.performance_matrix[-1, i] - self.performance_matrix[i, i])
        BWT /= (n-1)
        
        FWT = 0.0
        for i in range(1, n):
            baseline = self.performance_matrix[i-1, i]
            FWT += (self.performance_matrix[i, i] - baseline)
        FWT /= (n-1)
        
        FAP = np.mean([self.performance_matrix[-1, i] for i in range(n)])
        
        forgetting = []
        for i in range(n-1):
            max_perf = self.performance_matrix[i, i]
            final_perf = self.performance_matrix[-1, i]
            forgetting.append(max(0, max_perf - final_perf))
        
        avg_forgetting = np.mean(forgetting)
        max_forgetting = np.max(forgetting)
        
        return {
            'BWT': float(BWT),
            'FWT': float(FWT),
            'FAP': float(FAP),
            'avg_forgetting': float(avg_forgetting),
            'max_forgetting': float(max_forgetting),
            'performance_matrix': self.performance_matrix.tolist(),
            'n_tasks': n,
        }
    
    def _print_results(self, results: Dict):
        print("\n" + "="*70)
        print("10-TASK CONTINUAL LEARNING RESULTS")
        print("="*70)
        print(f"\nBWT (Backward Transfer):   {results['BWT']:+.3f}")
        print(f"FWT (Forward Transfer):    {results['FWT']:+.3f}")
        print(f"FAP (Final Avg Perf):      {results['FAP']:.3f}")
        print(f"Avg Forgetting:            {results['avg_forgetting']:.3f}")
        print(f"Max Forgetting:            {results['max_forgetting']:.3f}")
        
        if results['BWT'] < -0.1:
            print("  ⚠ WARNING: Catastrophic forgetting detected")
        if results['avg_forgetting'] > 0.2:
            print("  ⚠ WARNING: High forgetting rate")


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
        
        overall_robustness = (
            np.mean(noise_scores) * 0.5 +
            recovery_score * 0.3 +
            sparse_score * 0.2
        )
        results['overall_robustness'] = float(overall_robustness)
        
        print(f"\n→ Overall Robustness: {overall_robustness:.3f}")
        return results
    
    def _test_with_noise(self, noise_level: float) -> float:
        system = Ecology2DSystem(self.config)
        for _ in range(500):
            system.step()
        initial_alive = sum(1 for p in system.particles if p.alive)
        for _ in range(100):
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, system.chemical_field.concentrations.shape)
                system.chemical_field.concentrations += noise
                system.chemical_field.concentrations = np.clip(system.chemical_field.concentrations, 0, 10)
            system.step()
        final_alive = sum(1 for p in system.particles if p.alive)
        survival_ratio = final_alive / max(initial_alive, 1)
        return min(survival_ratio, 1.0)
    
    def _test_catastrophe_recovery(self) -> float:
        system = Ecology2DSystem(self.config)
        for _ in range(500):
            system.step()
        before_alive = sum(1 for p in system.particles if p.alive)
        system.apply_external_input(ExternalInput('catastrophe', {'event_type': 'mass_extinction'}))
        for _ in range(150):
            system.step()
        recovered_alive = sum(1 for p in system.particles if p.alive)
        recovery_ratio = recovered_alive / max(before_alive, 1)
        return min(recovery_ratio, 1.0)
    
    def _test_sparse(self) -> float:
        config = Ecology2DConfig(
            world_size=self.config.world_size,
            n_particles=100,
            genome_length=self.config.genome_length,
        )
        system = Ecology2DSystem(config)
        for _ in range(500):
            system.step()
        initial_alive = sum(1 for p in system.particles if p.alive)
        for _ in range(150):
            system.step()
        final_alive = sum(1 for p in system.particles if p.alive)
        survival_ratio = final_alive / max(initial_alive, 1)
        return min(survival_ratio, 1.0)


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
        
        print("\n[2/2] Testing RANDOM BASELINE...")
        random_score = self._test_random()
        print(f"  Random: {random_score:.3f}")
        
        improvement = (your_score - random_score) / random_score if random_score > 0 else 0
        
        print(f"\n→ Improvement over random: {improvement*100:.1f}%")
        
        if improvement < 0.2:
            print("  ⚠ CRITICAL: Barely better than random!")
        elif improvement < 0.5:
            print("  ⚠ WARNING: Modest improvement only")
        else:
            print("  ✓ Significant improvement")
        
        return {
            'your_system': float(your_score),
            'random_baseline': float(random_score),
            'improvement_over_random': float(improvement),
        }
    
    def _test_system(self) -> float:
        system = Ecology2DSystem(self.config)
        for _ in range(500):
            system.step()
        TaskLibrary.create_gradient(system, 'center')
        for _ in range(250):
            system.step()
        return Metrics.calculate_alignment(system.particles, 'center', self.config.world_size)
    
    def _test_random(self) -> float:
        config = Ecology2DConfig(
            world_size=self.config.world_size,
            n_particles=self.config.n_particles,
            genome_length=self.config.genome_length,
            mutation_rate=0.0,
        )
        system = Ecology2DSystem(config)
        for _ in range(500):
            system.step()
        TaskLibrary.create_gradient(system, 'center')
        for _ in range(250):
            system.step()
        return Metrics.calculate_alignment(system.particles, 'center', self.config.world_size)


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
        print("BRUTAL PROFESSIONAL BENCHMARK - FAST VERSION")
        print("="*70)
        print("\nSame harsh testing, 4-8x faster execution")
        print("Estimated time: 5-8 minutes")
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
        cl_score = (
            cl['FAP'] * 0.5 +
            max(0, cl['BWT']) * 0.3 +
            max(0, cl['FWT']) * 0.2 -
            cl['avg_forgetting'] * 0.2
        ) * 100
        
        robust_score = robust['overall_robustness'] * 100
        
        improvement = baseline['improvement_over_random']
        if improvement < 0.2:
            baseline_score = 30
        elif improvement < 0.5:
            baseline_score = 50
        else:
            baseline_score = 70 + min(improvement * 30, 30)
        
        overall = cl_score * 0.4 + robust_score * 0.3 + baseline_score * 0.3
        
        if overall >= 85:
            grade = "A - Exceptional"
            assessment = "Publication-ready. Top 1% territory."
        elif overall >= 75:
            grade = "B - Strong"
            assessment = "Competitive with state-of-art. Minor refinement needed."
        elif overall >= 65:
            grade = "C - Acceptable"
            assessment = "Meets basic standards. Significant gaps remain. Major work needed."
        elif overall >= 50:
            grade = "D - Weak"
            assessment = "Below academic standards. Fundamental issues detected."
        else:
            grade = "F - Failed"
            assessment = "Does not demonstrate meaningful learning. Barely better than random."
        
        return float(overall), grade, assessment
    
    def _print_final_report(self, results: FastBrutalResults, duration: float):
        print("\n" + "="*70)
        print("BRUTAL FINAL REPORT")
        print("="*70)
        
        print(f"\nDuration: {duration/60:.1f} minutes")
        
        print("\n" + "-"*70)
        print("SCORES")
        print("-"*70)
        
        cl = results.continual_learning_10tasks
        print(f"\n1. 10-Task Continual Learning:")
        print(f"   BWT: {cl['BWT']:+.3f} | FWT: {cl['FWT']:+.3f} | FAP: {cl['FAP']:.3f}")
        print(f"   Forgetting: {cl['avg_forgetting']:.3f}")
        
        robust = results.robustness_tests
        print(f"\n2. Robustness: {robust['overall_robustness']:.3f}")
        print(f"   Noise degradation: {robust['noise_resilience']['degradation']:.3f}")
        print(f"   Recovery: {robust['catastrophe_recovery']['score']:.3f}")
        
        baseline = results.baseline_comparison
        print(f"\n3. Baseline: +{baseline['improvement_over_random']*100:.1f}% vs random")
        
        print("\n" + "-"*70)
        print("OVERALL ASSESSMENT")
        print("-"*70)
        print(f"\nScore: {results.overall_score:.1f} / 100")
        print(f"Grade: {results.grade}")
        print(f"\n{results.brutal_assessment}")
        print("="*70)
    
    def _save_results(self, results: FastBrutalResults):
        output_dir = Path("/mnt/f/avalanche-持续学习/daerwen3.5/benchmark_results")
        output_dir.mkdir(exist_ok=True)
        filename = f"brutal_fast_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nResults saved: {filepath}")


if __name__ == "__main__":
    suite = FastBrutalBenchmark()
    results = suite.run()
    print("\n✅ Fast brutal benchmark complete!")
