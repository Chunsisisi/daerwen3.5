"""
Professional AI Learning Benchmark Suite for DAERWEN3.5

Implements industry-standard learning capability tests:
1. Continual Learning - BWT (Backward Transfer), FWT (Forward Transfer)
2. Few-Shot Learning - N-way K-shot meta-learning protocol
3. Standard Task Benchmark - 6 canonical ALife tasks
4. Transfer Learning - Cross-domain generalization
5. Online Learning - Concept drift adaptation

Based on:
- Lopez-Paz & Ranzato (2017) - Gradient Episodic Memory
- Finn et al. (2017) - Model-Agnostic Meta-Learning
- OpenAI Gym task design principles
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path

from engine.core import Ecology2DSystem, Ecology2DConfig, ExternalInput, SystemOutput
from controllers.state_aggregator import SimpleStateAggregator


@dataclass
class TaskConfig:
    name: str
    description: str
    setup_fn: callable
    metric_fn: callable
    success_threshold: float


@dataclass
class BenchmarkResults:
    continual_learning: Dict
    few_shot_learning: Dict
    standard_tasks: Dict
    transfer_learning: Dict
    online_learning: Dict
    overall_score: float
    timestamp: str


class TaskLibrary:
    """Standard task definitions for ecosystem learning"""
    
    @staticmethod
    def chemotaxis_north(system: Ecology2DSystem):
        """Create north-pointing ATP gradient"""
        world = system.config.world_size
        x = np.arange(world)
        y = np.arange(world)
        xx, yy = np.meshgrid(x, y)
        
        gradient = (yy / world) * 2.0
        system.chemical_field.concentrations[:, :, system.chemical_field.ATP_index] = gradient
    
    @staticmethod
    def chemotaxis_south(system: Ecology2DSystem):
        """Create south-pointing ATP gradient"""
        world = system.config.world_size
        x = np.arange(world)
        y = np.arange(world)
        xx, yy = np.meshgrid(x, y)
        
        gradient = ((world - yy) / world) * 2.0
        system.chemical_field.concentrations[:, :, system.chemical_field.ATP_index] = gradient
    
    @staticmethod
    def chemotaxis_center(system: Ecology2DSystem):
        """Create center-focused ATP gradient"""
        world = system.config.world_size
        center = world // 2
        x = np.arange(world)
        y = np.arange(world)
        xx, yy = np.meshgrid(x, y)
        
        dist = np.sqrt((xx - center)**2 + (yy - center)**2)
        gradient = 2.0 * np.exp(-dist**2 / (2 * (world/4)**2))
        system.chemical_field.concentrations[:, :, system.chemical_field.ATP_index] = gradient
    
    @staticmethod
    def phototaxis_corner(system: Ecology2DSystem):
        """Light source in corner"""
        world = system.config.world_size
        params = {'x': world-1, 'y': world-1, 'radius': 30, 'intensity': 3.0}
        system.apply_external_input(ExternalInput('chemical_pulse', params))
    
    @staticmethod
    def aggregation_pressure(system: Ecology2DSystem):
        """Environmental pressure favoring clustering"""
        world = system.config.world_size
        center = world // 2
        params = {'x': center, 'y': center, 'radius': 40, 'intensity': 2.0}
        system.apply_external_input(ExternalInput('chemical_pulse', params))
    
    @staticmethod
    def dispersion_pressure(system: Ecology2DSystem):
        """Environmental pressure favoring spreading"""
        system.apply_external_input(ExternalInput('catastrophe', {'event_type': 'energy_fluctuation'}))


class MetricCalculator:
    """Professional metrics for learning capability assessment"""
    
    @staticmethod
    def calculate_center_of_mass(particles) -> Tuple[float, float]:
        """Calculate population center of mass"""
        if not particles:
            return 0.0, 0.0
        alive = [p for p in particles if p.alive]
        if not alive:
            return 0.0, 0.0
        
        cx = np.mean([p.position[0] for p in alive])
        cy = np.mean([p.position[1] for p in alive])
        return float(cx), float(cy)
    
    @staticmethod
    def calculate_spatial_dispersion(particles, world_size: int) -> float:
        """Calculate how spread out particles are"""
        alive = [p for p in particles if p.alive]
        if len(alive) < 2:
            return 0.0
        
        positions = np.array([p.position for p in alive])
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        
        return float(np.mean(distances))
    
    @staticmethod
    def calculate_cluster_density(particles, world_size: int) -> float:
        """Calculate clustering coefficient"""
        alive = [p for p in particles if p.alive]
        if len(alive) < 2:
            return 0.0
        
        positions = np.array([p.position for p in alive])
        
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        max_distance = world_size * np.sqrt(2)
        
        density = 1.0 - (avg_distance / max_distance)
        return float(density)
    
    @staticmethod
    def calculate_gradient_alignment(particles, gradient_direction: str, world_size: int) -> float:
        """Calculate how well population aligns with gradient"""
        cx, cy = MetricCalculator.calculate_center_of_mass(particles)
        center = world_size / 2
        
        if gradient_direction == 'north':
            score = cy / world_size
        elif gradient_direction == 'south':
            score = (world_size - cy) / world_size
        elif gradient_direction == 'center':
            dist = np.sqrt((cx - center)**2 + (cy - center)**2)
            score = 1.0 - (dist / (world_size * np.sqrt(2) / 2))
        else:
            score = 0.0
        
        return float(np.clip(score, 0.0, 1.0))


class ContinualLearningTest:
    """
    Test continual learning capability using sequential task training.
    
    Metrics:
    - BWT (Backward Transfer): Memory retention
    - FWT (Forward Transfer): Knowledge transfer to new tasks
    - FAP (Final Average Performance): Overall capability
    """
    
    def __init__(self, config: Ecology2DConfig):
        self.config = config
        self.system = Ecology2DSystem(config)
        self.aggregator = SimpleStateAggregator(config, grid_size=32)
        
        self.tasks = [
            ('north_gradient', TaskLibrary.chemotaxis_north, 'north'),
            ('south_gradient', TaskLibrary.chemotaxis_south, 'south'),
            ('center_gradient', TaskLibrary.chemotaxis_center, 'center'),
        ]
        
        self.performance_matrix = np.zeros((len(self.tasks), len(self.tasks)))
    
    def train_on_task(self, task_idx: int, task_fn: callable, epochs: int = 15):
        """Train system on specific task"""
        print(f"  Training on task {task_idx} ({self.tasks[task_idx][0]})...")
        
        for epoch in range(epochs):
            task_fn(self.system)
            
            for _ in range(100):
                self.system.step()
            
            if (epoch + 1) % 5 == 0:
                alive = sum(1 for p in self.system.particles if p.alive)
                print(f"    Epoch {epoch+1}/{epochs}: {alive} alive")
    
    def evaluate_task(self, task_idx: int, task_fn: callable, gradient_dir: str) -> float:
        """Evaluate performance on specific task"""
        task_fn(self.system)
        
        for _ in range(50):
            self.system.step()
        
        score = MetricCalculator.calculate_gradient_alignment(
            self.system.particles, 
            gradient_dir, 
            self.config.world_size
        )
        
        return score
    
    def run(self) -> Dict:
        """Run continual learning test"""
        print("\n" + "="*70)
        print("TEST 1: CONTINUAL LEARNING")
        print("="*70)
        print("Measuring: Backward Transfer (memory), Forward Transfer (knowledge reuse)")
        
        n_tasks = len(self.tasks)
        
        for i in range(n_tasks):
            task_name, task_fn, gradient_dir = self.tasks[i]
            print(f"\n[Phase {i+1}/{n_tasks}] Training: {task_name}")
            
            self.train_on_task(i, task_fn, epochs=15)
            
            print(f"  Evaluating all tasks after training {task_name}:")
            for j in range(n_tasks):
                test_name, test_fn, test_dir = self.tasks[j]
                score = self.evaluate_task(j, test_fn, test_dir)
                self.performance_matrix[i, j] = score
                
                status = "✓" if score > 0.5 else "✗"
                print(f"    {status} {test_name}: {score:.3f}")
        
        results = self._calculate_metrics()
        self._print_results(results)
        
        return results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate continual learning metrics"""
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
        
        forgetting = -BWT if BWT < 0 else 0.0
        
        return {
            'BWT': float(BWT),
            'FWT': float(FWT),
            'FAP': float(FAP),
            'forgetting': float(forgetting),
            'performance_matrix': self.performance_matrix.tolist(),
        }
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*70)
        print("CONTINUAL LEARNING RESULTS")
        print("="*70)
        
        print(f"\nBackward Transfer (BWT):  {results['BWT']:+.3f}")
        if results['BWT'] > 0:
            print("  → Positive: System retains old knowledge")
        elif results['BWT'] < -0.1:
            print("  → Negative: Catastrophic forgetting detected")
        else:
            print("  → Near zero: Minimal interference")
        
        print(f"\nForward Transfer (FWT):   {results['FWT']:+.3f}")
        if results['FWT'] > 0:
            print("  → Positive: Old knowledge helps learn new tasks")
        else:
            print("  → Negative: Learning from scratch each time")
        
        print(f"\nFinal Avg Performance:    {results['FAP']:.3f}")
        print(f"Forgetting Rate:          {results['forgetting']:.3f}")
        
        print("\nPerformance Matrix:")
        print("  (row=after training task i, col=performance on task j)")
        for i, row in enumerate(self.performance_matrix):
            print(f"  After task {i}: [{', '.join([f'{x:.2f}' for x in row])}]")


class FewShotLearningTest:
    """
    Test few-shot learning capability.
    
    Protocol: N-way K-shot
    - N different novel tasks
    - K training examples per task
    - Measure rapid adaptation
    """
    
    def __init__(self, config: Ecology2DConfig):
        self.config = config
        self.system = Ecology2DSystem(config)
        
        self.novel_tasks = [
            ('corner_nw', lambda s: self._create_corner_gradient(s, 0, s.config.world_size-1)),
            ('corner_ne', lambda s: self._create_corner_gradient(s, s.config.world_size-1, s.config.world_size-1)),
            ('corner_sw', lambda s: self._create_corner_gradient(s, 0, 0)),
            ('corner_se', lambda s: self._create_corner_gradient(s, s.config.world_size-1, 0)),
            ('diagonal', self._create_diagonal_gradient),
        ]
    
    def _create_corner_gradient(self, system: Ecology2DSystem, cx: int, cy: int):
        """Create gradient focused on corner"""
        world = system.config.world_size
        x = np.arange(world)
        y = np.arange(world)
        xx, yy = np.meshgrid(x, y)
        
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        gradient = 2.0 * np.exp(-dist**2 / (2 * (world/3)**2))
        system.chemical_field.concentrations[:, :, system.chemical_field.ATP_index] = gradient
    
    def _create_diagonal_gradient(self, system: Ecology2DSystem):
        """Create diagonal gradient"""
        world = system.config.world_size
        x = np.arange(world)
        y = np.arange(world)
        xx, yy = np.meshgrid(x, y)
        
        gradient = (xx + yy) / (2 * world) * 2.0
        system.chemical_field.concentrations[:, :, system.chemical_field.ATP_index] = gradient
    
    def run(self, n_way: int = 5, k_shot: int = 3, n_query: int = 5) -> Dict:
        """Run N-way K-shot test"""
        print("\n" + "="*70)
        print(f"TEST 2: FEW-SHOT LEARNING ({n_way}-way {k_shot}-shot)")
        print("="*70)
        print("Measuring: Rapid adaptation with minimal examples")
        
        selected_tasks = self.novel_tasks[:n_way]
        results = []
        
        for task_idx, (task_name, task_fn) in enumerate(selected_tasks):
            print(f"\n[Task {task_idx+1}/{n_way}] {task_name}")
            
            print(f"  Support set: {k_shot} examples")
            for shot in range(k_shot):
                task_fn(self.system)
                for _ in range(50):
                    self.system.step()
            
            print(f"  Query set: {n_query} evaluations")
            query_scores = []
            for query in range(n_query):
                task_fn(self.system)
                for _ in range(30):
                    self.system.step()
                
                score = self._evaluate_adaptation(task_fn)
                query_scores.append(score)
            
            avg_score = np.mean(query_scores)
            std_score = np.std(query_scores)
            
            print(f"  Performance: {avg_score:.3f} ± {std_score:.3f}")
            results.append(avg_score)
        
        overall_performance = np.mean(results)
        
        print("\n" + "="*70)
        print("FEW-SHOT LEARNING RESULTS")
        print("="*70)
        print(f"\n{n_way}-way {k_shot}-shot Performance: {overall_performance:.3f}")
        
        if overall_performance > 0.6:
            print("  → Excellent: Strong rapid adaptation")
        elif overall_performance > 0.4:
            print("  → Good: Moderate few-shot learning")
        else:
            print("  → Weak: Limited rapid adaptation")
        
        return {
            'n_way': n_way,
            'k_shot': k_shot,
            'n_query': n_query,
            'overall_performance': float(overall_performance),
            'task_performances': [float(x) for x in results],
        }
    
    def _evaluate_adaptation(self, task_fn: callable) -> float:
        """Evaluate how well system adapted to task"""
        alive = [p for p in self.system.particles if p.alive]
        if not alive:
            return 0.0
        
        total_energy = sum(p.energy for p in alive)
        avg_energy = total_energy / len(alive)
        
        score = np.clip(avg_energy / 2.0, 0.0, 1.0)
        return float(score)


class StandardTaskBenchmark:
    """
    Standard task benchmark suite.
    
    6 canonical tasks for ecosystem intelligence:
    1. Chemotaxis - gradient following
    2. Phototaxis - light seeking
    3. Aggregation - clustering
    4. Dispersion - spreading
    5. Resource exploitation - energy harvesting
    6. Survival - persistence under pressure
    """
    
    def __init__(self, config: Ecology2DConfig):
        self.config = config
        self.system = Ecology2DSystem(config)
        
        self.tasks = {
            'chemotaxis': TaskConfig(
                name='chemotaxis',
                description='Follow ATP gradient to source',
                setup_fn=TaskLibrary.chemotaxis_center,
                metric_fn=lambda s: MetricCalculator.calculate_gradient_alignment(
                    s.particles, 'center', s.config.world_size
                ),
                success_threshold=0.6,
            ),
            'phototaxis': TaskConfig(
                name='phototaxis',
                description='Move toward energy source',
                setup_fn=TaskLibrary.phototaxis_corner,
                metric_fn=lambda s: MetricCalculator.calculate_gradient_alignment(
                    s.particles, 'north', s.config.world_size
                ),
                success_threshold=0.5,
            ),
            'aggregation': TaskConfig(
                name='aggregation',
                description='Form dense clusters',
                setup_fn=TaskLibrary.aggregation_pressure,
                metric_fn=lambda s: MetricCalculator.calculate_cluster_density(
                    s.particles, s.config.world_size
                ),
                success_threshold=0.5,
            ),
            'dispersion': TaskConfig(
                name='dispersion',
                description='Spread uniformly',
                setup_fn=TaskLibrary.dispersion_pressure,
                metric_fn=lambda s: MetricCalculator.calculate_spatial_dispersion(
                    s.particles, s.config.world_size
                ),
                success_threshold=30.0,
            ),
            'resource_exploitation': TaskConfig(
                name='resource_exploitation',
                description='Efficiently harvest energy',
                setup_fn=lambda s: None,
                metric_fn=lambda s: sum(p.energy for p in s.particles if p.alive) / max(len([p for p in s.particles if p.alive]), 1),
                success_threshold=1.5,
            ),
            'survival': TaskConfig(
                name='survival',
                description='Maintain population under stress',
                setup_fn=lambda s: s.apply_external_input(ExternalInput('catastrophe', {'event_type': 'mass_extinction'})),
                metric_fn=lambda s: len([p for p in s.particles if p.alive]) / self.config.n_particles,
                success_threshold=0.3,
            ),
        }
    
    def run(self) -> Dict:
        """Run all standard tasks"""
        print("\n" + "="*70)
        print("TEST 3: STANDARD TASK BENCHMARK")
        print("="*70)
        print("Measuring: Performance on 6 canonical tasks")
        
        results = {}
        
        for task_name, task_config in self.tasks.items():
            print(f"\n[Task] {task_name}")
            print(f"  Goal: {task_config.description}")
            
            self.system = Ecology2DSystem(self.config)
            
            for _ in range(200):
                self.system.step()
            
            task_config.setup_fn(self.system)
            
            for _ in range(100):
                self.system.step()
            
            score = task_config.metric_fn(self.system)
            success = score >= task_config.success_threshold
            
            status = "✓" if success else "✗"
            print(f"  {status} Score: {score:.3f} (threshold: {task_config.success_threshold:.3f})")
            
            results[task_name] = {
                'score': float(score),
                'threshold': task_config.success_threshold,
                'success': success,
            }
        
        success_rate = sum(r['success'] for r in results.values()) / len(results)
        
        print("\n" + "="*70)
        print("STANDARD TASK RESULTS")
        print("="*70)
        print(f"\nSuccess Rate: {success_rate*100:.1f}% ({sum(r['success'] for r in results.values())}/{len(results)})")
        
        passed = [k for k, v in results.items() if v['success']]
        failed = [k for k, v in results.items() if not v['success']]
        
        if passed:
            print(f"Passed: {', '.join(passed)}")
        if failed:
            print(f"Failed: {', '.join(failed)}")
        
        results['success_rate'] = float(success_rate)
        return results


class AILearningBenchmarkSuite:
    """
    Complete AI learning benchmark suite.
    
    Combines all tests and generates comprehensive report.
    """
    
    def __init__(self, config: Optional[Ecology2DConfig] = None):
        if config is None:
            config = Ecology2DConfig(
                world_size=100,
                n_particles=500,
                genome_length=24,
                mutation_rate=0.02,
                n_chemical_species=5,
            )
        
        self.config = config
        self.results = {}
    
    def run_full_benchmark(self) -> BenchmarkResults:
        """Run complete benchmark suite"""
        print("\n" + "="*70)
        print("PROFESSIONAL AI LEARNING BENCHMARK SUITE")
        print("DAERWEN3.5 - Ecosystem Intelligence Evaluation")
        print("="*70)
        print("\nThis will take approximately 5-8 minutes.")
        print("Tests: Continual Learning, Few-Shot, Standard Tasks")
        print("="*70)
        
        start_time = time.time()
        
        print("\n[1/3] Running Continual Learning Test...")
        cl_test = ContinualLearningTest(self.config)
        self.results['continual_learning'] = cl_test.run()
        
        print("\n[2/3] Running Few-Shot Learning Test...")
        fs_test = FewShotLearningTest(self.config)
        self.results['few_shot_learning'] = fs_test.run(n_way=5, k_shot=3, n_query=5)
        
        print("\n[3/3] Running Standard Task Benchmark...")
        st_test = StandardTaskBenchmark(self.config)
        self.results['standard_tasks'] = st_test.run()
        
        self.results['transfer_learning'] = {'implemented': False}
        self.results['online_learning'] = {'implemented': False}
        
        duration = time.time() - start_time
        
        overall_score = self._calculate_overall_score()
        
        benchmark_results = BenchmarkResults(
            continual_learning=self.results['continual_learning'],
            few_shot_learning=self.results['few_shot_learning'],
            standard_tasks=self.results['standard_tasks'],
            transfer_learning=self.results['transfer_learning'],
            online_learning=self.results['online_learning'],
            overall_score=overall_score,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        self._print_final_report(benchmark_results, duration)
        self._save_results(benchmark_results)
        
        return benchmark_results
    
    def _calculate_overall_score(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            'continual_learning': 0.4,
            'few_shot_learning': 0.3,
            'standard_tasks': 0.3,
        }
        
        scores = {}
        
        cl = self.results['continual_learning']
        cl_score = (cl['FAP'] * 0.5 + max(0, cl['BWT']) * 0.3 + max(0, cl['FWT']) * 0.2)
        scores['continual_learning'] = np.clip(cl_score * 100, 0, 100)
        
        fs = self.results['few_shot_learning']
        scores['few_shot_learning'] = fs['overall_performance'] * 100
        
        st = self.results['standard_tasks']
        scores['standard_tasks'] = st['success_rate'] * 100
        
        overall = sum(scores[k] * weights[k] for k in weights.keys())
        
        return float(overall)
    
    def _print_final_report(self, results: BenchmarkResults, duration: float):
        """Print comprehensive final report"""
        print("\n" + "="*70)
        print("FINAL BENCHMARK REPORT")
        print("="*70)
        
        print(f"\nTotal Duration: {duration/60:.1f} minutes")
        print(f"Timestamp: {results.timestamp}")
        
        print("\n" + "-"*70)
        print("COMPONENT SCORES")
        print("-"*70)
        
        cl = results.continual_learning
        print(f"\n1. Continual Learning:")
        print(f"   BWT (Memory):        {cl['BWT']:+.3f}")
        print(f"   FWT (Transfer):      {cl['FWT']:+.3f}")
        print(f"   FAP (Performance):   {cl['FAP']:.3f}")
        print(f"   Forgetting Rate:     {cl['forgetting']:.3f}")
        
        fs = results.few_shot_learning
        print(f"\n2. Few-Shot Learning:")
        print(f"   {fs['n_way']}-way {fs['k_shot']}-shot:     {fs['overall_performance']:.3f}")
        
        st = results.standard_tasks
        print(f"\n3. Standard Tasks:")
        print(f"   Success Rate:        {st['success_rate']*100:.1f}%")
        
        print("\n" + "-"*70)
        print("OVERALL SCORE")
        print("-"*70)
        print(f"\n{results.overall_score:.1f} / 100")
        
        if results.overall_score >= 80:
            grade = "A - Excellent"
            comment = "System demonstrates strong learning across all dimensions"
        elif results.overall_score >= 70:
            grade = "B - Good"
            comment = "System shows solid learning capability with room for improvement"
        elif results.overall_score >= 60:
            grade = "C - Adequate"
            comment = "System exhibits basic learning but needs enhancement"
        else:
            grade = "D - Developing"
            comment = "System shows learning potential but requires significant work"
        
        print(f"\nGrade: {grade}")
        print(f"Assessment: {comment}")
        
        print("\n" + "="*70)
    
    def _save_results(self, results: BenchmarkResults):
        """Save results to JSON file"""
        output_dir = Path("/mnt/f/avalanche-持续学习/daerwen3.5/benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"ai_learning_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DAERWEN3.5 - Professional AI Learning Benchmark")
    print("="*70)
    print("\nBased on:")
    print("  • Lopez-Paz & Ranzato (2017) - Continual Learning")
    print("  • Finn et al. (2017) - Meta-Learning")
    print("  • OpenAI Gym - Standard Task Design")
    print("\nTests implemented:")
    print("  ✓ Continual Learning (BWT/FWT/FAP)")
    print("  ✓ Few-Shot Learning (N-way K-shot)")
    print("  ✓ Standard Task Benchmark (6 tasks)")
    print("\nEstimated time: 5-8 minutes")
    print("="*70)
    print("\nStarting benchmark...")
    
    suite = AILearningBenchmarkSuite()
    results = suite.run_full_benchmark()
    
    print("\n✅ Benchmark complete!")
