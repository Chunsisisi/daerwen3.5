"""
Quick demo version of AI Learning Benchmark
Reduced epochs for fast demonstration
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.insert(0, '/mnt/f/avalanche-持续学习/daerwen3.5')

from tests.ai_learning_benchmark import (
    AILearningBenchmarkSuite,
    ContinualLearningTest,
    FewShotLearningTest,
    StandardTaskBenchmark,
    Ecology2DConfig
)


class QuickBenchmarkSuite(AILearningBenchmarkSuite):
    """Quick version with reduced iterations for demo"""
    
    def run_quick_benchmark(self):
        """Run abbreviated benchmark"""
        print("\n" + "="*70)
        print("QUICK AI LEARNING BENCHMARK (Demo Version)")
        print("="*70)
        print("\nReduced iterations for quick demonstration")
        print("Expected time: ~2 minutes")
        
        import time
        start_time = time.time()
        
        print("\n[1/3] Continual Learning (3 tasks, 5 epochs each)...")
        cl_config = Ecology2DConfig(
            world_size=80,
            n_particles=300,
            genome_length=20,
        )
        cl_test = ContinualLearningTest(cl_config)
        
        for i in range(len(cl_test.tasks)):
            task_name, task_fn, gradient_dir = cl_test.tasks[i]
            print(f"  Training task {i}: {task_name}")
            cl_test.train_on_task(i, task_fn, epochs=5)
            
            for j in range(len(cl_test.tasks)):
                test_name, test_fn, test_dir = cl_test.tasks[j]
                score = cl_test.evaluate_task(j, test_fn, test_dir)
                cl_test.performance_matrix[i, j] = score
        
        cl_results = cl_test._calculate_metrics()
        print(f"\n  BWT: {cl_results['BWT']:+.3f}, FWT: {cl_results['FWT']:+.3f}, FAP: {cl_results['FAP']:.3f}")
        
        print("\n[2/3] Few-Shot Learning (3-way 2-shot)...")
        fs_config = Ecology2DConfig(
            world_size=80,
            n_particles=300,
        )
        fs_test = FewShotLearningTest(fs_config)
        fs_results = fs_test.run(n_way=3, k_shot=2, n_query=3)
        
        print("\n[3/3] Standard Tasks (3 tasks)...")
        st_config = Ecology2DConfig(
            world_size=80,
            n_particles=300,
        )
        st_test = StandardTaskBenchmark(st_config)
        
        quick_tasks = ['chemotaxis', 'aggregation', 'survival']
        st_results_quick = {}
        
        for task_name in quick_tasks:
            task_config = st_test.tasks[task_name]
            print(f"  Testing: {task_name}")
            
            st_test.system = st_test.config.__class__(
                world_size=80,
                n_particles=300,
            )
            st_test.system = type(cl_test.system)(st_test.config)
            
            for _ in range(100):
                st_test.system.step()
            
            task_config.setup_fn(st_test.system)
            
            for _ in range(50):
                st_test.system.step()
            
            score = task_config.metric_fn(st_test.system)
            success = score >= task_config.success_threshold
            
            st_results_quick[task_name] = {
                'score': float(score),
                'success': success,
            }
            
            status = "✓" if success else "✗"
            print(f"    {status} Score: {score:.3f}")
        
        st_results_quick['success_rate'] = sum(r['success'] for r in st_results_quick.values() if isinstance(r, dict)) / len(quick_tasks)
        
        duration = time.time() - start_time
        
        print("\n" + "="*70)
        print("QUICK BENCHMARK RESULTS")
        print("="*70)
        print(f"\nDuration: {duration:.1f} seconds")
        print(f"\n1. Continual Learning:")
        print(f"   BWT: {cl_results['BWT']:+.3f} (memory retention)")
        print(f"   FWT: {cl_results['FWT']:+.3f} (knowledge transfer)")
        print(f"   FAP: {cl_results['FAP']:.3f} (final performance)")
        
        print(f"\n2. Few-Shot Learning:")
        print(f"   3-way 2-shot: {fs_results['overall_performance']:.3f}")
        
        print(f"\n3. Standard Tasks:")
        print(f"   Success: {st_results_quick['success_rate']*100:.0f}%")
        
        overall = (
            cl_results['FAP'] * 40 +
            fs_results['overall_performance'] * 30 +
            st_results_quick['success_rate'] * 30
        )
        
        print(f"\n" + "="*70)
        print(f"Overall Score: {overall:.1f}/100")
        print("="*70)
        
        if overall >= 70:
            print("\n✅ Excellent: System demonstrates strong learning capability")
        elif overall >= 50:
            print("\n✓ Good: System shows solid learning ability")
        else:
            print("\n⚠️  Developing: System shows learning potential")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUICK AI LEARNING BENCHMARK")
    print("="*70)
    print("\nThis is an abbreviated version for quick demonstration.")
    print("For full benchmark, use ai_learning_benchmark.py")
    print("="*70)
    
    suite = QuickBenchmarkSuite()
    suite.run_quick_benchmark()
    
    print("\n✅ Quick benchmark complete!")
    print("\nFor comprehensive testing, run:")
    print("  python tests/ai_learning_benchmark.py")
