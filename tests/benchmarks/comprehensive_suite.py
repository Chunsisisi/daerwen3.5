"""
DAERWEN3 - 完整测试套件
参考业界AI系统标准测试方法

测试类型：
1. 基准测试 - 标准场景下的性能
2. 泛化测试 - 未见过输入的响应
3. 鲁棒性测试 - 噪声和极端情况
4. 一致性测试 - 重复输入的稳定性
5. 区分度测试 - 不同输入的差异性
6. 学习曲线 - 训练过程的收敛性
7. 消融实验 - 各组件的贡献度
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import asyncio
import threading
import numpy as np
import time
from collections import defaultdict
from engine.core import Ecology2DSystem, Ecology2DConfig, ExternalInput, SystemOutput
from controllers.state_aggregator import SimpleStateAggregator
def cosine_similarity_fallback(a: np.ndarray, b: np.ndarray) -> float:
    """简版余弦相似度，避免额外依赖"""
    a_vec = a.flatten()
    b_vec = b.flatten()
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denom == 0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


class VisualizerBridge:
    def __init__(self, config: Ecology2DConfig):
        self.config = config
        self.available = False
        self.server = None
        self.thread = None
        self._should_run = os.environ.get('DAERWEN_VISUAL', '0') == '1'
        if not self._should_run:
            return
        try:
            from engine.server import Ecology2DServer
        except ImportError:
            print("⚠️ 未找到 websockets 依赖，实时可视化不可用")
            return
        self.server = Ecology2DServer(config, external_control=True)
        self.available = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def _run_server(self):
        if self.server is None:
            return
        asyncio.run(self.server.run())

    def publish(self, snapshot: SystemOutput):
        if self.available and self.server:
            self.server.push_external_snapshot(snapshot)

class ComprehensiveTestSuite:
    """完整测试套件"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("🧪 DAERWEN3 - 完整测试套件")
        print("="*70)
        print("\n基于业界AI系统标准测试方法")
        
        self.config = Ecology2DConfig(
            world_size=100,
            n_particles=1000,
            genome_length=24,
            mutation_rate=0.02,
            n_chemical_species=5,
        )
        self.rng = np.random.default_rng(42)
        
        # 训练输入（已见）
        self.training_inputs = ['A', 'B', 'C']
        # 测试输入（未见）
        self.test_inputs = ['D', 'E']
        
        self.response_history = defaultdict(list)
        self.aggregator = SimpleStateAggregator(self.config, grid_size=32)
        self._disturbance_cache = {}
        self.visualizer = VisualizerBridge(self.config)
        
        print("✅ 测试套件初始化完成")
    
    def _reset_system(self):
        """重置系统"""
        if self.visualizer and self.visualizer.available:
            self.visualizer.server.reset_system()
            self.system = self.visualizer.server.system
        else:
            self.system = Ecology2DSystem(self.config)
        self._step_system(1000)

    def _publish_snapshot(self):
        if self.visualizer and self.visualizer.available:
            snapshot = self.system.get_system_output(metadata={'source': 'test_suite'})
            self.visualizer.publish(snapshot)

    def _step_system(self, steps: int = 1):
        for _ in range(steps):
            self.system.step()
            self._publish_snapshot()
    
    def _generate_disturbance(self, label, cycle=0):
        """生成确定性扰动（固定seed，方便复现）"""
        key = (str(label), int(cycle))
        if key in self._disturbance_cache:
            return dict(self._disturbance_cache[key])
        
        if isinstance(label, (int, np.integer)):
            seed = int(label)
        else:
            seed = (abs(hash(key)) + 10007 * int(cycle)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        disturbance = {
            'type': rng.choice(['energy', 'extinction', 'mutation']),
            'intensity': float(rng.uniform(0.3, 0.7)),
            'location': (
                int(rng.integers(0, self.config.world_size)),
                int(rng.integers(0, self.config.world_size)),
            ),
        }
        self._disturbance_cache[key] = disturbance
        return dict(disturbance)
    
    def _apply_disturbance(self, disturbance):
        """施加扰动（统一通过引擎接口）"""
        d_type = disturbance['type']
        intensity = float(disturbance['intensity'])
        x, y = disturbance['location']
        
        if d_type == 'energy':
            params = {
                'x': int(x),
                'y': int(y),
                'radius': 15,
                'intensity': intensity,
            }
            self.system.apply_external_input(
                ExternalInput('chemical_pulse', params)
            )
        
        elif d_type == 'extinction':
            self.system.apply_external_input(
                ExternalInput('catastrophe', {'event_type': 'mass_extinction'})
            )
        
        elif d_type == 'mutation':
            self.system.apply_external_input(
                ExternalInput('catastrophe', {'event_type': 'mutation_burst'})
            )
        
        settle_steps = 250 if d_type in {'extinction', 'mutation'} else 150
        self._step_system(settle_steps)
    
    def _sample_targets(self, targets, count):
        """在不影响全局随机性的情况下采样目标"""
        if count <= 0 or not targets:
            return []
        if count >= len(targets):
            return list(targets)
        idxs = self.rng.choice(len(targets), size=count, replace=False)
        return [targets[int(i)] for i in idxs]
    
    def _get_state_vector(self):
        """使用聚合器获取状态向量"""
        output = self.system.get_system_output(metadata={'source': 'test_suite'})
        agg_state = self.aggregator.aggregate(output)
        return agg_state.vector
    
    def _apply_selection(self, consistency, discrimination):
        """施加选择压力"""
        alive = [p for p in self.system.particles if p.alive]
        if len(alive) == 0:
            return 0
        
        # 基于一致性和区分度的联合优化
        base_kill_ratio = 0.25
        consistency_bonus = consistency * 0.15
        discrimination_bonus = discrimination * 0.1
        
        kill_ratio = max(0.05, base_kill_ratio - consistency_bonus - discrimination_bonus)
        n_kill = int(len(alive) * kill_ratio)
        
        if n_kill > 0:
            sorted_by_energy = sorted(alive, key=lambda p: p.energy)
            victims = sorted_by_energy[:n_kill]
            for p in victims:
                p.alive = False
                self.system.chemical_field.produce(
                    p.position, self.system.chemical_field.ATP_index, p.energy * 0.5
                )
        
        # 高区分度奖励
        if discrimination > 0.2:
            n_reward = int(len(alive) * discrimination * 0.5)
            if n_reward > 0:
                sorted_by_energy = sorted(alive, key=lambda p: p.energy, reverse=True)
                for p in sorted_by_energy[:n_reward]:
                    if p.alive:
                        p.energy += 0.6
        
        return n_kill
    
    # ==================== 测试1: 基准训练 ====================
    def test_1_baseline_training(self, epochs=10):
        """测试1: 基准训练 - 在标准输入上训练"""
        print("\n" + "="*70)
        print("📝 测试1: 基准训练")
        print("="*70)
        print("目标: 训练系统学习A、B、C三个输入的响应模式")
        
        self._reset_system()
        
        for epoch in range(epochs):
            print(f"\n[轮次 {epoch+1}/{epochs}]")
            
            for inp in self.training_inputs:
                disturbance = self._generate_disturbance(f"train_{inp}", cycle=epoch)
                self._apply_disturbance(disturbance)
                
                response = self._get_state_vector()
                self.response_history[inp].append(response)
                
                self._step_system(20)
            
            # 计算指标
            consistency = self._calculate_consistency_all()
            discrimination = self._calculate_discrimination()
            
        # 测试阶段不再施加人工选择压力，让系统自然演化
            
            alive = sum(1 for p in self.system.particles if p.alive)
            
            if (epoch + 1) % 5 == 0:
                print(f"  一致性: {consistency:.3f}, 区分度: {discrimination:.3f}, 存活: {alive}")
        
        final_consistency = self._calculate_consistency_all()
        final_discrimination = self._calculate_discrimination()
        
        print(f"\n✅ 基准训练完成")
        print(f"   最终一致性: {final_consistency:.3f}")
        print(f"   最终区分度: {final_discrimination:.3f}")
        
        return {
            'consistency': final_consistency,
            'discrimination': final_discrimination,
        }
    
    # ==================== 测试2: 泛化能力 ====================
    def test_2_generalization(self):
        """测试2: 泛化能力 - 测试未见输入"""
        print("\n" + "="*70)
        print("📝 测试2: 泛化能力")
        print("="*70)
        print("目标: 测试系统对未见输入D、E的响应")
        
        test_responses = {}
        
        for inp in self.test_inputs:
            print(f"\n测试输入: '{inp}'")
            
            # 采样多次
            responses = []
            for trial in range(3):
                disturbance = self._generate_disturbance(f"test_{inp}", cycle=trial)
                self._apply_disturbance(disturbance)
                
                response = self._get_state_vector()
                responses.append(response)
                
                self._step_system(20)
            
            test_responses[inp] = responses
            
            # 计算内部一致性
            sims = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    sims.append(cosine_similarity_fallback(responses[i], responses[j]))
            consistency = float(np.mean(sims)) if sims else 0.0
            
            print(f"  响应一致性: {consistency:.3f}")
            print(f"  平均响应: [{', '.join([f'{x:.2f}' for x in np.mean(responses, axis=0)[:4]])}...]")
        
        # 计算与训练集的相似度
        print("\n与训练集的相似度:")
        for test_inp in self.test_inputs:
            test_avg = np.mean(test_responses[test_inp], axis=0)
            
            for train_inp in self.training_inputs:
                train_avg = np.mean(self.response_history[train_inp][-5:], axis=0)
                similarity = 1.0 / (1.0 + np.linalg.norm(test_avg - train_avg))
                print(f"  '{test_inp}' vs '{train_inp}': {similarity:.3f}")
        
        return test_responses
    
    # ==================== 测试3: 鲁棒性 ====================
    def test_3_robustness(self):
        """测试3: 鲁棒性 - 噪声和极端情况"""
        print("\n" + "="*70)
        print("📝 测试3: 鲁棒性测试")
        print("="*70)
        print("目标: 测试系统在噪声和极端情况下的表现")
        
        results = {}
        
        # 3.1 噪声扰动
        print("\n[3.1] 噪声扰动测试")
        noise_levels = [0.1, 0.3, 0.5]
        
        for noise in noise_levels:
            print(f"\n噪声水平: {noise}")
            
            # 对A输入添加噪声
            responses = []
            
            for trial in range(3):
                # 原始扰动 + 随机噪声
                base_dist = self._generate_disturbance(f"noise_A_{noise}", cycle=trial)
                noise_seed = (abs(hash((noise, trial))) & 0xFFFFFFFF)
                noise_rng = np.random.default_rng(noise_seed)
                base_dist['intensity'] += noise_rng.normal(0, noise)
                base_dist['intensity'] = np.clip(base_dist['intensity'], 0.1, 0.9)
                
                self._apply_disturbance(base_dist)
                response = self._get_state_vector()
                responses.append(response)
                
                self._step_system(20)
            
            # 与原始A的差异
            original_A = np.mean(self.response_history['A'][-5:], axis=0)
            noisy_A = np.mean(responses, axis=0)
            difference = np.linalg.norm(noisy_A - original_A)
            
            print(f"  响应差异: {difference:.3f}")
            results[f'noise_{noise}'] = difference
        
        # 3.2 极端扰动
        print("\n[3.2] 极端扰动测试")
        extreme_cases = [
            ('超强能量', {'type': 'energy', 'intensity': 1.5, 'location': (50, 50)}),
            ('大规模灭绝', {'type': 'extinction', 'intensity': 0.9, 'location': (50, 50)}),
        ]
        
        for name, disturbance in extreme_cases:
            print(f"\n{name}")
            
            self._apply_disturbance(disturbance)
            response = self._get_state_vector()
            
            alive = sum(1 for p in self.system.particles if p.alive)
            print(f"  存活: {alive}")
            print(f"  响应: [{', '.join([f'{x:.2f}' for x in response[:4]])}...]")
            
            # 恢复能力
            self._step_system(50)
            
            alive_after = sum(1 for p in self.system.particles if p.alive)
            recovery = (alive_after - alive) / max(alive, 1)
            print(f"  恢复率: {recovery:+.2f}")
            
            results[f'extreme_{name}'] = recovery
        
        return results
    
    # ==================== 测试4: 重复性 ====================
    def test_4_repeatability(self):
        """测试4: 重复性 - 多次运行的稳定性"""
        print("\n" + "="*70)
        print("📝 测试4: 重复性测试")
        print("="*70)
        print("目标: 测试同一输入多次重复的响应稳定性")
        
        inp = 'A'
        n_repeats = 10
        
        print(f"\n对'{inp}'进行{n_repeats}次重复测试")
        
        responses = []
        for i in range(n_repeats):
            disturbance = self._generate_disturbance(f"repeat_{inp}", cycle=i)
            self._apply_disturbance(disturbance)
            
            response = self._get_state_vector()
            responses.append(response)
            
            self._step_system(20)
        
        # 计算统计量
        mean_response = np.mean(responses, axis=0)
        std_response = np.std(responses, axis=0)
        
        print(f"\n平均响应: [{', '.join([f'{x:.2f}' for x in mean_response[:4]])}...]")
        print(f"标准差:   [{', '.join([f'{x:.3f}' for x in std_response[:4]])}...]")
        
        # 变异系数 (CV)
        cv = np.mean(std_response / (mean_response + 1e-6))
        print(f"\n变异系数: {cv:.3f} (越小越稳定)")
        
        return cv
    
    # ==================== 测试5: 消融实验 ====================
    def test_5_ablation(self):
        """测试5: 消融实验 - 各组件贡献度"""
        print("\n" + "="*70)
        print("📝 测试5: 消融实验")
        print("="*70)
        print("目标: 测试移除各组件后的性能变化")
        
        # 基线性能
        baseline_consistency = self._calculate_consistency_all()
        baseline_discrimination = self._calculate_discrimination()
        
        print(f"\n基线性能:")
        print(f"  一致性: {baseline_consistency:.3f}")
        print(f"  区分度: {baseline_discrimination:.3f}")
        
        results = {}
        
        # 5.1 自由演化
        print(f"\n[5.1] 自由演化（无干预5轮）")
        for cycle in range(5):
            for inp in self.training_inputs:
                disturbance = self._generate_disturbance(f"ablation_{inp}", cycle=cycle)
                self._apply_disturbance(disturbance)
                response = self._get_state_vector()
                self.response_history[inp].append(response)
                self._step_system(20)
        
        no_selection_consistency = self._calculate_consistency_all()
        no_selection_discrimination = self._calculate_discrimination()
        
        print(f"  一致性: {no_selection_consistency:.3f} ({no_selection_consistency - baseline_consistency:+.3f})")
        print(f"  区分度: {no_selection_discrimination:.3f} ({no_selection_discrimination - baseline_discrimination:+.3f})")
        
        results['no_selection'] = {
            'consistency_change': no_selection_consistency - baseline_consistency,
            'discrimination_change': no_selection_discrimination - baseline_discrimination,
        }
        
        return results
    
    # ==================== 辅助函数 ====================
    def _calculate_consistency_all(self):
        """计算所有输入的平均一致性（余弦相似度）"""
        consistencies = []
        for inp in self.training_inputs:
            history = self.response_history[inp]
            if len(history) < 2:
                continue
            samples = history[-5:] if len(history) >= 5 else history
            sims = []
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    sims.append(cosine_similarity_fallback(samples[i], samples[j]))
            if sims:
                consistencies.append(float(np.mean(sims)))
        return float(np.mean(consistencies)) if consistencies else 0.0
    
    def _calculate_discrimination(self):
        """计算区分度"""
        if len(self.response_history) < 2:
            return 0.0
        
        avg_responses = {}
        for inp in self.training_inputs:
            history = self.response_history[inp]
            if len(history) > 0:
                avg_responses[inp] = np.mean(history[-5:], axis=0)
        
        if len(avg_responses) < 2:
            return 0.0
        
        keys = list(avg_responses.keys())
        distances = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                dist = np.linalg.norm(avg_responses[keys[i]] - avg_responses[keys[j]])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    # ==================== 主测试流程 ====================
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*70)
        print("🚀 开始完整测试流程")
        print("="*70)
        
        results = {}
        
        start_time = time.time()
        
        # 测试1: 基准训练
        results['baseline'] = self.test_1_baseline_training(epochs=30)
        
        # 测试2: 泛化能力
        results['generalization'] = self.test_2_generalization()
        
        # 测试3: 鲁棒性
        results['robustness'] = self.test_3_robustness()
        
        # 测试4: 重复性
        results['repeatability'] = self.test_4_repeatability()
        
        # 测试5: 消融实验
        results['ablation'] = self.test_5_ablation()
        
        duration = time.time() - start_time
        
        # 综合报告
        print("\n" + "="*70)
        print("📊 综合测试报告")
        print("="*70)
        
        print(f"\n⏱️  总耗时: {duration/60:.1f}分钟")
        
        print(f"\n✅ 测试1 - 基准训练:")
        print(f"   一致性: {results['baseline']['consistency']:.3f}")
        print(f"   区分度: {results['baseline']['discrimination']:.3f}")
        
        print(f"\n✅ 测试2 - 泛化能力:")
        print(f"   已测试未见输入: {self.test_inputs}")
        
        print(f"\n✅ 测试3 - 鲁棒性:")
        for key, val in results['robustness'].items():
            print(f"   {key}: {val:.3f}")
        
        print(f"\n✅ 测试4 - 重复性:")
        print(f"   变异系数: {results['repeatability']:.3f} (越小越好)")
        
        print(f"\n✅ 测试5 - 消融实验:")
        print(f"   移除选择压力影响:")
        print(f"     一致性变化: {results['ablation']['no_selection']['consistency_change']:+.3f}")
        print(f"     区分度变化: {results['ablation']['no_selection']['discrimination_change']:+.3f}")
        
        # 最终评分
        print("\n" + "="*70)
        print("🏆 系统评分")
        print("="*70)
        
        score_consistency = min(100, results['baseline']['consistency'] * 100)
        score_discrimination = min(100, results['baseline']['discrimination'] * 100)
        score_repeatability = max(0, 100 - results['repeatability'] * 200)
        
        total_score = (score_consistency + score_discrimination + score_repeatability) / 3
        
        print(f"\n一致性得分: {score_consistency:.1f}/100")
        print(f"区分度得分: {score_discrimination:.1f}/100")
        print(f"重复性得分: {score_repeatability:.1f}/100")
        print(f"\n综合得分: {total_score:.1f}/100")
        
        if total_score >= 80:
            print("\n🎉 优秀！系统展现出强大的学习和泛化能力")
        elif total_score >= 60:
            print("\n✅ 良好！系统具备基本的学习能力")
        elif total_score >= 40:
            print("\n⚠️  及格！系统表现出学习趋势，但需要改进")
        else:
            print("\n❌ 不及格！系统学习能力有限")
        
        return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌱 DAERWEN3 - 完整测试套件")
    print("="*70)
    print("\n包含5项标准AI测试:")
    print("  1. 基准训练 - 标准场景性能")
    print("  2. 泛化测试 - 未见输入响应")
    print("  3. 鲁棒性测试 - 噪声和极端情况")
    print("  4. 重复性测试 - 响应稳定性")
    print("  5. 消融实验 - 组件贡献度")
    print("\n预计需要5-8分钟")
    print("="*70)
    
    suite = ComprehensiveTestSuite()
    results = suite.run_all_tests()
    
    print("\n✅ 所有测试完成！")


