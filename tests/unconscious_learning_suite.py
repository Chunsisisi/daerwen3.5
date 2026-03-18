#!/usr/bin/env python3
"""
WORLD FIRST: Unconscious Learning Test Suite for Emergent AI Systems

This test suite measures "unconscious" (implicit, emergent, distributed) learning
rather than "conscious" (explicit, task-oriented, goal-directed) learning.

Inspired by research from:
- Cognitive Science: SICR (Statistically-Induced Chunking Recall)
- Complex Systems: CADI (Chaos-Aware Design Index)
- Swarm Intelligence: MetrIntMeas, Algebraic Connectivity

Philosophy:
- Tests IMPLICIT adaptation (no explicit goals)
- Tests EMERGENT properties (system-level, not individual)
- Tests DISTRIBUTED learning (population, not agents)

Author: daerwen3.5 project
Date: 2026-03-18
Status: WORLD FIRST PROTOTYPE
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from datetime import datetime
from engine.core import Ecology2DSystem, Ecology2DConfig
from typing import Dict, List, Tuple
import json


class UnconsciousLearningTestSuite:
    
    def __init__(self, world_size: int = 80, n_particles: int = 300):
        self.world_size = world_size
        self.n_particles = n_particles
        self.results = {}
        
    def _create_ecosystem(self) -> Ecology2DSystem:
        config = Ecology2DConfig(
            world_size=self.world_size,
            n_particles=self.n_particles,
            genome_length=24
        )
        return Ecology2DSystem(config)
    
    def _warmup(self, eco: Ecology2DSystem, steps: int = 2000):
        for _ in range(steps):
            eco.step()
    
    def _compute_spatial_distribution_entropy(self, eco: Ecology2DSystem) -> float:
        alive_particles = [p for p in eco.particles if p.alive]
        if len(alive_particles) == 0:
            return 0.0
        
        positions = np.array([p.position for p in alive_particles])
        
        grid_size = 8
        cell_size = self.world_size / grid_size
        
        counts = np.zeros((grid_size, grid_size))
        for pos in positions:
            i = min(int(pos[0] / cell_size), grid_size - 1)
            j = min(int(pos[1] / cell_size), grid_size - 1)
            counts[i, j] += 1
        
        probs = counts.flatten() / max(len(alive_particles), 1)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(grid_size * grid_size)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_genetic_diversity(self, eco: Ecology2DSystem) -> float:
        alive_particles = [p for p in eco.particles if p.alive]
        if len(alive_particles) == 0:
            return 0.0
        
        unique_genomes = len(set(p.genome.tobytes() for p in alive_particles))
        
        return unique_genomes / max(len(alive_particles), 1)
    
    def _compute_energy_efficiency(self, eco: Ecology2DSystem) -> float:
        alive_particles = [p for p in eco.particles if p.alive]
        if len(alive_particles) == 0:
            return 0.0
        
        energies = np.array([p.energy for p in alive_particles])
        
        return float(np.mean(energies))
    
    def _compute_population_stability(self, history: List[int]) -> float:
        if len(history) < 2:
            return 0.0
        
        history = np.array(history)
        mean_pop = np.mean(history)
        std_pop = np.std(history)
        
        if mean_pop == 0:
            return 0.0
        
        cv = std_pop / mean_pop
        
        return 1.0 / (1.0 + cv)
    
    def _compute_response_shift(self, before: Dict, after: Dict) -> float:
        spatial_shift = abs(after['spatial_entropy'] - before['spatial_entropy'])
        genetic_shift = abs(after['genetic_diversity'] - before['genetic_diversity'])
        energy_shift = abs(after['energy_efficiency'] - before['energy_efficiency'])
        
        return (spatial_shift + genetic_shift + energy_shift) / 3.0
    
    def test_1_implicit_pattern_extraction(self) -> Dict:
        print("\n" + "="*70)
        print("LEVEL 1: IMPLICIT PATTERN EXTRACTION")
        print("="*70)
        print("\nTests: Can the system extract statistical regularities WITHOUT explicit goals?")
        print("Inspired by: SICR (Statistically-Induced Chunking Recall)")
        print()
        
        eco = self._create_ecosystem()
        print("[Warmup] Stabilizing ecosystem (2000 steps)...")
        self._warmup(eco, 2000)
        
        alive_count = sum(1 for p in eco.particles if p.alive)
        if alive_count == 0:
            print("  ⚠ Ecosystem died during warmup!")
            return {'score': 0.0, 'status': 'extinct'}
        
        print(f"  After warmup: {alive_count} alive\n")
        
        print("[Phase 1] Expose to Pattern A (North-biased ATP) - 500 steps")
        print("  No explicit task. Just environmental statistics.")
        
        baseline_state = {
            'spatial_entropy': self._compute_spatial_distribution_entropy(eco),
            'genetic_diversity': self._compute_genetic_diversity(eco),
            'energy_efficiency': self._compute_energy_efficiency(eco)
        }
        
        for step in range(500):
            eco.fields['atp'][40:, :] *= 1.5
            eco.step()
        
        phase1_state = {
            'spatial_entropy': self._compute_spatial_distribution_entropy(eco),
            'genetic_diversity': self._compute_genetic_diversity(eco),
            'energy_efficiency': self._compute_energy_efficiency(eco)
        }
        
        adaptation_1 = self._compute_response_shift(baseline_state, phase1_state)
        print(f"  Population shift: {adaptation_1:.3f}")
        print(f"  Alive: {sum(1 for p in eco.particles if p.alive)}")
        
        print("\n[Phase 2] Switch to Pattern B (South-biased ATP) - 500 steps")
        print("  No announcement. Silent switch.")
        
        for step in range(500):
            eco.fields['atp'][:40, :] *= 1.5
            eco.step()
        
        phase2_state = {
            'spatial_entropy': self._compute_spatial_distribution_entropy(eco),
            'genetic_diversity': self._compute_genetic_diversity(eco),
            'energy_efficiency': self._compute_energy_efficiency(eco)
        }
        
        adaptation_2 = self._compute_response_shift(phase1_state, phase2_state)
        print(f"  Population shift: {adaptation_2:.3f}")
        print(f"  Alive: {sum(1 for p in eco.particles if p.alive)}")
        
        print("\n[Phase 3] Return to Pattern A - 500 steps")
        print("  Testing: Did implicit memory form?")
        
        for step in range(500):
            eco.fields['atp'][40:, :] *= 1.5
            eco.step()
        
        phase3_state = {
            'spatial_entropy': self._compute_spatial_distribution_entropy(eco),
            'genetic_diversity': self._compute_genetic_diversity(eco),
            'energy_efficiency': self._compute_energy_efficiency(eco)
        }
        
        adaptation_3 = self._compute_response_shift(phase2_state, phase3_state)
        print(f"  Population shift: {adaptation_3:.3f}")
        print(f"  Alive: {sum(1 for p in eco.particles if p.alive)}")
        
        faster_relearning = adaptation_3 < adaptation_1
        implicit_memory_score = 1.0 if faster_relearning else 0.5
        
        avg_adaptation = (adaptation_1 + adaptation_2 + adaptation_3) / 3.0
        
        total_score = (implicit_memory_score * 0.5 + avg_adaptation * 0.5)
        
        print(f"\n→ Implicit Pattern Extraction Score: {total_score:.3f}")
        print(f"  Avg Adaptation: {avg_adaptation:.3f}")
        print(f"  Faster Relearning: {'Yes' if faster_relearning else 'No'}")
        
        return {
            'score': total_score,
            'adaptations': [adaptation_1, adaptation_2, adaptation_3],
            'implicit_memory': implicit_memory_score,
            'status': 'success'
        }
    
    def test_2_emergent_system_dynamics(self) -> Dict:
        print("\n" + "="*70)
        print("LEVEL 2: EMERGENT SYSTEM DYNAMICS")
        print("="*70)
        print("\nTests: Does the system develop emergent properties beyond individuals?")
        print("Inspired by: CADI (Chaos-Aware Design Index), Order Parameters")
        print()
        
        eco = self._create_ecosystem()
        print("[Warmup] Stabilizing ecosystem (2000 steps)...")
        self._warmup(eco, 2000)
        
        if sum(1 for p in eco.particles if p.alive) == 0:
            print("  ⚠ Ecosystem died during warmup!")
            return {'score': 0.0, 'status': 'extinct'}
        
        print(f"  After warmup: {sum(1 for p in eco.particles if p.alive)} alive\n")
        
        print("[Test 2.1] Complexity Growth")
        print("  Measuring: Does genetic diversity & spatial complexity increase?")
        
        initial_genetic = self._compute_genetic_diversity(eco)
        initial_spatial = self._compute_spatial_distribution_entropy(eco)
        
        print(f"  Initial genetic diversity: {initial_genetic:.3f}")
        print(f"  Initial spatial entropy: {initial_spatial:.3f}")
        
        for _ in range(1000):
            eco.step()
        
        final_genetic = self._compute_genetic_diversity(eco)
        final_spatial = self._compute_spatial_distribution_entropy(eco)
        
        print(f"  Final genetic diversity: {final_genetic:.3f}")
        print(f"  Final spatial entropy: {final_spatial:.3f}")
        
        genetic_growth = max(0, final_genetic - initial_genetic)
        spatial_growth = max(0, final_spatial - initial_spatial)
        
        complexity_score = (genetic_growth + spatial_growth) / 2.0
        
        print(f"  Complexity growth: {complexity_score:.3f}")
        
        print("\n[Test 2.2] Population Stability")
        print("  Measuring: Self-regulation without external control")
        
        pop_history = []
        for _ in range(500):
            eco.step()
            pop_history.append(sum(1 for p in eco.particles if p.alive))
        
        stability_score = self._compute_population_stability(pop_history)
        
        print(f"  Population stability: {stability_score:.3f}")
        print(f"  Mean population: {np.mean(pop_history):.1f}")
        print(f"  Std population: {np.std(pop_history):.1f}")
        
        print("\n[Test 2.3] Resilience to Perturbation")
        print("  Measuring: Recovery after shock")
        
        pre_shock_pop = sum(1 for p in eco.particles if p.alive)
        
        kill_count = int(pre_shock_pop * 0.3)
        alive_particles = [p for p in eco.particles if p.alive]
        for i in range(min(kill_count, len(alive_particles))):
            alive_particles[i].alive = False
        
        print(f"  Killed 30% ({kill_count} particles)")
        print(f"  Population after shock: {sum(1 for p in eco.particles if p.alive)}")
        
        for _ in range(500):
            eco.step()
        
        post_recovery_pop = sum(1 for p in eco.particles if p.alive)
        recovery_rate = post_recovery_pop / max(pre_shock_pop, 1)
        
        resilience_score = min(1.0, recovery_rate)
        
        print(f"  Population after recovery: {post_recovery_pop}")
        print(f"  Recovery rate: {recovery_rate:.3f}")
        print(f"  Resilience score: {resilience_score:.3f}")
        
        total_score = (complexity_score * 0.3 + stability_score * 0.3 + resilience_score * 0.4)
        
        print(f"\n→ Emergent Dynamics Score: {total_score:.3f}")
        
        return {
            'score': total_score,
            'complexity_growth': complexity_score,
            'stability': stability_score,
            'resilience': resilience_score,
            'status': 'success'
        }
    
    def test_3_collective_intelligence(self) -> Dict:
        print("\n" + "="*70)
        print("LEVEL 3: COLLECTIVE INTELLIGENCE")
        print("="*70)
        print("\nTests: Does population exhibit distributed knowledge & coordination?")
        print("Inspired by: MetrIntMeas, Swarm Intelligence Metrics")
        print()
        
        eco = self._create_ecosystem()
        print("[Warmup] Stabilizing ecosystem (2000 steps)...")
        self._warmup(eco, 2000)
        
        if sum(1 for p in eco.particles if p.alive) == 0:
            print("  ⚠ Ecosystem died during warmup!")
            return {'score': 0.0, 'status': 'extinct'}
        
        print(f"  After warmup: {sum(1 for p in eco.particles if p.alive)} alive\n")
        
        print("[Test 3.1] Distributed Adaptation")
        print("  Measuring: Can subpopulations specialize for different niches?")
        
        eco.fields['atp'][:40, :40] *= 2.0
        eco.fields['atp'][40:, 40:] *= 2.0
        
        for _ in range(1000):
            eco.step()
        
        alive_particles = [p for p in eco.particles if p.alive]
        positions = np.array([p.position for p in alive_particles])
        
        nw_count = np.sum((positions[:, 0] < 40) & (positions[:, 1] < 40))
        se_count = np.sum((positions[:, 0] >= 40) & (positions[:, 1] >= 40))
        
        total = nw_count + se_count
        specialization = (nw_count + se_count) / max(sum(1 for p in eco.particles if p.alive), 1)
        
        print(f"  NW quadrant: {nw_count} particles")
        print(f"  SE quadrant: {se_count} particles")
        print(f"  Specialization: {specialization:.3f}")
        
        print("\n[Test 3.2] Collective Response Coordination")
        print("  Measuring: Synchronized population-level response")
        
        initial_energy = self._compute_energy_efficiency(eco)
        
        eco.fields['atp'] *= 0.5
        
        energy_history = []
        for _ in range(500):
            eco.step()
            energy_history.append(self._compute_energy_efficiency(eco))
        
        energy_recovery = (energy_history[-1] - energy_history[0]) / max(initial_energy, 0.01)
        coordination_score = min(1.0, max(0, energy_recovery))
        
        print(f"  Initial avg energy: {initial_energy:.3f}")
        print(f"  Final avg energy: {energy_history[-1]:.3f}")
        print(f"  Coordination score: {coordination_score:.3f}")
        
        print("\n[Test 3.3] Information Distribution")
        print("  Measuring: Is knowledge distributed across population?")
        
        genetic_diversity = self._compute_genetic_diversity(eco)
        spatial_entropy = self._compute_spatial_distribution_entropy(eco)
        
        information_distribution = (genetic_diversity + spatial_entropy) / 2.0
        
        print(f"  Genetic diversity: {genetic_diversity:.3f}")
        print(f"  Spatial entropy: {spatial_entropy:.3f}")
        print(f"  Information distribution: {information_distribution:.3f}")
        
        total_score = (specialization * 0.3 + coordination_score * 0.3 + information_distribution * 0.4)
        
        print(f"\n→ Collective Intelligence Score: {total_score:.3f}")
        
        return {
            'score': total_score,
            'specialization': specialization,
            'coordination': coordination_score,
            'information_distribution': information_distribution,
            'status': 'success'
        }
    
    def run_full_suite(self) -> Dict:
        print("="*70)
        print("WORLD FIRST: UNCONSCIOUS LEARNING TEST SUITE")
        print("="*70)
        print()
        print("Testing IMPLICIT, EMERGENT, DISTRIBUTED learning")
        print("Not testing EXPLICIT, TASK-ORIENTED, GOAL-DIRECTED learning")
        print()
        
        start_time = datetime.now()
        
        results = {}
        
        results['level_1'] = self.test_1_implicit_pattern_extraction()
        
        results['level_2'] = self.test_2_emergent_system_dynamics()
        
        results['level_3'] = self.test_3_collective_intelligence()
        
        duration = (datetime.now() - start_time).total_seconds() / 60.0
        
        level1_score = results['level_1']['score']
        level2_score = results['level_2']['score']
        level3_score = results['level_3']['score']
        
        overall_score = (level1_score * 0.3 + level2_score * 0.3 + level3_score * 0.4) * 100
        
        if overall_score >= 80:
            grade = 'A - Excellent'
        elif overall_score >= 70:
            grade = 'B - Good'
        elif overall_score >= 60:
            grade = 'C - Fair'
        elif overall_score >= 50:
            grade = 'D - Poor'
        else:
            grade = 'F - Failed'
        
        print("\n" + "="*70)
        print("UNCONSCIOUS LEARNING FINAL REPORT")
        print("="*70)
        print()
        print(f"Duration: {duration:.1f} minutes")
        print()
        print("SCORES:")
        print(f"  Level 1 (Implicit Pattern Extraction): {level1_score:.3f}")
        print(f"  Level 2 (Emergent Dynamics):           {level2_score:.3f}")
        print(f"  Level 3 (Collective Intelligence):     {level3_score:.3f}")
        print()
        print(f"Overall Score: {overall_score:.1f} / 100")
        print(f"Grade: {grade}")
        print()
        
        if overall_score >= 70:
            print("✅ System demonstrates UNCONSCIOUS LEARNING")
            print("   - Implicit adaptation to statistical patterns")
            print("   - Emergent system-level properties")
            print("   - Distributed collective intelligence")
        elif overall_score >= 50:
            print("⚠ System shows PARTIAL unconscious learning")
            print("   - Some emergent properties detected")
            print("   - Collective effects present but weak")
        else:
            print("❌ System does NOT demonstrate unconscious learning")
            print("   - Mainly reactive behavior")
            print("   - Limited emergence")
        
        print()
        print("="*70)
        
        results['overall'] = {
            'score': overall_score,
            'grade': grade,
            'duration_minutes': duration
        }
        
        self.results = results
        
        return results
    
    def save_results(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved: {filepath}")


if __name__ == '__main__':
    suite = UnconsciousLearningTestSuite()
    
    results = suite.run_full_suite()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'benchmark_results/unconscious_{timestamp}.json'
    suite.save_results(filepath)
    
    print("\n✅ Unconscious learning test suite complete!")
