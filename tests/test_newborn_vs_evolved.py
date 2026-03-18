"""
测试"新生儿"（初始随机基因组）vs "进化后"的性能对比

这个测试回答：进化到底贡献了多少？
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.insert(0, '/mnt/f/avalanche-持续学习/daerwen3.5')

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig

def create_center_gradient(system):
    world = system.config.world_size
    x = np.arange(world)
    y = np.arange(world)
    xx, yy = np.meshgrid(x, y)
    center = world // 2
    dist = np.sqrt((xx - center)**2 + (yy - center)**2)
    gradient = 2.0 * np.exp(-dist**2 / (2 * (world/4)**2))
    system.chemical_field.concentrations[:, :, system.chemical_field.ATP_index] = gradient

def calculate_center_alignment(particles, world_size):
    alive = [p for p in particles if p.alive]
    if not alive:
        return 0.0
    positions = np.array([p.position for p in alive])
    cx, cy = np.mean(positions[:, 0]), np.mean(positions[:, 1])
    center = world_size / 2
    dist = np.sqrt((cx - center)**2 + (cy - center)**2)
    score = 1.0 - (dist / (world_size * np.sqrt(2) / 2))
    return float(np.clip(score, 0.0, 1.0))

def test_newborn():
    """测试新生儿（无进化）"""
    print("\n" + "="*70)
    print("TEST 1: NEWBORN (No evolution, random genomes)")
    print("="*70)
    
    config = Ecology2DConfig(
        world_size=80,
        n_particles=300,
        genome_length=24,
        mutation_rate=0.0,
    )
    
    system = Ecology2DSystem(config)
    
    print("\n[Warmup] 500 steps...")
    for _ in range(500):
        system.step()
    
    print("[Task] Center gradient, 250 steps...")
    create_center_gradient(system)
    
    scores = []
    for step in range(250):
        system.step()
        if step % 50 == 49:
            score = calculate_center_alignment(system.particles, system.config.world_size)
            alive = sum(1 for p in system.particles if p.alive)
            scores.append(score)
            print(f"  Step {step+1}: {alive} alive, score {score:.3f}")
    
    final_score = calculate_center_alignment(system.particles, system.config.world_size)
    print(f"\n→ Final score: {final_score:.3f}")
    return final_score, scores

def test_evolved():
    """测试进化后（有mutation）"""
    print("\n" + "="*70)
    print("TEST 2: EVOLVED (With mutation and selection)")
    print("="*70)
    
    config = Ecology2DConfig(
        world_size=80,
        n_particles=300,
        genome_length=24,
        mutation_rate=0.03,
    )
    
    system = Ecology2DSystem(config)
    
    print("\n[Warmup] 500 steps...")
    for _ in range(500):
        system.step()
    
    print("[Task] Center gradient, 250 steps...")
    create_center_gradient(system)
    
    scores = []
    for step in range(250):
        system.step()
        if step % 50 == 49:
            score = calculate_center_alignment(system.particles, system.config.world_size)
            alive = sum(1 for p in system.particles if p.alive)
            scores.append(score)
            print(f"  Step {step+1}: {alive} alive, score {score:.3f}")
    
    final_score = calculate_center_alignment(system.particles, system.config.world_size)
    print(f"\n→ Final score: {final_score:.3f}")
    return final_score, scores

def test_evolved_longer():
    """测试长期进化（2000步训练）"""
    print("\n" + "="*70)
    print("TEST 3: EVOLVED LONGER (2000 steps training)")
    print("="*70)
    
    config = Ecology2DConfig(
        world_size=80,
        n_particles=300,
        genome_length=24,
        mutation_rate=0.03,
    )
    
    system = Ecology2DSystem(config)
    
    print("\n[Warmup] 500 steps...")
    for _ in range(500):
        system.step()
    
    print("[Training] Center gradient, 2000 steps...")
    create_center_gradient(system)
    for step in range(2000):
        system.step()
        if step % 500 == 499:
            score = calculate_center_alignment(system.particles, system.config.world_size)
            alive = sum(1 for p in system.particles if p.alive)
            print(f"  Step {step+1}: {alive} alive, score {score:.3f}")
    
    final_score = calculate_center_alignment(system.particles, system.config.world_size)
    alive = sum(1 for p in system.particles if p.alive)
    print(f"\n→ After training: {alive} alive, score {final_score:.3f}")
    return final_score

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NEWBORN vs EVOLVED COMPARISON TEST")
    print("="*70)
    print("\nQuestion: How much does evolution contribute?")
    print("="*70)
    
    newborn_score, newborn_curve = test_newborn()
    evolved_score, evolved_curve = test_evolved()
    evolved_long_score = test_evolved_longer()
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\nNewborn (no evolution):       {newborn_score:.3f}")
    print(f"Evolved (250 steps):          {evolved_score:.3f}")
    print(f"Evolved (2000 steps training):{evolved_long_score:.3f}")
    
    improvement_short = (evolved_score - newborn_score) / newborn_score if newborn_score > 0 else 0
    improvement_long = (evolved_long_score - newborn_score) / newborn_score if newborn_score > 0 else 0
    
    print(f"\nImprovement (short): {improvement_short*100:+.1f}%")
    print(f"Improvement (long):  {improvement_long*100:+.1f}%")
    
    if improvement_long < 0.05:
        print("\n⚠️  CRITICAL: Evolution contributes < 5%!")
        print("   Most 'intelligence' comes from physics, not learning.")
    elif improvement_long < 0.2:
        print("\n⚠️  WARNING: Evolution contributes 5-20%")
        print("   Modest evolutionary benefit.")
    else:
        print("\n✓ Evolution contributes > 20%")
        print("  Significant learning effect demonstrated.")
    
    print("="*70)
