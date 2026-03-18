#!/usr/bin/env python3
"""快速验证恢复的参数配置"""

import sys
import os
sys.path.insert(0, '/mnt/f/avalanche-持续学习/daerwen3.5')

from engine.core import Ecology2DSystem, Ecology2DConfig

def quick_verify():
    print("=" * 70)
    print("快速参数验证测试")
    print("=" * 70)
    
    config = Ecology2DConfig(world_size=80, n_particles=300)
    eco = Ecology2DSystem(config)
    
    print(f"\n✓ 系统初始化成功")
    print(f"   世界大小: {eco.config.world_size}×{eco.config.world_size}")
    print(f"   初始粒子: {len([p for p in eco.particles if p.alive])}")
    
    print(f"\n✓ 关键参数验证:")
    print(f"   mutation_rate: {eco.config.mutation_rate} (期望: 0.01)")
    print(f"   solar_energy_rate: {eco.config.solar_energy_rate} (期望: 1.0)")
    print(f"   metabolic_cost: {eco.config.metabolic_cost} (期望: 0.01)")
    
    assert eco.config.mutation_rate == 0.01, f"mutation_rate错误: {eco.config.mutation_rate}"
    assert eco.config.solar_energy_rate == 1.0, f"solar_energy_rate错误: {eco.config.solar_energy_rate}"
    assert eco.config.metabolic_cost == 0.01, f"metabolic_cost错误: {eco.config.metabolic_cost}"
    
    print(f"\n✓ 参数验证通过！\n")
    
    print("=" * 70)
    print("运行5000步稳定性测试...")
    print("=" * 70)
    
    for step in range(5000):
        eco.step()
        if step % 1000 == 0:
            alive = sum(1 for p in eco.particles if p.alive)
            print(f"  Step {step:5d}: {alive:4d} alive")
    
    final_alive = sum(1 for p in eco.particles if p.alive)
    print(f"\n✓ 最终存活: {final_alive}")
    
    if final_alive > 0:
        print("✅ 验证成功！生态系统稳定运行")
        return True
    else:
        print("❌ 警告：种群全部死亡")
        return False

if __name__ == '__main__':
    success = quick_verify()
    sys.exit(0 if success else 1)
