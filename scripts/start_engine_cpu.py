"""
DAERWEN3 2D生态网络系统启动器 - CPU版本
强制使用NumPy，不使用GPU加速
"""
import sys
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import multiprocessing
n_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_MAX_THREADS'] = str(n_cores)

import asyncio

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.server import Ecology2DServer
from engine.core import Ecology2DConfig

def main():
    print("\n" + "="*60)
    print(f"⚠️  强制使用CPU模式（NumPy） - {n_cores} 核心")
    print("="*60 + "\n")
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║           🌱 DAERWEN3 2D生态网络系统 v2.0               ║
    ║              (CPU优化版 - 适合长期演化)                  ║
    ║                                                          ║
    ║   基于纯粹涌现的自持生态AGI实验平台                      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    启动步骤：
    1. 服务器将在 ws://localhost:8765 启动
    2. 打开浏览器访问 engine/web.html
    3. 观察生态网络的实时演化
    4. 真正的涌现需要 >1000 步才可能出现
    
    """)
    
    # CPU优化配置：平衡性能和速度
    config = Ecology2DConfig(
        world_size=150,           # 世界大小
        n_particles=3000,         # 粒子数量
        genome_length=24,         # 基因长度
        mutation_rate=0.02,       # 变异率
        n_chemical_species=5,     # 化学物质种类
    )
    
    print(f"✅ 2D生态系统初始化完成")
    print(f"   世界大小: {config.world_size}×{config.world_size}")
    print(f"   初始粒子: {config.n_particles}")
    print(f"   化学物质种类: {config.n_chemical_species}\n")
    
    # 创建服务器
    server = Ecology2DServer(config, run_mode='demo')
    
    # 启动
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n\n✅ 服务器已安全停止")

if __name__ == "__main__":
    main()
