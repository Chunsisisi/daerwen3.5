#!/usr/bin/env python3
"""
DAERWEN3 2D生态网络 - 启动脚本
一键启动2D自持生态系统的实时可视化服务器
"""
import sys
import os
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.core import Ecology2DSystem, Ecology2DConfig
from engine.server import Ecology2DServer

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║           🌱 DAERWEN3 2D生态网络系统 v2.0               ║
    ║                                                          ║
    ║   基于纯粹涌现的自持生态AGI实验平台                      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    启动步骤：
    1. 服务器将在 ws://localhost:8765 启动
    2. 打开浏览器访问 engine/web.html
    3. 使用“外部输入”面板注入化学脉冲/调节参数/触发灾难
    4. 观察“最近输入”和系统输出的即时反馈
    5. 等待涌现指标达到 > 0.5（可能需要数万步）
    
    按 Ctrl+C 停止服务器
    """)
    
    try:
        # 配置
        config = Ecology2DConfig(
            world_size=200,
            n_particles=3000,
            genome_length=32,
            mutation_rate=0.01,
            n_chemical_species=10,
        )
        
        # 创建服务器
        server = Ecology2DServer(config, run_mode='demo')
        
        # 启动
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n\n✅ 服务器已安全停止")
        sys.exit(0)
