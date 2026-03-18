"""
DAERWEN3 - 2D自持生态网络系统
基于物理-化学-基因整合的纯粹涌现AGI实验平台

核心理念：
- 使用2D空间（高效计算）
- 只定义最小物理规则
- 让生态关系自发涌现
- 无预设的捕食/共生等高层概念
"""

__version__ = "2.0.0"
__author__ = "DAERWEN3 Team"

from .core import (
    Ecology2DSystem,
    Ecology2DConfig,
    Particle2D,
    ChemicalField2D,
    ExternalInput,
    SystemOutput,
)

__all__ = [
    'Ecology2DSystem',
    'Ecology2DConfig',
    'Particle2D',
    'ChemicalField2D',
    'ExternalInput',
    'SystemOutput',
]

