"""
LEGACY gene expression — DEPRECATED as of 2026-04-17.

This module preserves the original "12 hand-crafted formulas" gene→phenotype
mapping that DAERWEN used before switching to the uniform composition-based
chem_sim-inspired mapping.

Why deprecated:
  - Uses 12 different mathematical transforms (tanh, sin, sigmoid, std, mean,
    modulo, linear) picked by the designer without principled justification.
  - Accidentally produces extreme phenotype distributions that cause
    boom-or-bust dynamics (σ of final population ~373 vs ~225 for new mapping).
  - Violates the project's "minimize the Designer's Trap" commitment.

Why kept:
  - Historical reference for the migration.
  - If a specific dynamic (e.g., occasional population bloom) is ever needed
    for a comparative experiment, this module can be imported explicitly.
  - Documentation of what a high-variance gene-expression system looks like.

Do not import this from production code. It is not wired into any default
code path as of 2026-04-17.
"""
from __future__ import annotations
from typing import Dict
import numpy as np


def express_phenotypes_legacy(genome: np.ndarray, system_ref=None) -> Dict[str, float]:
    """Reproduces the pre-2026-04-17 gene expression with 12 distinct formulas.

    Kept for comparative experiments only.
    """
    genome_len = len(genome)
    seg_len = max(1, genome_len // 8)

    # 段0: 场交互强度 (-1到1，负=释放，正=吸收)
    seg0 = genome[:seg_len]
    field_interaction = float(np.tanh(np.sum(seg0) / len(seg0) - 1.5))

    # 段1: 移动敏感度 (0-2)
    seg1 = genome[seg_len:2 * seg_len]
    movement_response = float(np.abs(np.std(seg1)) * 2)

    # 段2: 交互基线倾向
    seg2 = genome[2 * seg_len:3 * seg_len]
    interaction_mode = float(np.sin(np.sum(seg2)))

    # 段3: 复制阈值 (0.5-4)
    seg3 = genome[3 * seg_len:4 * seg_len]
    replication_threshold = 0.5 + float(np.mean(seg3) * 1.5)

    # 段4: 场转化阈值 (0-1)
    seg4 = genome[4 * seg_len:5 * seg_len]
    conversion_threshold = float((np.sum(seg4) % 10) / 10.0)

    # 段5: 交互强度阈值
    seg5 = genome[5 * seg_len:6 * seg_len]
    interaction_threshold = float(np.mean(seg5) / 4.0)

    # 段6: 合作倾向阈值
    seg6 = genome[6 * seg_len:7 * seg_len]
    sig_input = np.sum(seg6) / len(seg6) - 1.5
    cooperation_threshold = float(1.0 / (1.0 + np.exp(-sig_input)))

    # 段7: 衰老抗性
    seg7 = genome[7 * seg_len:8 * seg_len]
    aging_resistance = float(np.mean(seg7) / 2.0) if len(seg7) > 0 else 0.5

    # 扩展段（第三层）
    _EXT_SEG = 4
    _ext_base = 8 * seg_len

    def _decode_ext(n):
        start = _ext_base + n * _EXT_SEG
        end = start + _EXT_SEG
        if end <= genome_len:
            return genome[start:end]
        return None

    _s8 = _decode_ext(0)
    inhibitor_sensitivity = (float(np.mean(_s8)) / 30.0) if _s8 is not None else (
        system_ref.config.inhibitor_damage_coeff if system_ref else 0.02)

    _s9 = _decode_ext(1)
    chemotaxis_gene_strength = (float(np.mean(_s9)) / 20.0) if _s9 is not None else (
        system_ref.config.chemotaxis_strength if system_ref else 0.05)

    _s10 = _decode_ext(2)
    replication_energy_split = (0.3 + float(np.mean(_s10)) * 0.4 / 3.0) if _s10 is not None else 0.5

    _s11 = _decode_ext(3)
    atp_absorption_rate = (0.05 + float(np.mean(_s11)) * 0.2 / 3.0) if _s11 is not None else (
        system_ref.config.atp_absorption_scale if system_ref else 0.1)

    return {
        'field_interaction':         field_interaction,
        'movement_response':         movement_response,
        'interaction_mode':          interaction_mode,
        'replication_threshold':     replication_threshold,
        'conversion_threshold':      conversion_threshold,
        'interaction_threshold':     interaction_threshold,
        'cooperation_threshold':     cooperation_threshold,
        'aging_resistance':          aging_resistance,
        'inhibitor_sensitivity':     inhibitor_sensitivity,
        'chemotaxis_gene_strength':  chemotaxis_gene_strength,
        'replication_energy_split':  replication_energy_split,
        'atp_absorption_rate':       atp_absorption_rate,
    }
