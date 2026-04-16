"""
Quick smoke test: does the chem_sim-style gene expression keep the system alive?

Compares old (12-formula) vs new (composition-based) gene expression on:
  - Population survival
  - Replication events
  - Phenotype distribution sanity
"""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only for fair compare

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig


def run_sim(use_chem_sim_genes: bool, steps: int = 1500, seed: int = 42) -> dict:
    cfg = Ecology2DConfig(
        world_size=80,
        n_particles=300,
        genome_length=48,
        mutation_rate=0.01,
        use_chem_sim_genes=use_chem_sim_genes,
    )
    system = Ecology2DSystem(cfg)
    system.rng = np.random.default_rng(seed)
    for _ in range(steps):
        system.step()

    alive = [p for p in system.particles if p.alive]
    if not alive:
        return {'alive': 0, 'avg_energy': 0.0, 'replications': int(system.stats['replication_events']),
                'deaths': int(system.stats['death_events']),
                'max_gen': 0, 'phenotype_stats': {}}

    # Phenotype diversity check
    pheno_keys = ['replication_threshold', 'field_interaction', 'movement_response', 'aging_resistance']
    pheno_stats = {}
    for k in pheno_keys:
        vals = [p.phenotype.get(k, 0.0) for p in alive]
        pheno_stats[k] = (float(np.mean(vals)), float(np.std(vals)))

    return {
        'alive': len(alive),
        'avg_energy': float(np.mean([p.energy for p in alive])),
        'replications': int(system.stats['replication_events']),
        'deaths': int(system.stats['death_events']),
        'max_gen': max(p.generation for p in alive),
        'phenotype_stats': pheno_stats,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("OLD MODE (12 different formulas)")
    print("=" * 60)
    old = run_sim(use_chem_sim_genes=False, steps=1500, seed=42)
    print(f"  alive: {old['alive']}  avg_energy: {old['avg_energy']:.3f}")
    print(f"  replications: {old['replications']}  deaths: {old['deaths']}")
    print(f"  max generation: {old['max_gen']}")
    print(f"  phenotype mean±std:")
    for k, (m, s) in old['phenotype_stats'].items():
        print(f"    {k:<24}: {m:+.3f} ± {s:.3f}")

    print()
    print("=" * 60)
    print("NEW MODE (chem_sim composition mapping)")
    print("=" * 60)
    new = run_sim(use_chem_sim_genes=True, steps=1500, seed=42)
    print(f"  alive: {new['alive']}  avg_energy: {new['avg_energy']:.3f}")
    print(f"  replications: {new['replications']}  deaths: {new['deaths']}")
    print(f"  max generation: {new['max_gen']}")
    print(f"  phenotype mean±std:")
    for k, (m, s) in new['phenotype_stats'].items():
        print(f"    {k:<24}: {m:+.3f} ± {s:.3f}")

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if new['alive'] == 0:
        print("  ❌ NEW MODE caused extinction — chemistry mapping incompatible")
    elif new['alive'] < old['alive'] * 0.3:
        print("  ⚠  NEW MODE has much lower survival than OLD")
    elif new['replications'] == 0 and old['replications'] > 0:
        print("  ⚠  NEW MODE doesn't reproduce — phenotype thresholds wrong")
    else:
        ratio_alive = new['alive'] / max(old['alive'], 1)
        ratio_rep = new['replications'] / max(old['replications'], 1)
        print(f"  ✅ NEW MODE viable. Alive ratio: {ratio_alive:.2f}, Rep ratio: {ratio_rep:.2f}")
