"""
(third step) Does the LEARNING MECHANISM itself emerge from generic chemistry?

`chemical_learning`: each particle has an internal molecule `mem` evolving by a
generic bilinear mass-action reaction with 5 gene-encoded, randomly-initialized
coefficients: d_mem = a·(cue·rew) + b·cue + c·rew − d·mem; mem drives cue-following
via evolved coupling e. Nothing is hand-set to be "coincidence detection" — only a
generic reaction whose coefficients evolve.

Behavioral proof: in a world where the reward patch MOVES (faster than generations),
only within-lifetime learning to follow the cue can track it. If the
chemical-learning population TRACKS the moving reward (high occupancy near it) while
a no-learning control (rich sensing but no internal chemistry) cannot, then the
evolved chemistry implements functional associative learning. And if the reaction
machinery is functional in the moving world but assimilated/decayed in a STATIC one,
the mechanism is demand-driven (Baldwin), not generic drift.

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/emergent_learning_chem.py
"""
import os, sys, math
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from engine.core import Ecology2DConfig, Ecology2DSystem
from association_experiment import bump, flat, cull_to_cap

CUE_CH, NUT_CH, ATP_CH, WORLD, PERIOD = 5, 1, 0, 50, 300
PROBE = [0, 500, 1000, 1500]


def reward_peak(t, moving):
    return (WORLD / 2 + (WORLD / 3) * math.sin(2 * math.pi * t / PERIOD)) if moving else WORLD / 2


def occ_near(s, peak):
    a = [p for p in s.particles if p.alive]
    if not a:
        return None
    xs = np.array([p.position[0] for p in a])
    dd = np.abs(xs - peak); dd = np.minimum(dd, WORLD - dd)
    return float(np.mean(dd < WORLD / 6))


def mean_ae(s):
    a = [p for p in s.particles if p.alive and p.rxn_genes is not None]
    if not a:
        return (None, None)
    g = np.mean([p.rxn_genes for p in a], axis=0)
    return float(g[0]), float(g[4])   # a (coincidence), e (coupling)


def run(moving, learning, seed, steps=1500):
    cfg = Ecology2DConfig(world_size=WORLD, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=800, orthogonal_expression=True,
                          rich_channels=True, cue_channels=(CUE_CH,),
                          chemical_learning=(learning == 'chem'))
    s = Ecology2DSystem(cfg)
    for _ in range(150):
        s.step()
    traj = {}
    for t in range(steps + 1):
        peak = reward_peak(t, moving)
        flat(s, ATP_CH, 0.04)
        bump(s, NUT_CH, peak, hi=1.6, lo=0.0)
        bump(s, CUE_CH, peak, hi=1.0, lo=0.02)       # cue marks the (moving) reward
        s.step(); cull_to_cap(s, 800)
        if t in PROBE:
            traj[t] = (occ_near(s, peak), mean_ae(s), sum(p.alive for p in s.particles))
        if sum(p.alive for p in s.particles) == 0:
            break
    return traj


def summarize(label, moving, learning, seeds):
    occ = {t: [] for t in PROBE}; A = {t: [] for t in PROBE}; E = {t: [] for t in PROBE}
    for seed in seeds:
        tr = run(moving, learning, seed)
        for t in PROBE:
            if t in tr and tr[t][0] is not None:
                occ[t].append(tr[t][0])
                if tr[t][1][0] is not None:
                    A[t].append(tr[t][1][0]); E[t].append(tr[t][1][1])
    print(f"-- {label} --")
    hdr = f"  {'step':>5} | {'reward-tracking occ':>19}"
    if learning == 'chem':
        hdr += f" | {'evolved a':>9} | {'evolved e':>9}"
    print(hdr)
    for t in PROBE:
        if occ[t]:
            line = f"  {t:>5} | {np.median(occ[t]):>19.3f}"
            if learning == 'chem' and A[t]:
                line += f" | {np.median(A[t]):>9.3f} | {np.median(E[t]):>9.3f}"
            print(line)
    print()
    return {t: (np.median(occ[t]) if occ[t] else None) for t in PROBE}


def main(seeds=range(4)):
    seeds = list(seeds)
    print(f"EMERGENT CHEMICAL LEARNING  seeds={seeds}  (chance occ ≈ {2*(WORLD/6)/WORLD:.2f})\n")
    a = summarize("MOVING reward, chemical_learning", True, 'chem', seeds)
    b = summarize("MOVING reward, NO learning (control)", True, 'none', seeds)
    c = summarize("STATIC reward, chemical_learning", False, 'chem', seeds)
    print("READING:")
    print(f"  moving+chem final occ={a[PROBE[-1]]}, moving+no-learning={b[PROBE[-1]]}")
    if a[PROBE[-1]] and b[PROBE[-1]] and a[PROBE[-1]] > b[PROBE[-1]] + 0.08:
        print("  => evolved internal chemistry TRACKS the moving reward better than a")
        print("     non-learning control: associative learning emerged from generic chemistry.")
    else:
        print("  => chemistry does not beat the non-learning control: mechanism did not emerge.")


if __name__ == "__main__":
    main()
