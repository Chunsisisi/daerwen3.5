"""
(b) Can LEARNING ITSELF emerge from selection, instead of being hand-designed?

Finding 13 got associative recall by hand-coding a plasticity rule with a fixed
learning rate — an added prior. Here we do NOT set the learning rate. It is a
heritable, mutating gene (`evolvable_plasticity`), randomly initialized across the
population (negative / zero / positive). We put the population in a world that
changes FASTER than generations — a reward patch that oscillates in space — so a
static genetic preference cannot track it; only *learning within a lifetime to
follow the cue that marks the reward* can. If the learning-rate gene evolves UP in a
learnable world but NOT in an unlearnable one, then selection itself discovered that
learning is worth having — learning emerges, rather than being engineered in. (This
is why learning evolved in biology: environments varying faster than generations.)

Conditions (moving reward patch, ATP starvation so the patch is survival-critical):
  learnable : cue channel 5 tracks the moving reward  -> following the cue pays off
  noise     : cue channel 5 is random noise, reward unmarked -> nothing to learn
  static    : reward patch fixed (a genetic preference could also solve it)

Measure: population-mean plasticity_gene over evolutionary time. Emergence signature:
rises in 'learnable', stays low/flat in 'noise'.

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/evolution_of_learning.py
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

CUE_CH = 5
NUT_CH = 1
ATP_CH = 0
WORLD = 50
PERIOD = 300          # reward oscillation period (< a few generations => faster than evolution)
PROBE = [0, 600, 1200, 1800, 2400]


def reward_peak(t):
    return WORLD / 2 + (WORLD / 3) * math.sin(2 * math.pi * t / PERIOD)


def mean_plast_gene(s):
    a = [p for p in s.particles if p.alive]
    return float(np.mean([p.plasticity_gene for p in a])) if a else None


def learned_cue_w(s):
    ci = s.inert_channels.index(CUE_CH)
    a = [p for p in s.particles if p.alive and p.plastic_cue_w is not None]
    return float(np.mean([p.plastic_cue_w[ci] for p in a])) if a else None


def run(cond, seed, steps=2400):
    cfg = Ecology2DConfig(world_size=WORLD, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=800, orthogonal_expression=True,
                          rich_channels=True, cue_channels=(CUE_CH,),
                          lifetime_plasticity=True, evolvable_plasticity=True)
    s = Ecology2DSystem(cfg)
    for _ in range(150):
        s.step()
    traj = {}
    for t in range(steps + 1):
        peak = reward_peak(t)
        flat(s, ATP_CH, 0.04)                        # starvation
        bump(s, NUT_CH, peak, hi=1.6, lo=0.0)        # moving reward patch
        if cond == 'learnable':
            bump(s, CUE_CH, peak, hi=1.0, lo=0.02)   # cue tracks reward
        elif cond == 'static':
            bump(s, NUT_CH, WORLD / 2, hi=1.6, lo=0.0)   # override: fixed reward
            bump(s, CUE_CH, WORLD / 2, hi=1.0, lo=0.02)  # fixed cue
        elif cond == 'noise':
            rand = s.rng.uniform(0.0, 1.0, size=(WORLD, WORLD)).astype('float32')
            s.chemical_field.concentrations[:, :, CUE_CH] = rand   # uninformative cue
            s.chemical_field.invalidate_gradient_cache()
        s.step()
        cull_to_cap(s, 800)
        if t in PROBE:
            traj[t] = (mean_plast_gene(s), learned_cue_w(s), sum(p.alive for p in s.particles))
        if sum(p.alive for p in s.particles) == 0:
            break
    return traj


def main(seeds=range(4)):
    seeds = list(seeds)
    conds = ['learnable', 'noise', 'static']
    print(f"EVOLUTION OF LEARNING  seeds={seeds}  (does the learning-rate gene evolve up?)")
    print("plasticity_gene starts ~0 (random). Rises only if selection finds learning useful.\n")
    for cond in conds:
        rows = {t: [] for t in PROBE}
        cuew = {t: [] for t in PROBE}
        for seed in seeds:
            tr = run(cond, seed)
            for t in PROBE:
                if t in tr and tr[t][0] is not None:
                    rows[t].append(tr[t][0]); cuew[t].append(tr[t][1])
        print(f"-- {cond} --")
        print(f"  {'step':>5} | {'mean plasticity_gene':>20} | {'learned cue-w':>13}")
        for t in PROBE:
            if rows[t]:
                print(f"  {t:>5} | {np.median(rows[t]):>20.3f} | {np.median(cuew[t]):>13.3f}")
        print()

    print("READING: if plasticity_gene climbs in 'learnable' but stays low in 'noise',")
    print("selection itself discovered learning — the plasticity rule's usefulness is")
    print("emergent, not just its hand-set rate. (The rule TEMPLATE is still designed.)")


if __name__ == "__main__":
    main()
