"""
Does the root-cause fix (rich_channels) unlock ASSOCIATIVE learning that failed in
Finding 6?

Finding 6: even with an evolvable single-signal sense, the population never learned
to follow an arbitrary cue that predicts reward (no sign flip). Root cause: too few
causally-active channels. rich_channels gives every inert channel an evolvable
sense+metabolism gene. Re-run the sign-flip test with it ON.

Protocol (per condition, per seed, rich_channels=ON):
  cue        = channel 5 (an inert channel), spatial bump HIGH on the LEFT, every step
  ATP        = flat 0.03 (near-starvation: reaching the reward is survival-critical)
  reward     = nutrient bump, on the LEFT (correlated) or RIGHT (anti-correlated)
  measure    = evolved mean channel_genes[cue] after training

Genuine association = SIGN FLIP: channel_genes[cue] systematically higher when the
reward tracks the cue (left) than when it is anti-correlated (right). The sign-flip
control is robust to the confound that the cue is itself mildly nutritious under
rich_channels: if the gene just goes positive because the cue is food, it goes
positive in BOTH conditions -> no flip -> not counted as association.

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_association_rich.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig
from association_experiment import bump, flat, cull_to_cap, occupancy

CUE_CHANNEL = 5
NUTRIENT_CH = 1
ATP_CH = 0


def mean_cue_gene(system, cue_idx):
    alive = [p for p in system.particles if p.alive and p.channel_genes is not None]
    if not alive:
        return None
    return float(np.mean([p.channel_genes[cue_idx] for p in alive]))


def run(cond, seed, steps=1500):
    reward_left = cond != 'assoc_right'
    cfg = Ecology2DConfig(world_size=50, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=1200, orthogonal_expression=True,
                          rich_channels=True, cue_channels=(CUE_CHANNEL,))  # pure cue: sensable, not food
    cfg.brownian_strength = 0.05
    s = Ecology2DSystem(cfg)
    cue_idx = s.inert_channels.index(CUE_CHANNEL)
    for _ in range(200):
        s.step()
    g0 = mean_cue_gene(s, cue_idx)
    reward_peak = 0 if reward_left else s.config.world_size // 2
    for _ in range(steps):
        bump(s, CUE_CHANNEL, 0, hi=1.0, lo=0.02)     # cue always high-left
        flat(s, ATP_CH, 0.03)                        # starvation
        bump(s, NUTRIENT_CH, reward_peak, hi=1.6, lo=0.0)  # reward
        s.step()
        cull_to_cap(s, 1200)
        if sum(p.alive for p in s.particles) == 0:
            break
    return g0, mean_cue_gene(s, cue_idx), occupancy(s, reward_peak)


def main(seeds=range(6)):
    seeds = list(seeds)
    print(f"ASSOCIATION under rich_channels  cue=ch{CUE_CHANNEL}  seeds={seeds}")
    print("sign flip of channel_genes[cue] with reward location = genuine association\n")
    L, R = [], []
    for seed in seeds:
        gl0, gl1, lo = run('assoc_left', seed)
        gr0, gr1, ro = run('assoc_right', seed)
        L.append(gl1); R.append(gr1)
        print(f"  seed{seed}: left cue_gene->{gl1:+.3f} (rew_occ {lo:.2f})   "
              f"right cue_gene->{gr1:+.3f} (rew_occ {ro:.2f})   L-R={gl1-gr1:+.3f}")
    d = np.array(L) - np.array(R)
    print(f"\nmedian(left - right) = {np.median(d):+.3f}  "
          f"(consistently > 0 = association; ~0 or mixed = none)")
    n_pos = int(np.sum(d > 0.05))
    print(f"seeds with left>right by >0.05: {n_pos}/{len(seeds)}")
    if np.median(d) > 0.1 and n_pos >= len(seeds) * 0.7:
        print("=> SIGN FLIP: rich_channels UNLOCKS associative learning — the cue gene")
        print("   tracks the reward contingency. This is the recall half Finding 6 lacked.")
    else:
        print("=> no consistent sign flip: association still does not emerge even with")
        print("   rich_channels (capacity improved, but cue->reward association did not).")


if __name__ == "__main__":
    main()
