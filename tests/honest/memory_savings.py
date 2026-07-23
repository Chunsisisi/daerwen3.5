"""
Task 2: the RECALL half — test for "savings on relearning" (latent memory).

Finding 6 showed the substrate does not learn associative (cue->memory) recall.
Here we test a different, classic memory phenomenon that IS a form of latent
recall: SAVINGS. Even after a memory has decayed to un-decodable, re-exposure may
recover the original genetic state FASTER than a naive first exposure — meaning the
trace was not truly gone, just below the readout threshold.

Protocol (per seed, target environment A):
  target S_A  = mean heritable base-freq state after a reference exposure to A.
  EXPERIENCED = expose to A -> long neutral washout (until decoding ~ chance)
                -> RE-expose to A, tracking distance-to-S_A over re-exposure time.
  NAIVE       = fresh population -> first-ever exposure to A, same tracking.
  Savings     = EXPERIENCED approaches S_A faster / closer than NAIVE.

If experienced == naive, there is no latent memory beyond what decoding already
saw. If experienced is faster, there is genuine savings (recall-adjacent).

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_savings.py
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
from population_decoding import pop_state, apply_env  # reuse toxic/starved/abundant

TARGET = 'starved'          # environment whose memory we probe (leaves a clear signature)
PROBE_STEPS = [0, 40, 80, 160, 320]


def base_of(system):
    v = pop_state(system)
    return None if v is None else v[:4]


def new_system(seed):
    cfg = Ecology2DConfig(world_size=60, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=1500, orthogonal_expression=True)
    return Ecology2DSystem(cfg)


def reference_target(seed, encode=800, warmup=400):
    s = new_system(seed)
    for _ in range(warmup): s.step()
    for _ in range(encode): apply_env(s, TARGET); s.step()
    return base_of(s)


def trajectory_toward_target(s, target, steps=max(PROBE_STEPS)):
    """Expose s to TARGET; record L2 distance of heritable base-freq to `target`."""
    dist = {}
    dset = set(PROBE_STEPS)
    for t in range(steps + 1):
        if t in dset:
            b = base_of(s)
            dist[t] = float(np.linalg.norm(b - target)) if b is not None else np.nan
        apply_env(s, TARGET); s.step()
    return dist


def run_seed(seed, encode=800, washout=1000, warmup=400):
    S_A = reference_target(seed)

    # EXPERIENCED: learn A, forget (washout), then re-expose and track
    se = new_system(seed)
    for _ in range(warmup): se.step()
    for _ in range(encode): apply_env(se, TARGET); se.step()
    for _ in range(washout): se.step()          # neutral washout -> forgetting
    exp = trajectory_toward_target(se, S_A)

    # NAIVE: fresh population, first exposure to A, same tracking
    sn = new_system(seed + 10007)               # different seed so it never saw A
    for _ in range(warmup): sn.step()
    nai = trajectory_toward_target(sn, S_A)
    return exp, nai


def main(seeds=range(8)):
    seeds = list(seeds)
    print(f"SAVINGS / RECALL TEST  target={TARGET}  seeds={seeds}")
    print("distance-to-S_A during (re)exposure; lower = closer to the remembered state\n")
    exp_all = {t: [] for t in PROBE_STEPS}
    nai_all = {t: [] for t in PROBE_STEPS}
    for seed in seeds:
        exp, nai = run_seed(seed)
        for t in PROBE_STEPS:
            if not np.isnan(exp[t]): exp_all[t].append(exp[t])
            if not np.isnan(nai[t]): nai_all[t].append(nai[t])

    print(f"{'re-exp step':>11} | {'EXPERIENCED':>11} | {'NAIVE':>8} | {'savings(naive-exp)':>18}")
    print("-" * 58)
    savings = []
    for t in PROBE_STEPS:
        e = np.median(exp_all[t]); n = np.median(nai_all[t])
        savings.append(n - e)
        print(f"{t:>11} | {e:>11.3f} | {n:>8.3f} | {n-e:>+18.3f}")
    med_sav = float(np.median([s for s in savings]))
    print(f"\nmedian savings (naive_dist - experienced_dist) = {med_sav:+.3f}")
    if med_sav > 0.01:
        print("=> EXPERIENCED returns to the remembered state faster than NAIVE:")
        print("   genuine savings — a latent memory persists below the decoding threshold.")
    elif med_sav < -0.01:
        print("=> experienced is SLOWER than naive (no savings; interference if consistent).")
    else:
        print("=> experienced ~ naive: no measurable savings (no latent recall beyond storage).")


if __name__ == "__main__":
    main()
