"""
First embodied probe: can the system's OUTPUT be shaped by its consequences?
(operant conditioning with chemical reward, contingent vs yoked control)

The "voltage" output = fraction of the population occupying a TARGET zone (a left
strip). This is the effector reading. The action (being in the zone) produces a
chemical reward, delivered per insight #2 (only chemical/selective pressure changes
genes): ATP injected into the zone, amount proportional to occupancy — so producing
the action reinforces it, and the reward differentially benefits the actors.

  contingent : reward ATP -> the TARGET zone (local, credits the actors)
  yoked      : same reward amount -> a RANDOM zone each step (decorrelated from action)

Learning (ground truth): target-zone occupancy rises above baseline AND above yoked.
We also log ID turnover in the zone to tell insight #1 apart — did the SAME particles
move in (behavior) or did NEW particles grow there (selection)?

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/operant_voltage.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig

WORLD = 50
ZONE = int(WORLD * 0.2)          # target zone: x < ZONE  (baseline occupancy ~0.2)
REWARD_K = 6.0                   # reward gain
PROBE = [0, 400, 800, 1200, 1600]


def target_frac(s):
    a = [p for p in s.particles if p.alive]
    if not a:
        return None
    xs = np.array([p.position[0] for p in a])
    return float(np.mean(xs < ZONE))


def zone_ids(s):
    return {p.id for p in s.particles if p.alive and p.position[0] < ZONE}


def inject(s, x_lo, amount, w=ZONE):
    x_lo = int(x_lo) % WORLD
    xs = (np.arange(x_lo, x_lo + w)) % WORLD
    s.chemical_field.concentrations[xs, :, s.chemical_field.ATP_index] += amount
    s.chemical_field.invalidate_gradient_cache()


def run(mode, seed, steps=1600):
    cfg = Ecology2DConfig(world_size=WORLD, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=800, orthogonal_expression=True)
    s = Ecology2DSystem(cfg)
    for _ in range(200):
        s.step()
    traj = {}
    prev_ids = zone_ids(s)
    turnover = []
    for t in range(steps + 1):
        tf = target_frac(s) or 0.0
        amount = REWARD_K * tf                     # reward proportional to the action
        if mode == 'contingent':
            inject(s, 0, amount)                    # -> target zone (credits actors)
        elif mode == 'yoked':
            rx = int(s.rng.integers(0, WORLD))      # -> random zone (decorrelated)
            inject(s, rx, amount)
        s.step()
        # turnover sampled every 20 steps (the set-diff is the expensive part)
        if t % 20 == 0:
            cur = zone_ids(s)
            if cur:
                turnover.append(len(cur - prev_ids) / len(cur))
            prev_ids = cur
        if t in PROBE:
            traj[t] = (target_frac(s), sum(p.alive for p in s.particles))
        if sum(p.alive for p in s.particles) == 0:
            break
    mean_turn = float(np.mean(turnover)) if turnover else None
    return traj, mean_turn


def main(seeds=range(3), steps=1200):
    seeds = list(seeds)
    print(f"OPERANT VOLTAGE PROBE  seeds={seeds}  baseline occupancy≈0.20", flush=True)
    for mode in ['contingent', 'yoked']:
        rows = {t: [] for t in PROBE}
        turns = []
        for seed in seeds:
            tr, mt = run(mode, seed, steps=steps)
            for t in PROBE:
                if t in tr and tr[t][0] is not None:
                    rows[t].append(tr[t][0])
            if mt is not None:
                turns.append(mt)
            final = tr.get(max(k for k in tr), (None,))[0]
            print(f"  ...{mode} seed{seed} done (final occ={final:.2f})", flush=True)
        print(f"-- {mode} --")
        for t in PROBE:
            if rows[t]:
                print(f"  step {t:>4}: target-zone occupancy = {np.median(rows[t]):.3f}")
        if turns:
            print(f"  in-zone turnover per step = {np.median(turns):.3f} "
                  f"({'grew (selection)' if np.median(turns) > 0.15 else 'stayed (movement)'})")
        print()
    print("READING: if contingent occupancy climbs above ~0.2 and above yoked, the")
    print("system's OUTPUT is being shaped by its consequences — confirmable operant")
    print("learning. Turnover tells whether it's by growth (selection) or movement.")


if __name__ == "__main__":
    main()
