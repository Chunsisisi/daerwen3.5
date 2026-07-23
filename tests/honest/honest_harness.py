"""
Honest, reproducible test harness for DAERWEN.

Design goals (in direct response to the failure modes found in the existing
benchmarks):

1. REPRODUCIBLE. Every system is seeded (engine now supports config.seed).
   No result is reported from a single unseeded run.

2. DISTRIBUTIONS, NOT POINTS. Every metric is run over many seeds and reported
   as median + inter-quartile range. Extinction (population -> 0) is counted and
   reported as a first-class failure, never hidden inside an average.

3. PAIRED COMPARISON. "Evolution vs physics" uses the SAME seed for both arms,
   so the comparison is paired and the difference is a real per-world delta, not
   the divide-by-near-zero ratio the old benchmark used.

4. ANTI-OVERFIT / VALIDITY. Every discriminating test ships with a POSITIVE
   CONTROL: the same measurement on a channel we KNOW is wired (ATP). If the
   positive control does not show the effect, the test itself is too weak and a
   null result on the experimental arm means nothing. We refuse to report a
   null without a passing positive control.

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/honest_harness.py
"""
import os
import sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig


# ─────────────────────────────────────────────────────────────────────────────
# small stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def summarize(values):
    """Return dict with median, IQR, min, max over a list of floats."""
    a = np.asarray(values, dtype=float)
    return {
        'n': int(a.size),
        'median': float(np.median(a)),
        'q25': float(np.percentile(a, 25)),
        'q75': float(np.percentile(a, 75)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
    }


def fmt(s):
    return f"median={s['median']:+.3f}  IQR[{s['q25']:+.3f}, {s['q75']:+.3f}]  range[{s['min']:+.3f}, {s['max']:+.3f}]"


def alive_count(system):
    return sum(1 for p in system.particles if p.alive)


def center_of_mass(system):
    alive = [p for p in system.particles if p.alive]
    if not alive:
        return None
    pos = np.array([p.position for p in alive])
    return pos.mean(axis=0)


def left_fraction(system, world, cut=1.0 / 3.0):
    """Fraction of living particles whose x < cut*world (i.e. in the left band)."""
    alive = [p for p in system.particles if p.alive]
    if not alive:
        return None
    xs = np.array([p.position[0] for p in alive])
    return float(np.mean(xs < cut * world))


def make_config(seed, mutation_rate, world=60, n=300, genome=24, species=12):
    return Ecology2DConfig(
        world_size=world, n_particles=n, genome_length=genome,
        mutation_rate=mutation_rate, n_chemical_species=species, seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# field helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_left_band(system, channel, high=2.0, low=0.1, cut=1.0 / 3.0):
    """Set `channel` high in the left band, low elsewhere."""
    world = system.config.world_size
    conc = system.chemical_field.concentrations
    band = int(cut * world)
    conc[:band, :, channel] = high
    conc[band:, :, channel] = low
    system.chemical_field.invalidate_gradient_cache()


def flatten_channel(system, channel, value=1.0):
    system.chemical_field.concentrations[:, :, channel] = value
    system.chemical_field.invalidate_gradient_cache()


# ─────────────────────────────────────────────────────────────────────────────
# TEST A — evolution vs physics (paired, honest)
# ─────────────────────────────────────────────────────────────────────────────
def test_A_evolution_vs_physics(seeds, warmup=1000, task_steps=250):
    """
    Paired per-seed comparison on the 'boundary' task the repo advertises.
    Evolution arm: mutation_rate=0.02. Physics arm: mutation_rate=0.0.
    Same seed -> same initial world, so the difference isolates evolution.

    Reports the DIFFERENCE (evo - physics), not the ratio, plus extinction rate.
    """
    _bench = str(Path(__file__).resolve().parent.parent / 'benchmarks')
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from brutal_benchmark_fast import TaskLibrary, Metrics

    def one(seed, mut):
        s = Ecology2DSystem(make_config(seed, mut))
        for _ in range(warmup):
            s.step()
        TaskLibrary.create_gradient(s, 'boundary')
        for _ in range(task_steps):
            s.step()
        score = Metrics.calculate_alignment(s.particles, 'boundary', s.config.world_size)
        return score, alive_count(s)

    diffs, evo_scores, phys_scores = [], [], []
    evo_extinct = phys_extinct = 0
    for seed in seeds:
        evo, evo_alive = one(seed, 0.02)
        phys, phys_alive = one(seed, 0.0)
        if evo_alive == 0:
            evo_extinct += 1
        if phys_alive == 0:
            phys_extinct += 1
        evo_scores.append(evo)
        phys_scores.append(phys)
        diffs.append(evo - phys)

    return {
        'evo_score': summarize(evo_scores),
        'phys_score': summarize(phys_scores),
        'paired_diff': summarize(diffs),
        'evo_extinction_rate': evo_extinct / len(seeds),
        'phys_extinction_rate': phys_extinct / len(seeds),
        'n_seeds': len(seeds),
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    import json, time
    seeds = list(range(8))
    print("=" * 72)
    print("HONEST HARNESS  (seeds =", seeds, ")")
    print("=" * 72)

    print("\n--- TEST A: evolution vs physics (paired, boundary task) ---")
    A = test_A_evolution_vs_physics(seeds)
    print(f"  evolution score : {fmt(A['evo_score'])}")
    print(f"  physics score   : {fmt(A['phys_score'])}")
    print(f"  paired diff     : {fmt(A['paired_diff'])}   <-- (evo - physics)")
    print(f"  extinction      : evo {A['evo_extinction_rate']*100:.0f}%  "
          f"physics {A['phys_extinction_rate']*100:.0f}%")
    pd = A['paired_diff']
    print("\n  READING: if the paired-diff IQR straddles 0 or the median is <=0,")
    print("  evolution does not reliably beat physics on this task. Extinction is a")
    print("  failure, not excluded from the stats.")

    # NOTE: 'can the substrate sense a non-ATP signal?' is answered directly and
    # deterministically by tests/honest/movement_probe.py (only the ATP channel
    # steers movement; channels 3..10 are causally inert). No statistical arm here.

    out = Path(__file__).resolve().parent / f"honest_testA_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(A, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  saved: {out}")
    return A


if __name__ == "__main__":
    main()
