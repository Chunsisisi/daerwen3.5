"""
Memory vs Tracking Test

The cleanest test of REAL memory: train with a gradient, then REMOVE the
gradient and see if the population retains its spatial bias.

If particles stay in the "correct" region after gradient removal →
  they REMEMBER (gene-encoded spatial preference persists)
If particles drift to uniform after removal →
  they were just TRACKING the gradient in real-time (no memory)

This is the hippocampus-specific test: memory means information persists
AFTER the stimulus is gone.

Control: frozen system (mutation_rate=0) should NOT retain, because
it can't evolve spatial preferences — it was just physics-tracking.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _root)

import numpy as np


def create_system(seed, mutation_rate=0.01, n_particles=800):
    import engine.chem_sim_genes as gm
    gm._USING_RUST = False
    from engine.core import Ecology2DSystem, Ecology2DConfig
    import io
    old = sys.stdout; sys.stdout = io.StringIO()
    np.random.seed(seed)
    cfg = Ecology2DConfig(
        world_size=80, n_particles=n_particles,
        genome_length=48, mutation_rate=mutation_rate,
    )
    system = Ecology2DSystem(cfg)
    system.rng = np.random.default_rng(seed)
    sys.stdout = old
    return system


def apply_north_gradient(system):
    world = system.config.world_size
    yy = np.arange(world, dtype=np.float32).reshape(-1, 1).repeat(world, axis=1)
    gradient = (yy / world) * 2.0
    cf = system.chemical_field
    cf.concentrations[:, :, cf.nutrient_index] = gradient.T
    cf.invalidate_gradient_cache()


def clear_gradient(system):
    """Remove all gradients — uniform nutrient field."""
    cf = system.chemical_field
    cf.concentrations[:, :, cf.nutrient_index] = 0.3  # uniform baseline
    cf.invalidate_gradient_cache()


def north_fraction(system):
    alive = [p for p in system.particles if p.alive]
    if len(alive) < 10:
        return 0.5
    positions = np.array([p.position for p in alive])
    return float(np.mean(positions[:, 1] > system.config.world_size / 2))


def alive_count(system):
    return sum(1 for p in system.particles if p.alive)


def run_trial(seed, mutation_rate, train_steps, hold_steps, label):
    system = create_system(seed, mutation_rate=mutation_rate)

    # Phase 1: warmup (no gradient, let population stabilize)
    for _ in range(500):
        system.step()
    baseline_north = north_fraction(system)

    # Phase 2: TRAIN with north gradient
    trajectory_train = []
    for s in range(train_steps):
        apply_north_gradient(system)
        system.step()
        if s % 200 == 0:
            trajectory_train.append(north_fraction(system))
    after_train = north_fraction(system)
    alive_after_train = alive_count(system)

    # Phase 3: REMOVE gradient — test memory retention
    clear_gradient(system)
    trajectory_hold = []
    for s in range(hold_steps):
        system.step()  # NO gradient applied
        if s % 100 == 0:
            trajectory_hold.append(north_fraction(system))
    after_hold = north_fraction(system)
    alive_after_hold = alive_count(system)

    return {
        'label': label,
        'seed': seed,
        'mutation_rate': mutation_rate,
        'baseline_north': baseline_north,
        'after_train': after_train,
        'after_hold': after_hold,
        'alive_train': alive_after_train,
        'alive_hold': alive_after_hold,
        'learning': after_train - baseline_north,      # did it learn?
        'retention': after_hold - 0.5,                  # does it remember? (0.5 = random)
        'memory_strength': after_hold - baseline_north, # net memory
        'trajectory_train': trajectory_train,
        'trajectory_hold': trajectory_hold,
    }


if __name__ == '__main__':
    N_SEEDS = 8
    TRAIN_STEPS = 3000
    HOLD_STEPS = 2000

    print("MEMORY vs TRACKING TEST")
    print(f"  N_PARTICLES=800, TRAIN={TRAIN_STEPS} steps, HOLD={HOLD_STEPS} steps (no gradient)")
    print(f"  N_SEEDS={N_SEEDS}")
    print(f"  Metric: north_fraction (random=0.50)")
    print()

    t0 = time.time()

    # ── Evolving system ──
    print("Running EVOLVING system (mutation_rate=0.01)...")
    evolving_results = []
    for seed in range(N_SEEDS):
        r = run_trial(seed, mutation_rate=0.01, train_steps=TRAIN_STEPS,
                      hold_steps=HOLD_STEPS, label='evolving')
        print(f"  seed {seed}: baseline={r['baseline_north']:.3f} → "
              f"after_train={r['after_train']:.3f} → "
              f"after_hold={r['after_hold']:.3f}  "
              f"alive={r['alive_hold']}  "
              f"memory={r['memory_strength']:+.3f}")
        evolving_results.append(r)

    # ── Frozen system (control) ──
    print("\nRunning FROZEN system (mutation_rate=0.0)...")
    frozen_results = []
    for seed in range(N_SEEDS):
        r = run_trial(seed, mutation_rate=0.0, train_steps=TRAIN_STEPS,
                      hold_steps=HOLD_STEPS, label='frozen')
        print(f"  seed {seed}: baseline={r['baseline_north']:.3f} → "
              f"after_train={r['after_train']:.3f} → "
              f"after_hold={r['after_hold']:.3f}  "
              f"alive={r['alive_hold']}  "
              f"memory={r['memory_strength']:+.3f}")
        frozen_results.append(r)

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY: MEMORY vs TRACKING")
    print(f"{'='*70}")

    for label, results in [('EVOLVING', evolving_results), ('FROZEN', frozen_results)]:
        bl = np.mean([r['baseline_north'] for r in results])
        at = np.mean([r['after_train'] for r in results])
        ah = np.mean([r['after_hold'] for r in results])
        ms = np.mean([r['memory_strength'] for r in results])
        ms_std = np.std([r['memory_strength'] for r in results])
        al = int(np.mean([r['alive_hold'] for r in results]))
        print(f"\n  {label}:")
        print(f"    baseline (before gradient): {bl:.4f}")
        print(f"    after training (gradient):  {at:.4f}  (learning = {at-bl:+.4f})")
        print(f"    after hold (NO gradient):   {ah:.4f}  (retention vs random = {ah-0.5:+.4f})")
        print(f"    memory_strength:            {ms:+.4f} ± {ms_std:.4f}")
        print(f"    alive after hold:           {al}")

    # ── Verdict ──
    ev_mem = np.mean([r['memory_strength'] for r in evolving_results])
    fr_mem = np.mean([r['memory_strength'] for r in frozen_results])
    ev_ret = np.mean([r['after_hold'] for r in evolving_results])
    fr_ret = np.mean([r['after_hold'] for r in frozen_results])

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Evolving memory_strength: {ev_mem:+.4f}")
    print(f"  Frozen memory_strength:   {fr_mem:+.4f}")
    print(f"  Difference:               {ev_mem - fr_mem:+.4f}")
    print()

    if ev_ret > 0.55 and ev_ret > fr_ret + 0.03:
        print("  ✓ REAL MEMORY: evolving system retains spatial bias after gradient removal")
        print("    Population genetics encode lasting spatial preference.")
    elif ev_ret > 0.52:
        print("  ~ WEAK MEMORY: slight retention, but not strong enough to be conclusive")
    else:
        print("  ✗ NO MEMORY: spatial bias disappears when gradient is removed")
        print("    System was tracking, not remembering.")

    if fr_ret > 0.55:
        print("  ⚠ WARNING: frozen system ALSO retains → retention is physics, not evolution")
    elif fr_ret < 0.52:
        print("  ✓ CONTROL PASSES: frozen system returns to random after gradient removal")

    print(f"\n  Total time: {elapsed/60:.1f} minutes")
