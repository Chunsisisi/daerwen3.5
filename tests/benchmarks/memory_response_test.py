"""
Memory = Differential Population Response

NOT: "are particles still in the north after training?"
BUT: "does a trained population respond DIFFERENTLY to a weak signal
      than a naive population?"

Protocol:
  Group A (trained):   3000 steps north gradient → 1000 steps no signal → weak probe
  Group B (naive):     4000 steps no gradient                           → same weak probe

  Measure RESPONSE to probe:
  - How fast does north_fraction increase? (response speed)
  - How high does it go? (response amplitude)
  - How does population dynamics change? (alive count, replication rate)

  If A responds faster/stronger than B → memory exists
  If A ≈ B → no memory (training had no lasting effect on population behavior)

This is the hippocampal test: memory means a CUE triggers a DIFFERENT
RESPONSE in experienced vs naive systems. Not that the system stays in
a fixed state.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _root)

import numpy as np


def create_system(seed, n_particles=800):
    import engine.chem_sim_genes as gm
    gm._USING_RUST = False
    from engine.core import Ecology2DSystem, Ecology2DConfig
    import io
    old = sys.stdout; sys.stdout = io.StringIO()
    np.random.seed(seed)
    cfg = Ecology2DConfig(
        world_size=80, n_particles=n_particles,
        genome_length=48, mutation_rate=0.01,
    )
    system = Ecology2DSystem(cfg)
    system.rng = np.random.default_rng(seed)
    sys.stdout = old
    return system


def apply_strong_north(system):
    w = system.config.world_size
    yy = np.arange(w, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
    system.chemical_field.concentrations[:, :, system.chemical_field.nutrient_index] = (yy.T / w) * 2.0
    system.chemical_field.invalidate_gradient_cache()


def apply_weak_north(system):
    """Subtle north hint — just 20% of full gradient strength."""
    w = system.config.world_size
    yy = np.arange(w, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
    cf = system.chemical_field
    # ADD weak gradient on top of existing field (don't overwrite)
    cf.concentrations[:, :, cf.nutrient_index] += (yy.T / w) * 0.4
    cf.invalidate_gradient_cache()


def clear_field(system):
    cf = system.chemical_field
    cf.concentrations[:, :, cf.nutrient_index] = 0.3
    cf.invalidate_gradient_cache()


def north_fraction(system):
    alive = [p for p in system.particles if p.alive]
    if len(alive) < 10:
        return 0.5
    return float(np.mean([p.position[1] > system.config.world_size / 2 for p in alive]))


def alive_count(system):
    return sum(1 for p in system.particles if p.alive)


def measure_probe_response(system, probe_steps=500, sample_every=10):
    """Apply weak north probe and record population response curve."""
    response = []
    for s in range(probe_steps):
        apply_weak_north(system)
        system.step()
        if s % sample_every == 0:
            response.append({
                'step': s,
                'north_frac': north_fraction(system),
                'alive': alive_count(system),
            })
    return response


def response_metrics(response):
    """Extract speed and amplitude from response curve."""
    norths = [r['north_frac'] for r in response]
    if not norths:
        return 0.5, 0.5, 0.0

    initial = norths[0]
    final = norths[-1]
    peak = max(norths)

    # Speed: how many steps to reach 80% of peak response
    target = initial + 0.8 * (peak - initial)
    speed_steps = len(norths)  # default: never reached
    for i, n in enumerate(norths):
        if n >= target:
            speed_steps = i
            break

    return initial, final, peak, speed_steps


def run_group(label, seed, do_training, train_steps=3000, rest_steps=1000):
    system = create_system(seed)

    # Warmup
    for _ in range(500):
        system.step()

    if do_training:
        # TRAIN: strong north gradient
        for _ in range(train_steps):
            apply_strong_north(system)
            system.step()
    else:
        # NAIVE: just run without gradient
        for _ in range(train_steps):
            system.step()

    # REST: clear field, let physics settle (remove tracking bias)
    for _ in range(rest_steps):
        clear_field(system)
        system.step()

    before_probe = north_fraction(system)
    alive_before = alive_count(system)

    # PROBE: weak north signal
    response = measure_probe_response(system, probe_steps=500, sample_every=10)
    initial, final, peak, speed = response_metrics(response)

    return {
        'label': label,
        'seed': seed,
        'before_probe': before_probe,
        'alive': alive_before,
        'probe_initial': initial,
        'probe_final': final,
        'probe_peak': peak,
        'speed_to_80pct': speed,
        'response_amplitude': final - initial,
    }


if __name__ == '__main__':
    N_SEEDS = 8
    print("MEMORY = DIFFERENTIAL POPULATION RESPONSE")
    print(f"  Protocol: trained(3000 steps) vs naive(no training)")
    print(f"  After 1000 steps rest (clear field), apply WEAK north probe")
    print(f"  Measure: response speed and amplitude")
    print(f"  N_SEEDS={N_SEEDS}")
    print()

    t0 = time.time()

    trained_results = []
    naive_results = []

    for seed in range(N_SEEDS):
        print(f"  seed {seed}...", end=" ", flush=True)
        t = run_group('trained', seed, do_training=True)
        n = run_group('naive', seed, do_training=False)
        trained_results.append(t)
        naive_results.append(n)
        print(f"trained: before={t['before_probe']:.3f} peak={t['probe_peak']:.3f} "
              f"amp={t['response_amplitude']:+.3f} | "
              f"naive: before={n['before_probe']:.3f} peak={n['probe_peak']:.3f} "
              f"amp={n['response_amplitude']:+.3f}")

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for label, results in [('TRAINED', trained_results), ('NAIVE', naive_results)]:
        bp = np.mean([r['before_probe'] for r in results])
        pi = np.mean([r['probe_initial'] for r in results])
        pf = np.mean([r['probe_final'] for r in results])
        pp = np.mean([r['probe_peak'] for r in results])
        amp = np.mean([r['response_amplitude'] for r in results])
        amp_std = np.std([r['response_amplitude'] for r in results])
        spd = np.mean([r['speed_to_80pct'] for r in results])
        al = int(np.mean([r['alive'] for r in results]))
        print(f"\n  {label}:")
        print(f"    before probe (after rest): {bp:.4f}  (random=0.50)")
        print(f"    probe response:  initial={pi:.4f} → final={pf:.4f} → peak={pp:.4f}")
        print(f"    response amplitude: {amp:+.4f} ± {amp_std:.4f}")
        print(f"    speed to 80% peak: {spd:.0f} sample points")
        print(f"    alive: {al}")

    # Comparison
    t_amp = np.mean([r['response_amplitude'] for r in trained_results])
    n_amp = np.mean([r['response_amplitude'] for r in naive_results])
    t_peak = np.mean([r['probe_peak'] for r in trained_results])
    n_peak = np.mean([r['probe_peak'] for r in naive_results])
    t_speed = np.mean([r['speed_to_80pct'] for r in trained_results])
    n_speed = np.mean([r['speed_to_80pct'] for r in naive_results])

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Response amplitude:  trained={t_amp:+.4f}  naive={n_amp:+.4f}  diff={t_amp-n_amp:+.4f}")
    print(f"  Peak response:       trained={t_peak:.4f}   naive={n_peak:.4f}   diff={t_peak-n_peak:+.4f}")
    print(f"  Response speed:      trained={t_speed:.0f}      naive={n_speed:.0f}")
    print()

    if t_amp > n_amp + 0.02 or t_peak > n_peak + 0.03:
        print("  ✓ MEMORY DETECTED: trained population responds MORE STRONGLY to weak probe")
        print("    Population genetics encode a lasting sensitivity to previously-experienced patterns.")
    elif t_speed < n_speed * 0.7:
        print("  ✓ MEMORY DETECTED: trained population responds FASTER to weak probe")
    elif abs(t_amp - n_amp) < 0.01 and abs(t_peak - n_peak) < 0.02:
        print("  ✗ NO MEMORY: trained and naive populations respond identically to probe")
        print("    Training had no lasting effect on population response behavior.")
    else:
        print("  ~ INCONCLUSIVE: small difference, need more seeds or longer training")

    print(f"\n  Total time: {elapsed/60:.1f} minutes")
