"""
Multimodal Integration Benchmark

Tests whether DAERWEN genuinely INTEGRATES multiple sensory modalities, or
just responds to each one independently.

⚠ KNOWN METHODOLOGICAL CONCERN (to be addressed in follow-up):
  Spatial response metrics (north_bias etc.) CAN be produced by two
  distinct mechanisms that this test does NOT distinguish:
    (a) true movement — existing particles relocate up a gradient
    (b) selection-via-replication — particles in a favorable region
        replicate faster (offspring spawn nearby), particles in poor
        regions starve. Over time the population SHIFTS without any
        individual actually moving.
  Both produce identical north_bias/east_bias readings. For the claim
  "DAERWEN integrates multi-modal SIGNALS", only (a) is relevant;
  (b) is just differential fitness.
  A clean follow-up would disable replication (mutation_rate=0 and
  unreachable replication_threshold) and check whether the responses
  persist. That isolates (a) from (b).

Three modalities mapped to DAERWEN primitives:
  - Vision  = persistent spatial gradient (north-up nutrient field)
  - Audio   = periodic chemical pulse at world centre (every 50 steps)
  - Touch   = localised inhibitor pulse at fixed corner (20, 20)

Eight conditions: {}, {V}, {A}, {T}, {VA}, {VT}, {AT}, {VAT}

Response vector measured over the LAST 500 steps of each run:
  north_bias         : fraction of particles in north half (expects ↑ under V)
  east_bias          : fraction in east half (sanity baseline)
  oscillation_power  : FFT energy at audio frequency (expects ↑ under A)
  touch_avoidance    : 1 − density around (20, 20) (expects ↑ under T)
  alive              : final population (control)
  replications       : total reps (control)

Integration test:
  For each multi-modal condition XY, check whether
      response(XY) ≈ response(X alone) + response(Y alone) − baseline
  If yes → linear superposition (no integration)
  If deviation is large and consistent → the system integrates modalities
  non-linearly (this is the core DAERWEN claim to validate).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _root)

import numpy as np


# ── Constants ────────────────────────────────────────────────────────

WORLD_SIZE = 80
N_PARTICLES = 800
GENOME_LENGTH = 48
N_STEPS = 2500
MEASURE_LAST = 800           # measure response over last N steps
AUDIO_PERIOD = 50            # steps between audio pulses
TOUCH_ZONE = (20.0, 20.0)    # corner where touch is applied
TOUCH_RADIUS = 8.0


# ── Modality application functions ───────────────────────────────────

def apply_vision(system, step):
    """Persistent north-up nutrient gradient. Re-applied every 100 steps
    to counteract diffusion."""
    if step % 100 == 0:
        from engine.core import ExternalInput
        system.apply_external_input(ExternalInput(
            input_type='gradient_field',
            params={'axis': 'y', 'start_value': 0.0, 'end_value': 1.5,
                    'chemical_index': 1},  # nutrient
        ))


def apply_audio(system, step):
    """Periodic central ATP pulse at a fixed frequency.
    Low intensity (0.1) so it functions as a SIGNAL not a resource —
    detectable as rhythm, but doesn't meaningfully feed the population."""
    if step % AUDIO_PERIOD == 0:
        from engine.core import ExternalInput
        system.apply_external_input(ExternalInput(
            input_type='chemical_pulse',
            params={'x': WORLD_SIZE // 2, 'y': WORLD_SIZE // 2,
                    'radius': 25, 'intensity': 0.1,  # reduced from 0.6
                    'chemical_index': 0},  # ATP
        ))


def apply_touch(system, step):
    """Localised inhibitor pulse in the corner, repeated every 10 steps."""
    if step % 10 == 0:
        from engine.core import ExternalInput
        system.apply_external_input(ExternalInput(
            input_type='chemical_pulse',
            params={'x': int(TOUCH_ZONE[0]), 'y': int(TOUCH_ZONE[1]),
                    'radius': int(TOUCH_RADIUS), 'intensity': 0.4,
                    'chemical_index': 2},  # inhibitor
        ))


MODALITIES = {
    'V': apply_vision,
    'A': apply_audio,
    'T': apply_touch,
}


# ── Response measurement ─────────────────────────────────────────────

def measure_response(position_history, alive_history):
    """Compute response vector from simulation history.

    position_history: list of (step, positions_array)  for last MEASURE_LAST steps
    alive_history:    list of alive counts, one per step, full simulation
    """
    n_buckets_positions = len(position_history)
    if n_buckets_positions == 0:
        return {k: 0.0 for k in ['north_bias', 'east_bias', 'oscillation_power',
                                  'touch_avoidance', 'alive_mean', 'alive_final']}

    # Aggregate positions
    all_positions = []
    for _, pos in position_history:
        if pos is not None and len(pos) > 0:
            all_positions.append(pos)

    north_bias = 0.0
    east_bias = 0.0
    touch_avoidance = 1.0
    if all_positions:
        all_xy = np.concatenate(all_positions, axis=0)
        north_bias = float(np.mean(all_xy[:, 1] > WORLD_SIZE / 2))
        east_bias  = float(np.mean(all_xy[:, 0] > WORLD_SIZE / 2))
        # fraction within touch zone
        dx = all_xy[:, 0] - TOUCH_ZONE[0]
        dy = all_xy[:, 1] - TOUCH_ZONE[1]
        in_zone = np.mean((dx * dx + dy * dy) < (TOUCH_RADIUS * 2) ** 2)
        touch_avoidance = float(1.0 - in_zone)

    # Audio oscillation: FFT of alive count over last MEASURE_LAST steps
    oscillation_power = 0.0
    if len(alive_history) >= MEASURE_LAST:
        ts = np.array(alive_history[-MEASURE_LAST:], dtype=float)
        ts = ts - np.mean(ts)
        spectrum = np.abs(np.fft.rfft(ts))
        # target frequency index = MEASURE_LAST / AUDIO_PERIOD
        target_idx = int(round(MEASURE_LAST / AUDIO_PERIOD))
        target_idx = min(target_idx, len(spectrum) - 1)
        # normalize by total spectrum energy
        total = float(np.sum(spectrum)) + 1e-9
        oscillation_power = float(spectrum[target_idx]) / total

    alive_mean = float(np.mean(alive_history[-MEASURE_LAST:])) if alive_history else 0.0
    alive_final = float(alive_history[-1]) if alive_history else 0.0

    return {
        'north_bias':        north_bias,
        'east_bias':         east_bias,
        'oscillation_power': oscillation_power,
        'touch_avoidance':   touch_avoidance,
        'alive_mean':        alive_mean,
        'alive_final':       alive_final,
    }


# ── Single trial ─────────────────────────────────────────────────────

def run_trial(args):
    """One trial: condition × seed."""
    condition, seed = args
    # Force Python fallback in subprocess (avoids pyo3 deadlock on Windows mp).
    import engine.chem_sim_genes as gm
    gm._USING_RUST = False

    from engine.core import Ecology2DSystem, Ecology2DConfig
    import io
    sys.stdout = io.StringIO()

    np.random.seed(seed)
    cfg = Ecology2DConfig(
        world_size=WORLD_SIZE,
        n_particles=N_PARTICLES,
        genome_length=GENOME_LENGTH,
        mutation_rate=0.01,
    )
    system = Ecology2DSystem(cfg)
    system.rng = np.random.default_rng(seed)

    alive_history = []
    position_history = []

    for step in range(N_STEPS):
        # Apply active modalities
        for mod_name in condition:
            MODALITIES[mod_name](system, step)
        system.step()
        # Record alive count every step (for FFT)
        alive = sum(1 for p in system.particles if p.alive)
        alive_history.append(alive)
        # Record positions in the measurement window (every 5 steps to save memory)
        if step >= N_STEPS - MEASURE_LAST and step % 5 == 0:
            pos = np.array([p.position for p in system.particles if p.alive])
            position_history.append((step, pos))

    sys.stdout = sys.__stdout__

    response = measure_response(position_history, alive_history)
    response['condition'] = ''.join(condition) or 'baseline'
    response['seed'] = seed
    return response


# ── Integration analysis ─────────────────────────────────────────────

def integration_analysis(by_cond):
    """For each pair (X, Y), predict linear superposition and compare
    to actual multi-modal response."""
    metrics = ['north_bias', 'east_bias', 'oscillation_power', 'touch_avoidance']
    baseline = by_cond.get('baseline', {})
    if not baseline:
        print("no baseline data")
        return

    print()
    print("INTEGRATION TEST: multi = linear superposition of singles?")
    print("-" * 78)
    print(f"{'combo':>5} {'metric':<20} {'single_sum':>12} {'actual':>10} {'Δ':>8} {'ratio':>8}")
    print("-" * 78)
    for combo in ['VA', 'VT', 'AT', 'VAT']:
        if combo not in by_cond:
            continue
        actual = by_cond[combo]
        singles = list(combo)
        for m in metrics:
            baseline_v = baseline.get(m, 0)
            # Linear prediction: sum of (single - baseline) + baseline
            predicted = baseline_v + sum(by_cond[s][m] - baseline_v for s in singles if s in by_cond)
            actual_v = actual.get(m, 0)
            delta = actual_v - predicted
            ratio = (actual_v / predicted) if abs(predicted) > 1e-6 else float('nan')
            print(f"{combo:>5} {m:<20} {predicted:>12.4f} {actual_v:>10.4f} {delta:>+8.4f} {ratio:>8.3f}")
        print("-" * 78)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    seeds = [42, 123, 7, 314, 2718, 999, 31415, 1111, 2048, 55555]
    conditions = [
        (),                 # baseline
        ('V',), ('A',), ('T',),
        ('V', 'A'), ('V', 'T'), ('A', 'T'),
        ('V', 'A', 'T'),
    ]
    tasks = [(c, s) for c in conditions for s in seeds]
    print(f"Running {len(tasks)} trials ({len(conditions)} conditions × {len(seeds)} seeds)")
    print(f"  N_STEPS={N_STEPS}, MEASURE_LAST={MEASURE_LAST}")
    print(f"  modalities: V (vision/north gradient), A (audio/periodic pulse),")
    print(f"              T (touch/corner inhibitor)")
    print()

    t0 = time.time()
    with Pool(processes=min(10, len(tasks))) as pool:
        results = pool.map(run_trial, tasks)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s\n")

    # Average across seeds per condition
    by_cond = {}
    all_conds = sorted(set(r['condition'] for r in results))
    for cond_name in all_conds:
        rs = [r for r in results if r['condition'] == cond_name]
        agg = {}
        for k in ['north_bias', 'east_bias', 'oscillation_power', 'touch_avoidance',
                  'alive_mean', 'alive_final']:
            vals = [r[k] for r in rs]
            agg[k] = float(np.mean(vals))
        by_cond[cond_name] = agg

    # Report
    print("=" * 78)
    print("RESPONSE VECTOR by condition (mean over seeds)")
    print("=" * 78)
    header = f"{'cond':<10}"
    for k in ['north_bias', 'east_bias', 'oscillation', 'touch_avoid', 'alive_mean']:
        header += f" {k:>12}"
    print(header)
    print("-" * 78)
    for cond_name in all_conds:
        r = by_cond[cond_name]
        line = f"{cond_name:<10}"
        line += f" {r['north_bias']:>12.4f}"
        line += f" {r['east_bias']:>12.4f}"
        line += f" {r['oscillation_power']:>12.4f}"
        line += f" {r['touch_avoidance']:>12.4f}"
        line += f" {r['alive_mean']:>12.1f}"
        print(line)

    # Integration test
    integration_analysis(by_cond)

    # Save JSON
    import json
    out_dir = Path(_root) / 'benchmark_results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'multimodal_integration_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'world_size': WORLD_SIZE, 'n_particles': N_PARTICLES,
                'n_steps': N_STEPS, 'measure_last': MEASURE_LAST,
                'audio_period': AUDIO_PERIOD, 'touch_zone': TOUCH_ZONE,
                'seeds': seeds,
            },
            'by_cond': by_cond,
            'raw': results,
            'elapsed_seconds': elapsed,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
