"""
Task 3 (emotional weighting): does stronger selection leave a DEEPER, more
PERSISTENT genetic memory?

The biological hippocampus encodes survival-relevant / high-arousal events more
strongly and for longer. Analog here: a stronger selection pressure (a more lethal
environment) should shift gene frequencies more decisively, so the memory of that
environment should (a) decode HIGHER right after exposure and (b) DECAY SLOWER.

We contrast a 'stress' environment (elevated inhibitor) against 'neutral' (no
imposed field) at three stress intensities. For each intensity we measure the
stress-vs-neutral decoding accuracy across a forgetting delay grid.

Prediction if emotional weighting holds: higher intensity -> higher d=0 accuracy
AND slower decay (larger area under the forgetting curve).

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_emotional_weighting.py
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
from population_decoding import pop_state, decode_at_delay

INTENSITIES = [0.12, 0.25, 0.45]     # inhibitor level of the 'stress' environment
DELAYS = [0, 100, 200, 400, 700]


def run_trial(seed, stress_intensity, warmup=400, encode=800, delays=DELAYS):
    cfg = Ecology2DConfig(world_size=60, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=1500, orthogonal_expression=True)
    s = Ecology2DSystem(cfg)
    for _ in range(warmup):
        s.step()
    for _ in range(encode):
        if stress_intensity is not None:                 # 'stress' env
            s.chemical_field.concentrations[:, :, s.chemical_field.inhibitor_index] = stress_intensity
            s.chemical_field.concentrations[:, :, s.chemical_field.ATP_index] = 0.35
            s.chemical_field.invalidate_gradient_cache()
        # neutral env: impose nothing
        s.step()
    states, dset = {}, set(delays)
    for t in range(max(delays) + 1):
        if t in dset:
            states[t] = pop_state(s)
        s.step()
    return states


def main(seeds=range(10)):
    seeds = list(seeds)
    rng = np.random.default_rng(0)
    print(f"EMOTIONAL WEIGHTING  seeds={seeds}  intensities={INTENSITIES}  (chance 0.5)")

    # neutral trials shared across intensities (label 0)
    neutral = {seed: run_trial(seed, None) for seed in seeds}

    summary = []
    for inten in INTENSITIES:
        stress = {seed: run_trial(seed, inten) for seed in seeds}
        print(f"\n-- stress inhibitor = {inten} --")
        print(f"{'delay':>6} | {'stress-vs-neutral acc':>21} | {'shuffled':>9}")
        accs = []
        for d in DELAYS:
            X, y, groups = [], [], []
            for seed in seeds:
                for lab, src in [(0, neutral), (1, stress)]:
                    v = src[seed].get(d)
                    if v is not None:
                        X.append(v[:4]); y.append(lab); groups.append(seed)
            X = np.array(X); y = np.array(y); groups = np.array(groups)
            if len(set(y)) < 2:
                continue
            acc = decode_at_delay(X, y, groups)
            shuf = np.mean([decode_at_delay(X, y, groups, shuffle=True, rng=rng) for _ in range(4)])
            accs.append((d, acc))
            print(f"{d:>6} | {acc:>21.3f} | {shuf:>9.3f}")
        d0 = accs[0][1] if accs else float('nan')
        auc = float(np.mean([a for _, a in accs])) if accs else float('nan')  # mean over delays ~ persistence
        summary.append((inten, d0, auc))

    print("\nintensity | d0_acc | mean_over_delays (persistence)")
    for inten, d0, auc in summary:
        print(f"{inten:>9} | {d0:>6.3f} | {auc:>6.3f}")
    if len(summary) >= 2:
        d0s = [s[1] for s in summary]; aucs = [s[2] for s in summary]
        rising_d0 = d0s[-1] > d0s[0] + 0.05
        rising_auc = aucs[-1] > aucs[0] + 0.05
        print()
        if rising_d0 and rising_auc:
            print("=> stronger selection -> higher initial decoding AND greater persistence:")
            print("   emotional-weighting analog CONFIRMED (deeper, longer-lasting memory).")
        elif rising_d0 or rising_auc:
            print("=> partial: one of {depth, persistence} rises with intensity; the other flat.")
        else:
            print("=> no emotional-weighting effect: memory depth/persistence ~ independent of selection strength.")


if __name__ == "__main__":
    main()
