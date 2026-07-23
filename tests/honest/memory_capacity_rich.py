"""
Payoff test for the root-cause fix (rich_channels): does activating the inert
channels expand MEMORY CAPACITY?

Root cause of the low-capacity memory (Finding 9b): only ~3 channels are causally
active, so environments that differ in the other channels leave no distinct genetic
trace. `rich_channels` gives each particle an independently-evolvable gene per inert
channel (attractive+nutritious if >0, repulsive+toxic if <0). Prediction: with it
ON, environments that differ in DIFFERENT inert channels become distinguishable in
the heritable `channel_genes` vector; with it OFF they are unremembered (chance).

6 environments, each elevating one inert channel (3..8). Decode which environment
from the heritable state (channel_genes when rich; base-freq otherwise), leave-one-
seed-out CV, shuffle control, + forgetting curve + RSA.

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_capacity_rich.py
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
from population_decoding import decode_at_delay

CHANNELS = [3, 4, 5, 6, 7, 8]      # each environment elevates one inert channel
DELAYS = [0, 100, 200, 400, 800]


def apply_env(system, channel):
    cf = system.chemical_field
    c = cf.concentrations
    c[:, :, cf.ATP_index] = 0.30                 # moderate energy so populations survive
    for ch in CHANNELS:
        c[:, :, ch] = 1.5 if ch == channel else 0.05
    cf.invalidate_gradient_cache()


def heritable_state(system, rich):
    alive = [p for p in system.particles if p.alive]
    if len(alive) < 10:
        return None
    if rich:
        return np.mean([p.channel_genes for p in alive], axis=0)          # 8-d, heritable
    bases = np.concatenate([np.asarray(p.genome) for p in alive])
    return np.array([np.mean(bases == b) for b in range(4)])              # base freq (4-d)


def run_trial(seed, channel, rich, warmup=300, encode=700, delays=DELAYS):
    cfg = Ecology2DConfig(world_size=60, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=1000, orthogonal_expression=True,
                          rich_channels=rich)
    s = Ecology2DSystem(cfg)
    for _ in range(warmup):
        s.step()
    for _ in range(encode):
        apply_env(s, channel); s.step()
    states, dset = {}, set(delays)
    for t in range(max(delays) + 1):
        if t in dset:
            states[t] = heritable_state(s, rich)
        s.step()
    return states


def run_condition(rich, seeds):
    rng = np.random.default_rng(0)
    data = {}
    for seed in seeds:
        for ch in CHANNELS:
            data[(seed, ch)] = run_trial(seed, ch, rich)
    label = 'rich_channels ON (channel_genes memory)' if rich else 'rich_channels OFF (base-freq memory)'
    print(f"\n=== {label} ===  {len(CHANNELS)} envs, chance={1/len(CHANNELS):.3f}")
    print(f"{'delay':>6} | {'decode_acc':>10} | {'shuffled':>9}")
    curve = []
    for d in DELAYS:
        X, y, g = [], [], []
        for (seed, ch), st in data.items():
            v = st.get(d)
            if v is not None:
                X.append(v); y.append(CHANNELS.index(ch)); g.append(seed)
        X = np.array(X); y = np.array(y); g = np.array(g)
        if len(set(y)) < 2:
            print(f"{d:>6} | insufficient"); continue
        acc = decode_at_delay(X, y, g)
        shuf = np.mean([decode_at_delay(X, y, g, shuffle=True, rng=rng) for _ in range(5)])
        curve.append((d, acc)); print(f"{d:>6} | {acc:>10.3f} | {shuf:>9.3f}")
    # RSA collinearity at d=0
    by_env = {ch: [] for ch in CHANNELS}
    for (seed, ch), st in data.items():
        v = st.get(0)
        if v is not None:
            by_env[ch].append(v)
    means = np.array([np.mean(by_env[ch], axis=0) for ch in CHANNELS])
    sim = np.corrcoef(means)
    off = sim[np.triu_indices(len(CHANNELS), k=1)]
    print(f"RSA mean |off-diagonal similarity| = {np.mean(np.abs(off)):.2f} "
          f"(lower = more distinct environments)")
    return curve


def main(seeds=range(6)):
    seeds = list(seeds)
    print(f"MEMORY CAPACITY — rich_channels payoff test  seeds={seeds}")
    off = run_condition(False, seeds)
    on = run_condition(True, seeds)
    chance = 1 / len(CHANNELS)
    print("\nSUMMARY (d=0 decoding of 6 inert-channel environments):")
    print(f"  rich OFF: {off[0][1]:.3f}   rich ON: {on[0][1]:.3f}   chance: {chance:.3f}")
    if on[0][1] > off[0][1] + 0.15 and on[0][1] > chance + 0.15:
        print("  => rich_channels EXPANDS memory capacity: environments that were")
        print("     unremembered (inert channels) become a decodable, heritable memory.")
    else:
        print("  => no capacity gain from rich_channels under this protocol.")


if __name__ == "__main__":
    main()
