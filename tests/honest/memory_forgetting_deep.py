"""
Task 1 + Task 3(RSA): solidify the Finding-9 memory result and test its structure.

Extends population_decoding.py to 6 environments, 10 seeds, and a denser delay grid,
then adds Representational Similarity Analysis (RSA): do *similar* environments
produce *similar* genetic states, and is that geometry consistent across seeds?

- Decoding curve on the purely HERITABLE base-frequency vector (the strict test).
- RSA: 6x6 similarity matrix of per-environment mean genetic states, plus a
  split-seeds consistency check (is the representational geometry reproducible?).

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_forgetting_deep.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig, PHENOTYPE_KEYS
from population_decoding import pop_state, decode_at_delay

# 6 chemically distinct environments (all survivable under orthogonal encoding)
ENVIRONMENTS = ['toxic', 'starved', 'abundant', 'waste_rich', 'nutrient_rich', 'harsh']
DELAYS = [0, 100, 200, 400, 800]


def apply_env(system, env):
    cf = system.chemical_field
    c = cf.concentrations
    if env == 'toxic':
        c[:, :, cf.inhibitor_index] = 0.25; c[:, :, cf.ATP_index] = 0.35
    elif env == 'starved':
        c[:, :, cf.ATP_index] = 0.08; c[:, :, cf.nutrient_index] = 0.0
    elif env == 'abundant':
        c[:, :, cf.ATP_index] = 0.80; c[:, :, cf.inhibitor_index] = 0.0
    elif env == 'waste_rich':
        c[:, :, cf.waste_index] = 1.0; c[:, :, cf.ATP_index] = 0.20
    elif env == 'nutrient_rich':
        c[:, :, cf.nutrient_index] = 1.2; c[:, :, cf.ATP_index] = 0.10
    elif env == 'harsh':
        c[:, :, cf.inhibitor_index] = 0.20; c[:, :, cf.ATP_index] = 0.10
    cf.invalidate_gradient_cache()


def run_trial(seed, env, warmup=300, encode=600, delays=DELAYS):
    cfg = Ecology2DConfig(world_size=60, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=800, orthogonal_expression=True)
    s = Ecology2DSystem(cfg)
    for _ in range(warmup):
        s.step()
    for _ in range(encode):
        apply_env(s, env); s.step()
    states, dset = {}, set(delays)
    for t in range(max(delays) + 1):
        if t in dset:
            states[t] = pop_state(s)
        s.step()
    return states


def rsa(base_states_by_env):
    """base_states_by_env[env] = list of base-freq vectors (per seed). Returns 6x6
    correlation matrix of per-env mean states, and split-half consistency r."""
    envs = ENVIRONMENTS
    means = np.array([np.mean(base_states_by_env[e], axis=0) for e in envs])
    # similarity = correlation between env mean genetic states
    sim = np.corrcoef(means)
    # split-seeds consistency: geometry from odd seeds vs even seeds
    def geom(idx):
        m = np.array([np.mean([base_states_by_env[e][i] for i in idx if i < len(base_states_by_env[e])], axis=0) for e in envs])
        return np.corrcoef(m)
    n = min(len(base_states_by_env[e]) for e in envs)
    odd = list(range(1, n, 2)); even = list(range(0, n, 2))
    g1, g2 = geom(odd), geom(even)
    iu = np.triu_indices(len(envs), k=1)
    consist = float(np.corrcoef(g1[iu], g2[iu])[0, 1])
    return sim, consist


def main(seeds=range(6)):
    seeds = list(seeds)
    rng = np.random.default_rng(0)
    chance = 1 / len(ENVIRONMENTS)
    print(f"DEEP FORGETTING + RSA  seeds={seeds}  {len(ENVIRONMENTS)} envs  chance={chance:.3f}")

    data = {}
    for si, seed in enumerate(seeds):
        for env in ENVIRONMENTS:
            data[(seed, env)] = run_trial(seed, env)
        print(f"  ...collected seed {si+1}/{len(seeds)}", flush=True)

    print(f"\n{'delay':>6} | {'baseFreq(heritable)':>19} | {'shuffled':>9}")
    print("-" * 44)
    for d in DELAYS:
        X, y, groups = [], [], []
        for (seed, env), st in data.items():
            v = st.get(d)
            if v is None:
                continue
            X.append(v[:4]); y.append(ENVIRONMENTS.index(env)); groups.append(seed)
        X = np.array(X); y = np.array(y); groups = np.array(groups)
        if len(set(y)) < 2:
            print(f"{d:>6} | insufficient data"); continue
        acc = decode_at_delay(X, y, groups)
        shuf = np.mean([decode_at_delay(X, y, groups, shuffle=True, rng=rng) for _ in range(5)])
        print(f"{d:>6} | {acc:>19.3f} | {shuf:>9.3f}")

    # RSA on d=0 heritable states
    base_by_env = {e: [] for e in ENVIRONMENTS}
    for (seed, env), st in data.items():
        v = st.get(0)
        if v is not None:
            base_by_env[env].append(v[:4])
    sim, consist = rsa(base_by_env)
    print("\nRSA — genetic-state similarity between environments (d=0, heritable):")
    print("        " + "  ".join(f"{e[:5]:>6}" for e in ENVIRONMENTS))
    for i, e in enumerate(ENVIRONMENTS):
        print(f"{e[:7]:>7} " + "  ".join(f"{sim[i,j]:>6.2f}" for j in range(len(ENVIRONMENTS))))
    print(f"\nsplit-seeds geometry consistency r = {consist:.3f} "
          f"({'reproducible structure' if consist > 0.5 else 'weak/unstable structure'})")
    return data


if __name__ == "__main__":
    main()
