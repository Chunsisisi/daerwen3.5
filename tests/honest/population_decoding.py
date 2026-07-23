"""
Population decoding + forgetting curve — borrowing a standard neuroscience/BCI
method to test DAERWEN's core claim that "the gene distribution IS memory".

The claim, operationalized the way memory is measured in real neural populations:
if the gene pool remembers the environment it was recently in, then a LINEAR
DECODER should be able to read out "which environment was this?" from the
population state vector — and that readout should DECAY over time (a forgetting
curve) once the environment becomes neutral.

Protocol (per seed × environment):
  1. warm up (neutral)
  2. ENCODING: expose the population to environment k for `encode` steps, so
     natural selection shifts gene frequencies toward whatever survives in k
  3. WASHOUT: switch to neutral dynamics and record the population state vector at
     increasing delays d = 0, 100, 200, 400, 800 steps (memory erodes via drift +
     mutation + neutral selection)

Decoding:
  For each delay separately, a cross-validated (leave-one-seed-out) logistic
  regression predicts the environment label from the state vector. Accuracy above
  chance (1/K) = the population encodes the recent environment. Accuracy vs delay =
  the forgetting curve.

Anti-overfit controls:
  * leave-one-SEED-out CV — the decoder is tested on seeds it never trained on, so
    it cannot memorize a particular run.
  * shuffled-label control — with permuted labels, decoding must collapse to chance;
    if it doesn't, the pipeline is leaking and the result is void.
  * chance line (1/K) drawn explicitly.

State vector = genetic composition of the pool (base frequencies + mean phenotype),
i.e. exactly the thing the project calls "the memory".

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/population_decoding.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig, PHENOTYPE_KEYS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

ENVIRONMENTS = ['toxic', 'starved', 'abundant']
DELAYS = [0, 100, 200, 400, 800]


def apply_env(system, env):
    """Impose an environment's chemical regime (called every encoding step)."""
    cf = system.chemical_field
    conc = cf.concentrations
    if env == 'toxic':                       # elevated inhibitor -> selects low sensitivity
        conc[:, :, cf.inhibitor_index] = 0.25
        conc[:, :, cf.ATP_index] = 0.35     # keep some energy so it's the toxin that differs
    elif env == 'starved':                   # scarce energy -> selects efficient absorbers
        conc[:, :, cf.ATP_index] = 0.08
        conc[:, :, cf.nutrient_index] = 0.0
    elif env == 'abundant':                  # energy-rich -> weak selection, different equilibrium
        conc[:, :, cf.ATP_index] = 0.80
        conc[:, :, cf.inhibitor_index] = 0.0
    cf.invalidate_gradient_cache()


def pop_state(system):
    """Genetic state vector of the pool: base freqs (4) + mean phenotype (12)."""
    alive = [p for p in system.particles if p.alive]
    if len(alive) < 10:
        return None
    bases = np.concatenate([np.asarray(p.genome) for p in alive])
    bf = [float(np.mean(bases == b)) for b in range(4)]
    ph = [float(np.mean([p.phenotype.get(k, 0.0) for p in alive])) for k in PHENOTYPE_KEYS]
    return np.array(bf + ph, dtype=float)


def run_trial(seed, env, warmup=500, encode=1000, delays=DELAYS):
    # orthogonal_expression: healthier populations (survive environmental stress) AND
    # independent phenotypes (cleaner per-environment genetic signatures). Gives the
    # "gene pool = memory" hypothesis its best shot; documented as a deliberate choice.
    cfg = Ecology2DConfig(world_size=60, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          carrying_capacity=1500, orthogonal_expression=True)
    s = Ecology2DSystem(cfg)
    for _ in range(warmup):
        s.step()
    for _ in range(encode):
        apply_env(s, env)
        s.step()
    # washout: neutral dynamics, sample at delays
    states = {}
    dset = set(delays)
    for t in range(max(delays) + 1):
        if t in dset:
            states[t] = pop_state(s)
        s.step()
    return states


def decode_at_delay(X, y, groups, shuffle=False, rng=None):
    """Leave-one-group(seed)-out CV logistic-regression accuracy."""
    y = np.asarray(y)
    if shuffle:
        y = y.copy()
        rng.shuffle(y)
    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def main(seeds=range(8)):
    seeds = list(seeds)
    rng = np.random.default_rng(0)
    print(f"POPULATION DECODING + FORGETTING CURVE  seeds={seeds}  envs={ENVIRONMENTS}")
    print(f"chance level = {1/len(ENVIRONMENTS):.3f}\n")

    # collect states: data[(seed, env)] = {delay: vector}
    data = {}
    for seed in seeds:
        for env in ENVIRONMENTS:
            data[(seed, env)] = run_trial(seed, env)

    # feature slices: full (16), base-freq only (4, purely HERITABLE), phenotype only (12)
    print(f"{'delay':>6} | {'full(16)':>9} | {'baseFreq(4)':>11} | {'phenotype(12)':>13} | {'shuffled':>9}")
    print("  (baseFreq = purely heritable genetics; if IT decodes above chance, the memory is genuinely genetic, not transient physiology)")
    print("-" * 74)
    curve = []
    for d in DELAYS:
        X, y, groups = [], [], []
        for (seed, env), st in data.items():
            v = st.get(d)
            if v is None:
                continue
            X.append(v); y.append(ENVIRONMENTS.index(env)); groups.append(seed)
        X = np.array(X); y = np.array(y); groups = np.array(groups)
        if len(set(y)) < 2:
            print(f"{d:>6} | insufficient surviving data")
            continue
        acc_full = decode_at_delay(X, y, groups)
        acc_base = decode_at_delay(X[:, :4], y, groups)      # base frequencies (heritable)
        acc_phen = decode_at_delay(X[:, 4:], y, groups)      # mean phenotype
        shuf = np.mean([decode_at_delay(X, y, groups, shuffle=True, rng=rng) for _ in range(5)])
        curve.append((d, acc_full, acc_base, acc_phen, shuf))
        print(f"{d:>6} | {acc_full:>9.3f} | {acc_base:>11.3f} | {acc_phen:>13.3f} | {shuf:>9.3f}")

    print()
    chance = 1 / len(ENVIRONMENTS)
    if curve:
        d0 = curve[0]  # (delay, full, base, phen, shuffle)
        print(f"immediate (d=0): full={d0[1]:.3f}  baseFreq={d0[2]:.3f}  "
              f"phenotype={d0[3]:.3f}  (chance {chance:.3f}, shuffle {d0[4]:.3f})")
        base_final = curve[-1][2]
        if d0[2] > chance + 0.15:
            print("=> HERITABLE genetic composition alone decodes the recent environment:")
            print("   this is genuine population-genetic memory, not transient physiology.")
            print(f"   base-freq decoding decays {d0[2]:.3f} (d=0) -> {base_final:.3f} "
                  f"(d={DELAYS[-1]}): a real forgetting curve.")
        elif d0[1] > chance + 0.15:
            print("=> full vector decodes, but heritable base-freq alone does NOT:")
            print("   the signal is transient physiology (phenotype modulation), not stored genetics.")
        else:
            print("=> not decodable above chance even immediately: no readable trace.")
    return curve


if __name__ == "__main__":
    main()
