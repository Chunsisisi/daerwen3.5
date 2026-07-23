"""
Can the substrate LEARN a signal->reward association? (evolvable_sensing)

Background: with the default engine, movement follows the ATP gradient alone, so
an arbitrary signal can never be associated with reward (proved in
movement_probe.py). We added an opt-in `evolvable_sensing` pathway: an
independently-encoded `signal_affinity` gene lets movement also respond to one
designated signal channel. This experiment asks whether EVOLUTION can discover
and use it.

Design (reward is reachable ONLY via the signal):
  * signal_channel (4) carries a gradient, HIGH on the LEFT, every step.
  * ATP is held FLAT (re-imposed each step) -> ATP-chemotaxis gives no direction
    and no spatial energy cue. field_interaction cannot locate the reward.
  * the REWARD is nutrient (channel 1, an energy source that movement does NOT
    follow). Placing nutrient on one side makes that side reproductively better,
    but the only way to *steer* there is to follow the correlated signal.

Three conditions:
  assoc_left : evolvable, reward(nutrient) on LEFT  (correlated with signal)
               -> to get reward, follow signal -> signal_affinity should go +.
  assoc_right: evolvable, reward on RIGHT (anti-correlated with signal)
               -> to get reward, move AWAY from signal -> signal_affinity should go -.
  sensing_off: reward on LEFT but evolvable_sensing OFF (signal unusable) -> the
               signal cannot help; control for "left-clustering from something else".

ANTI-OVERFIT LOGIC: signal channel 4 is otherwise causally inert, so the ONLY
reason signal_affinity would shift is the reward contingency. The decisive
signature is a SIGN FLIP: median signal_affinity(assoc_left) > 0 >
median signal_affinity(assoc_right). A fixed bias (both same sign) would mean the
result is an artifact, not learning. If both ~0, the pathway exists but evolution
did not find it (honest negative result).

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/association_experiment.py
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
SIGNAL_CH = 4
NUTRIENT_CH = 1
ATP_CH = 0


def bump(system, ch, peak_x, hi, lo):
    """Smooth PERIODIC field on `ch`: peak `hi` at x=peak_x, min `lo` opposite.
    Periodic (cos) so there is no boundary discontinuity under np.mod wrapping —
    the earlier linear ramp had a seam that corrupted the spatial readout."""
    w = system.config.world_size
    x = np.arange(w)
    val = lo + (hi - lo) * 0.5 * (1.0 + np.cos(2 * np.pi * (x - peak_x) / w))
    system.chemical_field.concentrations[:, :, ch] = val.astype('float32')[:, None]


def occupancy(system, peak_x):
    """Fraction of living particles within circular x-distance < world/6 of peak_x."""
    a = [p for p in system.particles if p.alive]
    if not a:
        return None
    w = system.config.world_size
    xs = np.array([p.position[0] for p in a])
    d = np.abs(xs - peak_x)
    d = np.minimum(d, w - d)          # circular distance
    return float(np.mean(d < w / 6))


def flat(system, ch, val):
    system.chemical_field.concentrations[:, :, ch] = val


def cull_to_cap(system, cap):
    """Impose a carrying capacity the engine lacks: randomly kill excess living
    particles down to `cap`. Random => unbiased w.r.t. genes, so gene-frequency
    evolution (incl. signal_affinity) is unaffected; this only bounds runtime."""
    alive = [p for p in system.particles if p.alive]
    if len(alive) <= cap:
        return
    n_kill = len(alive) - cap
    victims = system.rng.choice(len(alive), size=n_kill, replace=False)
    for i in np.atleast_1d(victims):
        alive[int(i)].alive = False


def mean_signal_affinity(system):
    a = [p for p in system.particles if p.alive]
    if not a:
        return None
    return float(np.mean([p.phenotype.get('signal_affinity', 0.0) for p in a]))


def left_fraction(system):
    a = [p for p in system.particles if p.alive]
    if not a:
        return None
    return float(np.mean([p.position[0] < system.config.world_size / 3 for p in a]))


def run_condition(seed, condition, train_steps=1500, warmup=200):
    evolvable = condition != 'sensing_off'
    reward_left = condition != 'assoc_right'
    cfg = Ecology2DConfig(world_size=WORLD, n_particles=300, genome_length=24,
                          mutation_rate=0.02, n_chemical_species=12, seed=seed,
                          evolvable_sensing=evolvable, signal_channel=SIGNAL_CH)
    cfg.brownian_strength = 0.05        # moderate signal-following leverage
    s = Ecology2DSystem(cfg)
    for _ in range(warmup):
        s.step()
    w = s.config.world_size
    signal_peak = 0                       # signal bump always at x=0
    reward_peak = signal_peak if reward_left else w // 2   # co-located vs opposite
    sa0 = mean_signal_affinity(s)
    peak_alive = 0
    cap = 1200                            # externally imposed carrying capacity (engine lacks one)
    for _ in range(train_steps):
        bump(s, SIGNAL_CH, signal_peak, hi=2.0, lo=0.02)   # the cue
        flat(s, ATP_CH, 0.03)                              # ATP near-starvation: reward is survival-critical
        bump(s, NUTRIENT_CH, reward_peak, hi=1.6, lo=0.0)  # the reward (energy), movement can't follow it
        s.step()
        cull_to_cap(s, cap)
        n = sum(1 for p in s.particles if p.alive)
        peak_alive = max(peak_alive, n)
        if n == 0:
            break
    return {
        'condition': condition, 'seed': seed,
        'sa_before': sa0, 'sa_after': mean_signal_affinity(s),
        'reward_occupancy': occupancy(s, reward_peak),     # are they where the reward is?
        'signal_occupancy': occupancy(s, signal_peak),     # are they where the signal is?
        'alive': sum(1 for p in s.particles if p.alive),
        'peak_alive': peak_alive,
    }


def summarize(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None
    return {'n': int(a.size), 'median': float(np.median(a)),
            'q25': float(np.percentile(a, 25)), 'q75': float(np.percentile(a, 75))}


def main():
    seeds = [0, 1, 2, 3]
    conditions = ['assoc_left', 'assoc_right', 'sensing_off']
    print("SIGNAL->REWARD ASSOCIATION EXPERIMENT  seeds=", seeds)
    print("(signal high-LEFT always; ATP flat; reward=nutrient on rewarded side)\n")
    results = {c: [] for c in conditions}
    for c in conditions:
        for seed in seeds:
            r = run_condition(seed, c)
            results[c].append(r)
            print(f"  {c:12s} seed{seed}  sa {r['sa_before']:+.3f}->{r['sa_after']:+.3f}  "
                  f"reward_occ={r['reward_occupancy']:.2f} signal_occ={r['signal_occupancy']:.2f}  "
                  f"alive={r['alive']}")

    print("\n  === signal_affinity after training (median, IQR) ===")
    summ = {}
    for c in conditions:
        s = summarize([r['sa_after'] for r in results[c]])
        summ[c] = s
        print(f"  {c:12s}: {('median=%+.3f IQR[%+.3f,%+.3f]'%(s['median'],s['q25'],s['q75'])) if s else 'no data'}")

    print("\n  VERDICT:")
    L, R = summ.get('assoc_left'), summ.get('assoc_right')
    if L and R:
        if L['median'] > 0.1 and R['median'] < -0.1:
            print("  SIGN FLIP: signal_affinity goes + when reward tracks the signal and")
            print("  - when reward is anti-correlated. => genuine learned signal-reward")
            print("  association (the capability was impossible before evolvable_sensing).")
        elif abs(L['median']) < 0.1 and abs(R['median']) < 0.1:
            print("  Both ~0: pathway exists but evolution did NOT discover it in this")
            print("  regime. Honest negative result — try longer training / stronger reward.")
        else:
            print("  No clean sign flip; inconclusive / possible confound — investigate.")
    return results, summ


if __name__ == "__main__":
    main()
