"""
Intensive Forgetting Test — high training, large population, cleaner metric.

Previous test showed alignment scores near zero → "nothing to forget."
This version fixes three issues:
  1. Training: 200 steps → 2000 steps (10x more evolution time)
  2. Population: 300 → 800 particles (more statistical power)
  3. Metric: correlation → fraction in correct region (less noisy)

Same 5 tests, same questions, better signal.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _root)

import numpy as np

# ── Config ──────────────────────────────────────────────────────────
WORLD_SIZE = 80
N_PARTICLES = 800
TRAIN_STEPS = 2000   # 10x previous
TEST_STEPS = 100
N_SEEDS = 5

# ── Simpler, less noisy metric ──────────────────────────────────────

def apply_gradient(system, direction):
    world = system.config.world_size
    x = np.arange(world, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    grads = {
        'north': yy / world * 2.0,
        'south': (world - yy) / world * 2.0,
        'east':  xx / world * 2.0,
        'west':  (world - xx) / world * 2.0,
        'ne':    (xx + yy) / (2 * world) * 2.0,
        'nw':    ((world-xx) + yy) / (2*world) * 2.0,
        'se':    (xx + (world-yy)) / (2*world) * 2.0,
        'sw':    ((world-xx) + (world-yy)) / (2*world) * 2.0,
        'top_stripe': np.where(yy >= world*0.7, 2.0, 0.3),
        'bottom_stripe': np.where(yy <= world*0.3, 2.0, 0.3),
    }
    cf = system.chemical_field
    cf.concentrations[:, :, cf.nutrient_index] = grads[direction]
    cf.invalidate_gradient_cache()


def measure_score(system, direction):
    """Fraction of particles in the 'correct' region for each gradient.
    Random = 0.5. Perfect = 1.0. Completely wrong = 0.0."""
    alive = [p for p in system.particles if p.alive]
    if len(alive) < 10:
        return 0.5  # not enough data
    positions = np.array([p.position for p in alive])
    w = system.config.world_size
    half = w / 2.0

    scores = {
        'north': np.mean(positions[:, 1] > half),
        'south': np.mean(positions[:, 1] < half),
        'east':  np.mean(positions[:, 0] > half),
        'west':  np.mean(positions[:, 0] < half),
        'ne':    np.mean((positions[:, 0] > half) & (positions[:, 1] > half)) * 2,
        'nw':    np.mean((positions[:, 0] < half) & (positions[:, 1] > half)) * 2,
        'se':    np.mean((positions[:, 0] > half) & (positions[:, 1] < half)) * 2,
        'sw':    np.mean((positions[:, 0] < half) & (positions[:, 1] < half)) * 2,
        'top_stripe': np.mean(positions[:, 1] > w * 0.7) / 0.3,
        'bottom_stripe': np.mean(positions[:, 1] < w * 0.3) / 0.3,
    }
    return float(np.clip(scores.get(direction, 0.5), 0, 1))


def create_system(seed, mutation_rate=0.01):
    import engine.chem_sim_genes as gm
    gm._USING_RUST = False
    from engine.core import Ecology2DSystem, Ecology2DConfig
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    np.random.seed(seed)
    cfg = Ecology2DConfig(
        world_size=WORLD_SIZE, n_particles=N_PARTICLES,
        genome_length=48, mutation_rate=mutation_rate,
    )
    system = Ecology2DSystem(cfg)
    system.rng = np.random.default_rng(seed)
    sys.stdout = old_stdout
    return system


def train(system, task, steps=TRAIN_STEPS):
    apply_gradient(system, task)
    for _ in range(steps):
        system.step()


def test(system, task, steps=TEST_STEPS):
    apply_gradient(system, task)
    scores = []
    for _ in range(steps):
        system.step()
        scores.append(measure_score(system, task))
    return float(np.mean(scores))


# ── TEST 1: CONTROL ─────────────────────────────────────────────────

def test1():
    print("\n" + "="*70)
    print("TEST 1: CONTROL — evolving vs frozen (2000 steps training)")
    print("="*70)
    results = {}
    for mode in ['evolving', 'frozen']:
        mu = 0.01 if mode == 'evolving' else 0.0
        scores_list = []
        for seed in range(N_SEEDS):
            system = create_system(seed, mutation_rate=mu)
            tasks = ['north', 'east', 'south', 'west']
            for t in tasks:
                train(system, t)
            task_scores = [test(system, t) for t in tasks]
            scores_list.append(np.mean(task_scores))
        m, s = np.mean(scores_list), np.std(scores_list)
        results[mode] = (m, s)
        print(f"  {mode:>10}: retention = {m:.4f} ± {s:.4f}  (random=0.50)")
    diff = results['evolving'][0] - results['frozen'][0]
    print(f"  difference (evolving - frozen) = {diff:+.4f}")
    print(f"  → {'EVOLVING LEARNS MORE' if diff > 0.02 else 'NO SIGNIFICANT DIFFERENCE'}")


# ── TEST 2: RETENTION CURVE ─────────────────────────────────────────

def test2():
    print("\n" + "="*70)
    print("TEST 2: RETENTION CURVE (2000 steps/task, 800 particles)")
    print("="*70)
    print(f"  {'distractors':>12} {'baseline':>10} {'after':>10} {'forgetting':>10} {'alive':>8}")
    print("  " + "-"*55)
    for n_dist in [0, 3, 6, 10]:
        baselines, afters, forgets, alives = [], [], [], []
        for seed in range(N_SEEDS):
            system = create_system(seed)
            train(system, 'north')
            bl = test(system, 'north', steps=50)
            distractors = ['east','south','west','ne','nw','se','sw','top_stripe','bottom_stripe','east'][:n_dist]
            for d in distractors:
                train(system, d)
            af = test(system, 'north', steps=50)
            alive = sum(1 for p in system.particles if p.alive)
            baselines.append(bl); afters.append(af)
            forgets.append(max(0, bl - af)); alives.append(alive)
        print(f"  {n_dist:>12} {np.mean(baselines):>10.4f} {np.mean(afters):>10.4f} "
              f"{np.mean(forgets):>10.4f} {int(np.mean(alives)):>8}")
    print("  random baseline = 0.50")


# ── TEST 3: PAIRED (idle vs interference) ───────────────────────────

def test3():
    print("\n" + "="*70)
    print("TEST 3: PAIRED — idle vs distractor (2000 steps each)")
    print("="*70)
    for cond in ['idle', 'distractor']:
        bls, rets = [], []
        for seed in range(N_SEEDS):
            system = create_system(seed)
            train(system, 'north', steps=TRAIN_STEPS)
            bl = test(system, 'north', steps=50)
            if cond == 'idle':
                for _ in range(TRAIN_STEPS):
                    system.step()
            else:
                train(system, 'south', steps=TRAIN_STEPS)
            ret = test(system, 'north', steps=50)
            bls.append(bl); rets.append(ret)
        fg = np.mean(bls) - np.mean(rets)
        print(f"  {cond:>12}: baseline={np.mean(bls):.4f}  after={np.mean(rets):.4f}  "
              f"forgetting={fg:+.4f}")
    print("  distractor_forgetting > idle_forgetting → real interference")


# ── TEST 4: ORDER ───────────────────────────────────────────────────

def test4():
    print("\n" + "="*70)
    print("TEST 4: ORDER — recency bias test")
    print("="*70)
    tasks = ['north', 'east', 'south', 'west']
    for order_name, order in [('forward', tasks), ('reverse', list(reversed(tasks)))]:
        all_scores = {t: [] for t in tasks}
        for seed in range(N_SEEDS):
            system = create_system(seed)
            for t in order:
                train(system, t)
            for t in tasks:
                all_scores[t].append(test(system, t, steps=50))
        first = order[0]
        last = order[-1]
        print(f"  {order_name:>8}: trained_first({first})={np.mean(all_scores[first]):.4f}  "
              f"trained_last({last})={np.mean(all_scores[last]):.4f}  "
              f"diff={np.mean(all_scores[last])-np.mean(all_scores[first]):+.4f}")
    print("  positive diff → recency bias (last learned = better retained)")


# ── TEST 5: SCALING ─────────────────────────────────────────────────

def test5():
    print("\n" + "="*70)
    print("TEST 5: SCALING — first task retention vs total tasks")
    print("="*70)
    all_tasks = ['north','east','south','west','ne','nw','se','sw','top_stripe','bottom_stripe']
    print(f"  {'n_tasks':>8} {'first_ret':>10} {'last_ret':>10} {'avg_ret':>10} {'alive':>8}")
    print("  " + "-"*50)
    for n in [2, 4, 6, 8, 10]:
        firsts, lasts, avgs, al = [], [], [], []
        for seed in range(N_SEEDS):
            system = create_system(seed)
            task_list = all_tasks[:n]
            for t in task_list:
                train(system, t)
            first_score = test(system, task_list[0], steps=50)
            last_score = test(system, task_list[-1], steps=50)
            avg_score = np.mean([test(system, t, steps=30) for t in task_list])
            alive = sum(1 for p in system.particles if p.alive)
            firsts.append(first_score); lasts.append(last_score)
            avgs.append(avg_score); al.append(alive)
        print(f"  {n:>8} {np.mean(firsts):>10.4f} {np.mean(lasts):>10.4f} "
              f"{np.mean(avgs):>10.4f} {int(np.mean(al)):>8}")
    print("  random = 0.50")


# ── MAIN ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("INTENSIVE FORGETTING VALIDATION")
    print(f"  N_PARTICLES={N_PARTICLES}, TRAIN_STEPS={TRAIN_STEPS}, N_SEEDS={N_SEEDS}")
    print(f"  Metric: fraction of particles in correct region (random=0.50)")
    t0 = time.time()

    test1()
    test2()
    test3()
    test4()
    test5()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL TESTS DONE in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
