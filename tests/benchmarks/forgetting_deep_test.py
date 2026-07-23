"""
Deep Forgetting Test — is DAERWEN's 0.010 forgetting rate REAL or artifact?

Five tests, each attacking the claim from a different angle:

Test 1: CONTROL — does a non-evolving system also "not forget"?
    If mutation_rate=0 (no evolution) also scores low forgetting → our result
    is just physics, not learning.

Test 2: RETENTION CURVE — learn task A, then pile on 5/10/20 distractor tasks,
    re-test task A. How does retention decay with number of distractors?

Test 3: PAIRED FORGETTING — same initial conditions:
    (a) learn A → 1000 steps nothing → re-test A
    (b) learn A → 1000 steps of B → re-test A
    Difference = true interference-based forgetting.

Test 4: TASK ORDER — does order matter?
    (a) tasks: A → B → C → D → E, test all
    (b) tasks: E → D → C → B → A, test all
    If order doesn't matter → system isn't really learning sequences.

Test 5: LONG SEQUENCE — 20 tasks instead of 10. Does forgetting get worse?

All tests use multiple seeds for statistical significance.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _root)

import numpy as np


# ── Task library (gradient directions) ──────────────────────────────

def apply_gradient(system, direction):
    """Apply a spatial nutrient gradient. Returns alignment score."""
    world = system.config.world_size
    x = np.arange(world)
    xx, yy = np.meshgrid(x, x)

    gradients = {
        'north':    yy / world * 2.0,
        'south':    (world - yy) / world * 2.0,
        'east':     xx / world * 2.0,
        'west':     (world - xx) / world * 2.0,
        'ne':       (xx + yy) / (2 * world) * 2.0,
        'nw':       ((world - xx) + yy) / (2 * world) * 2.0,
        'se':       (xx + (world - yy)) / (2 * world) * 2.0,
        'sw':       ((world - xx) + (world - yy)) / (2 * world) * 2.0,
        'checkerboard': np.where((xx // 20 + yy // 20) % 2 == 0, 2.0, 0.5),
        'wave':     1.0 + np.sin(2 * np.pi * xx / world) * 0.5,
        'top_stripe': np.where(yy >= world * 0.75, 2.0, 0.3),
        'left_stripe': np.where(xx <= world * 0.25, 2.0, 0.3),
        'boundary': np.sqrt((xx - world//2)**2 + (yy - world//2)**2) / (world * 0.7) * 2.0,
        'center':   2.0 * np.exp(-((xx-world//2)**2 + (yy-world//2)**2) / (2*(world/4)**2)),
        'ring':     np.where(np.abs(np.sqrt((xx-world//2)**2+(yy-world//2)**2) - world*0.3) < world*0.08, 2.0, 0.3),
        'corners':  2.0 * np.exp(-np.minimum(np.minimum(xx**2+yy**2, (xx-world)**2+yy**2),
                                              np.minimum(xx**2+(yy-world)**2, (xx-world)**2+(yy-world)**2)) / (2*(world/6)**2)),
        'diagonal': (xx + yy) / (2 * world) * 2.0,
        'anti_diag': ((world - xx) + yy) / (2 * world) * 2.0,
        'h_stripes': np.where((yy // 15) % 2 == 0, 2.0, 0.3),
        'v_stripes': np.where((xx // 15) % 2 == 0, 2.0, 0.3),
    }

    gradient = gradients[direction].astype(np.float32)
    cf = system.chemical_field
    cf.concentrations[:, :, cf.nutrient_index] = gradient
    cf.invalidate_gradient_cache()


def measure_alignment(system, direction):
    """How well does population distribution match a gradient?"""
    alive = [p for p in system.particles if p.alive]
    if len(alive) < 5:
        return 0.0
    world = system.config.world_size
    positions = np.array([p.position for p in alive])
    x = np.arange(world)
    xx, yy = np.meshgrid(x, x)

    gradients = {
        'north': yy / world, 'south': (world-yy)/world,
        'east': xx/world, 'west': (world-xx)/world,
        'ne': (xx+yy)/(2*world), 'nw': ((world-xx)+yy)/(2*world),
        'se': (xx+(world-yy))/(2*world), 'sw': ((world-xx)+(world-yy))/(2*world),
        'checkerboard': np.where((xx//20+yy//20)%2==0, 1.0, 0.0),
        'wave': 0.5+np.sin(2*np.pi*xx/world)*0.25,
        'top_stripe': np.where(yy>=world*0.75, 1.0, 0.0),
        'left_stripe': np.where(xx<=world*0.25, 1.0, 0.0),
        'diagonal': (xx+yy)/(2*world),
        'anti_diag': ((world-xx)+yy)/(2*world),
        'boundary': np.sqrt((xx-world//2)**2+(yy-world//2)**2)/(world*0.7),
        'center': np.exp(-((xx-world//2)**2+(yy-world//2)**2)/(2*(world/4)**2)),
        'ring': (np.abs(np.sqrt((xx-world//2)**2+(yy-world//2)**2)-world*0.3)<world*0.08).astype(float),
        'corners': np.exp(-np.minimum(np.minimum(xx**2+yy**2,(xx-world)**2+yy**2),
                                       np.minimum(xx**2+(yy-world)**2,(xx-world)**2+(yy-world)**2))/(2*(world/6)**2)),
        'h_stripes': ((yy//15)%2==0).astype(float),
        'v_stripes': ((xx//15)%2==0).astype(float),
    }
    grad = gradients[direction].astype(np.float32)
    grad_norm = grad / (grad.max() + 1e-9)

    # Compute alignment: correlation between particle density and gradient
    bins = 10
    hist, _, _ = np.histogram2d(positions[:,0], positions[:,1],
                                 bins=bins, range=[[0,world],[0,world]])
    hist = hist / (hist.sum() + 1e-9)
    # Downsample gradient to same bins
    from scipy.ndimage import zoom
    grad_small = zoom(grad_norm, bins/world)[:bins,:bins]
    grad_small = grad_small / (grad_small.sum() + 1e-9)
    return float(np.corrcoef(hist.flatten(), grad_small.flatten())[0, 1])


def create_system(seed, world_size=80, n_particles=300, mutation_rate=0.01):
    import engine.chem_sim_genes as gm
    gm._USING_RUST = False
    from engine.core import Ecology2DSystem, Ecology2DConfig
    np.random.seed(seed)
    cfg = Ecology2DConfig(
        world_size=world_size, n_particles=n_particles,
        genome_length=48, mutation_rate=mutation_rate,
    )
    system = Ecology2DSystem(cfg)
    system.rng = np.random.default_rng(seed)
    return system


def train_on_task(system, task, steps=200):
    apply_gradient(system, task)
    for _ in range(steps):
        system.step()


def test_on_task(system, task, steps=50):
    apply_gradient(system, task)
    scores = []
    for _ in range(steps):
        system.step()
        scores.append(measure_alignment(system, task))
    return float(np.mean(scores)) if scores else 0.0


# ── Test 1: CONTROL (no evolution) ──────────────────────────────────

def _test1_worker(args):
    mode, seed = args
    import io; sys.stdout = io.StringIO()
    mu = 0.01 if mode == 'evolving' else 0.0
    system = create_system(seed, mutation_rate=mu)
    tasks = ['north', 'east', 'south', 'west', 'diagonal']
    # Train on all 5 tasks
    for t in tasks:
        train_on_task(system, t, steps=200)
    # Test retention on all 5
    scores = {}
    for t in tasks:
        scores[t] = test_on_task(system, t, steps=30)
    sys.stdout = sys.__stdout__
    avg_score = float(np.mean(list(scores.values())))
    return mode, seed, avg_score, scores


def test1_control():
    print("\n" + "="*70)
    print("TEST 1: CONTROL — does non-evolving system also 'not forget'?")
    print("="*70)
    tasks_list = [('evolving', s) for s in range(5)] + [('frozen', s) for s in range(5)]
    with Pool(5) as pool:
        results = pool.map(_test1_worker, tasks_list)
    for mode in ['evolving', 'frozen']:
        rs = [r for r in results if r[0] == mode]
        avg = np.mean([r[2] for r in rs])
        std = np.std([r[2] for r in rs])
        print(f"  {mode:>10}: avg_retention = {avg:.4f} ± {std:.4f}")
    print("  If frozen ≈ evolving → forgetting metric is trivial (just physics)")


# ── Test 2: RETENTION CURVE ─────────────────────────────────────────

def _test2_worker(args):
    n_distractors, seed = args
    import io; sys.stdout = io.StringIO()
    system = create_system(seed)
    target = 'north'
    distractors = ['east','south','west','diagonal','ne','nw','se','sw',
                   'checkerboard','wave','top_stripe','left_stripe',
                   'boundary','center','h_stripes','v_stripes',
                   'anti_diag','ring','corners','v_stripes'][:n_distractors]
    # Train on target
    train_on_task(system, target, steps=300)
    baseline = test_on_task(system, target, steps=30)
    # Pile on distractors
    for d in distractors:
        train_on_task(system, d, steps=200)
    # Re-test target
    retention = test_on_task(system, target, steps=30)
    sys.stdout = sys.__stdout__
    forgetting = max(0, baseline - retention)
    return n_distractors, seed, baseline, retention, forgetting


def test2_retention_curve():
    print("\n" + "="*70)
    print("TEST 2: RETENTION CURVE — how does retention decay with distractors?")
    print("="*70)
    tasks_list = [(n, s) for n in [0, 2, 5, 10, 15, 20] for s in range(3)]
    with Pool(5) as pool:
        results = pool.map(_test2_worker, tasks_list)
    print(f"  {'distractors':>12} {'baseline':>10} {'retention':>10} {'forgetting':>10}")
    print("  " + "-"*48)
    for n in [0, 2, 5, 10, 15, 20]:
        rs = [r for r in results if r[0] == n]
        bl = np.mean([r[2] for r in rs])
        rt = np.mean([r[3] for r in rs])
        fg = np.mean([r[4] for r in rs])
        print(f"  {n:>12} {bl:>10.4f} {rt:>10.4f} {fg:>10.4f}")
    print("  If forgetting increases with distractors → real forgetting")
    print("  If flat → system doesn't really learn tasks (just physics)")


# ── Test 3: PAIRED FORGETTING ───────────────────────────────────────

def _test3_worker(args):
    condition, seed = args
    import io; sys.stdout = io.StringIO()
    system = create_system(seed)
    # Learn task A
    train_on_task(system, 'north', steps=300)
    baseline_a = test_on_task(system, 'north', steps=30)
    # Condition: idle vs distractor
    if condition == 'idle':
        for _ in range(1000):
            system.step()
    elif condition == 'distractor':
        train_on_task(system, 'south', steps=1000)
    # Re-test A
    retention_a = test_on_task(system, 'north', steps=30)
    sys.stdout = sys.__stdout__
    return condition, seed, baseline_a, retention_a


def test3_paired():
    print("\n" + "="*70)
    print("TEST 3: PAIRED — idle vs distractor interference")
    print("="*70)
    tasks_list = [(c, s) for c in ['idle', 'distractor'] for s in range(5)]
    with Pool(5) as pool:
        results = pool.map(_test3_worker, tasks_list)
    for cond in ['idle', 'distractor']:
        rs = [r for r in results if r[0] == cond]
        bl = np.mean([r[2] for r in rs])
        rt = np.mean([r[3] for r in rs])
        print(f"  {cond:>12}: baseline={bl:.4f}  retention={rt:.4f}  forgetting={bl-rt:+.4f}")
    print("  If distractor forgetting >> idle forgetting → real interference")
    print("  If same → 'forgetting' is just temporal decay, not task interference")


# ── Test 4: TASK ORDER ──────────────────────────────────────────────

def _test4_worker(args):
    order_name, task_order, seed = args
    import io; sys.stdout = io.StringIO()
    system = create_system(seed)
    for t in task_order:
        train_on_task(system, t, steps=200)
    scores = {}
    for t in task_order:
        scores[t] = test_on_task(system, t, steps=30)
    sys.stdout = sys.__stdout__
    return order_name, seed, scores


def test4_order():
    print("\n" + "="*70)
    print("TEST 4: TASK ORDER — does sequence matter?")
    print("="*70)
    forward = ['north', 'east', 'south', 'west', 'diagonal']
    reverse = list(reversed(forward))
    tasks_list = [(n, o, s) for n, o in [('forward', forward), ('reverse', reverse)]
                  for s in range(5)]
    with Pool(5) as pool:
        results = pool.map(_test4_worker, tasks_list)
    for order_name in ['forward', 'reverse']:
        rs = [r for r in results if r[0] == order_name]
        all_scores = {t: np.mean([r[2][t] for r in rs]) for t in forward}
        last_task = forward[-1] if order_name == 'forward' else forward[0]
        first_task = forward[0] if order_name == 'forward' else forward[-1]
        print(f"  {order_name:>8}: first_task_retention={all_scores[first_task]:.4f}  "
              f"last_task_retention={all_scores[last_task]:.4f}  "
              f"avg={np.mean(list(all_scores.values())):.4f}")
    print("  If last_task >> first_task → recency bias (real sequential learning)")
    print("  If equal → no recency effect (not really learning in order)")


# ── Test 5: LONG SEQUENCE ───────────────────────────────────────────

def _test5_worker(args):
    n_tasks, seed = args
    import io; sys.stdout = io.StringIO()
    system = create_system(seed)
    all_tasks = ['north','east','south','west','diagonal','ne','nw','se','sw',
                 'checkerboard','wave','top_stripe','left_stripe','boundary',
                 'center','h_stripes','v_stripes','anti_diag','ring','corners'][:n_tasks]
    for t in all_tasks:
        train_on_task(system, t, steps=200)
    first_score = test_on_task(system, all_tasks[0], steps=30)
    last_score = test_on_task(system, all_tasks[-1], steps=30)
    avg_score = np.mean([test_on_task(system, t, steps=20) for t in all_tasks])
    sys.stdout = sys.__stdout__
    return n_tasks, seed, first_score, last_score, avg_score


def test5_long():
    print("\n" + "="*70)
    print("TEST 5: LONG SEQUENCE — does forgetting get worse with more tasks?")
    print("="*70)
    tasks_list = [(n, s) for n in [5, 10, 15, 20] for s in range(3)]
    with Pool(5) as pool:
        results = pool.map(_test5_worker, tasks_list)
    print(f"  {'n_tasks':>8} {'first_retention':>16} {'last_retention':>16} {'avg_retention':>16}")
    print("  " + "-"*60)
    for n in [5, 10, 15, 20]:
        rs = [r for r in results if r[0] == n]
        first = np.mean([r[2] for r in rs])
        last = np.mean([r[3] for r in rs])
        avg = np.mean([r[4] for r in rs])
        print(f"  {n:>8} {first:>16.4f} {last:>16.4f} {avg:>16.4f}")
    print("  If first_retention drops with n_tasks → forgetting scales with load")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("DEEP FORGETTING VALIDATION")
    print("Is DAERWEN's 0.010 forgetting rate real or artifact?")
    print(f"Running 5 tests with multiprocessing...\n")

    t0 = time.time()
    test1_control()
    test2_retention_curve()
    test3_paired()
    test4_order()
    test5_long()
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"ALL 5 TESTS COMPLETE in {elapsed:.1f}s")
    print(f"{'='*70}")
