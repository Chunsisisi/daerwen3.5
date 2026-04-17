"""
Video Input Experiment: feed time-varying 2D patterns as "visual" input to
DAERWEN and record full system state trajectory.

Phase 1 (this file): synthetic video — programmatic moving/pulsing patterns
Phase 2 (next):      real video from file — nature footage, traffic cams, etc.

Input pipeline:
  frame (H×W grayscale float [0,1])
    → resize to world_size × world_size
    → inject as nutrient field concentration
    → system.step() × N per frame
    → record hormones/state

Output:
  Time series of system state variables (population, energy, diversity,
  spatial distribution, etc.) saved as JSON. We then OBSERVE the patterns
  to discover what "hormones" should be — design follows observation,
  not the other way around.
"""
from __future__ import annotations
import os, sys, math, time, json
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

import numpy as np

# ── Synthetic "video" generators ─────────────────────────────────────

def gen_moving_spot(n_frames, world_size, speed=0.02):
    """Bright spot moving in a circle."""
    for i in range(n_frames):
        frame = np.zeros((world_size, world_size), dtype=np.float32)
        angle = i * speed * 2 * math.pi
        cx = world_size / 2 + world_size * 0.3 * math.cos(angle)
        cy = world_size / 2 + world_size * 0.3 * math.sin(angle)
        yy, xx = np.mgrid[:world_size, :world_size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        frame = np.exp(-dist ** 2 / (2 * (world_size * 0.08) ** 2)).astype(np.float32)
        yield frame

def gen_pulsing(n_frames, world_size, period=50):
    """Global brightness pulsing like a heartbeat."""
    for i in range(n_frames):
        intensity = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * i / period))
        frame = np.full((world_size, world_size), intensity, dtype=np.float32)
        yield frame

def gen_day_night(n_frames, world_size, day_len=80, night_len=40):
    """Sharp day/night cycle: bright then dark."""
    cycle = day_len + night_len
    for i in range(n_frames):
        phase = i % cycle
        intensity = 1.0 if phase < day_len else 0.05
        frame = np.full((world_size, world_size), intensity, dtype=np.float32)
        yield frame

def gen_two_spots(n_frames, world_size):
    """Two food spots: one stable (top-left), one blinking (bottom-right)."""
    for i in range(n_frames):
        frame = np.zeros((world_size, world_size), dtype=np.float32)
        yy, xx = np.mgrid[:world_size, :world_size]
        # Stable spot top-left
        d1 = np.sqrt((xx - world_size * 0.25) ** 2 + (yy - world_size * 0.25) ** 2)
        frame += 0.5 * np.exp(-d1 ** 2 / (2 * (world_size * 0.1) ** 2))
        # Blinking spot bottom-right (on 60 steps, off 60 steps)
        if (i // 60) % 2 == 0:
            d2 = np.sqrt((xx - world_size * 0.75) ** 2 + (yy - world_size * 0.75) ** 2)
            frame += 1.0 * np.exp(-d2 ** 2 / (2 * (world_size * 0.1) ** 2))
        yield frame

def gen_from_video_file(video_path, world_size, max_frames=300, skip=1):
    """Read a real video file and yield grayscale frames resized to world_size.
    skip=20 at 60fps → ~3fps effective, covering 100s in 300 frames."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    frame_idx = 0
    while cap.isOpened() and count < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        resized = cv2.resize(gray, (world_size, world_size))
        yield resized
        count += 1
    cap.release()


# ── Inject frame into DAERWEN ────────────────────────────────────────

def inject_frame(system, frame):
    """Replace the nutrient field with the frame's intensity pattern.
    Higher brightness = more nutrient = attracting particles."""
    cf = system.chemical_field
    nutrient = cf.concentrations[:, :, cf.nutrient_index]
    # Blend: 70% new frame + 30% old (smooth transition)
    nutrient[:] = 0.3 * nutrient + 0.7 * (frame * 1.5)
    cf.invalidate_gradient_cache()


# ── Measure system state ─────────────────────────────────────────────

def snapshot(system):
    """Minimal state snapshot for time-series recording."""
    alive = [p for p in system.particles if p.alive]
    n = len(alive)
    if n == 0:
        return {
            'alive': 0, 'avg_energy': 0, 'energy_std': 0,
            'north_frac': 0, 'east_frac': 0,
            'spatial_std_x': 0, 'spatial_std_y': 0,
            'max_gen': 0, 'reps': int(system.stats['replication_events']),
            'deaths': int(system.stats['death_events']),
        }
    positions = np.array([p.position for p in alive])
    energies = np.array([p.energy for p in alive])
    return {
        'alive': n,
        'avg_energy': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
        'north_frac': float(np.mean(positions[:, 1] > system.config.world_size / 2)),
        'east_frac': float(np.mean(positions[:, 0] > system.config.world_size / 2)),
        'spatial_std_x': float(np.std(positions[:, 0])),
        'spatial_std_y': float(np.std(positions[:, 1])),
        'max_gen': int(max(p.generation for p in alive)),
        'reps': int(system.stats['replication_events']),
        'deaths': int(system.stats['death_events']),
    }


# ── Run experiment ───────────────────────────────────────────────────

def run_experiment(video_name, frame_generator, steps_per_frame=10,
                   world_size=80, n_particles=500, seed=42):
    """Run DAERWEN with video input, record state trajectory."""
    from engine.core import Ecology2DSystem, Ecology2DConfig

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {video_name}")
    print(f"  world={world_size}  particles={n_particles}  steps/frame={steps_per_frame}")
    print(f"{'='*60}")

    np.random.seed(seed)
    cfg = Ecology2DConfig(
        world_size=world_size,
        n_particles=n_particles,
        genome_length=48,
        mutation_rate=0.01,
    )
    system = Ecology2DSystem(cfg)
    system.rng = np.random.default_rng(seed)

    trajectory = []
    frame_count = 0
    t0 = time.time()

    for frame in frame_generator:
        inject_frame(system, frame)
        for _ in range(steps_per_frame):
            system.step()
        state = snapshot(system)
        state['frame'] = frame_count
        state['step'] = system.time_step
        trajectory.append(state)

        if frame_count % 50 == 0:
            print(f"  frame {frame_count:4d}: alive={state['alive']:4d}  "
                  f"energy={state['avg_energy']:.2f}  gen={state['max_gen']:3d}  "
                  f"north={state['north_frac']:.2f}")
        frame_count += 1

    elapsed = time.time() - t0
    print(f"  Done: {frame_count} frames, {system.time_step} steps in {elapsed:.1f}s")
    return trajectory


# ── Main ─────────────────────────────────────────────────────────────

def main():
    world_size = 80

    scenarios = {
        'moving_spot': gen_moving_spot(300, world_size, speed=0.01),
        'pulsing':     gen_pulsing(300, world_size, period=50),
        'day_night':   gen_day_night(300, world_size, day_len=80, night_len=40),
        'two_spots':   gen_two_spots(300, world_size),
    }

    # Check for real video file argument
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        if video_path.exists():
            scenarios['real_video'] = gen_from_video_file(video_path, world_size, max_frames=300, skip=20)
            print(f"Added real video: {video_path}")

    all_results = {}
    for name, gen in scenarios.items():
        trajectory = run_experiment(
            name, gen,
            steps_per_frame=10, world_size=world_size,
            n_particles=500, seed=42,
        )
        all_results[name] = trajectory

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: final state per scenario")
    print(f"{'='*60}")
    for name, traj in all_results.items():
        final = traj[-1] if traj else {}
        print(f"  {name:<15}: alive={final.get('alive',0):4d}  "
              f"gen={final.get('max_gen',0):3d}  "
              f"north={final.get('north_frac',0):.2f}  "
              f"east={final.get('east_frac',0):.2f}")

    # Save
    out_dir = Path(_root) / 'benchmark_results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'video_input_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
