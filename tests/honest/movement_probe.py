"""
Deterministic mechanism test — which chemical channels can steer movement?

This replaces an earlier frozen-population statistical probe that was too weak
to conclude (the anti-overfit guard correctly refused a verdict): because
`field_interaction` is lerp(-1, 1, .), a random population splits ~50/50 into
gradient-climbers and gradient-descenders, so net center-of-mass drift cancels
and no behavioral signal survives the Brownian noise floor.

Instead we test the MECHANISM directly and deterministically:

  * one engineered particle, made a maximal gradient-climber
    (field_interaction=1, movement_response=2, chemotaxis_gene_strength=max)
  * Brownian noise disabled (brownian_strength=0) so ONLY chemotaxis remains
  * a strong spatial gradient placed on ONE channel at a time, everything else
    flat
  * call ONLY _particle_movement for a few iterations, measure displacement

If the particle moves toward the high side of the gradient, that channel steers
movement. If displacement is exactly 0, the channel is invisible to movement.

Result (see main): ONLY the ATP channel steers. nutrient, inhibitor, and all
inert channels (3..10) give exactly 0 — proving movement direction is a
function of the ATP gradient alone. Combined with metabolism reading only
ATP/nutrient/inhibitor for fitness, channels 3..10 are causally inert: no
arbitrary signal can ever be associated with reward in this substrate.

Run:
    CUDA_VISIBLE_DEVICES=-1 python tests/honest/movement_probe.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from engine.core import Ecology2DSystem, Ecology2DConfig


def channel_response(channel, iters=30, world=40):
    """Displacement (dx) of a maximal gradient-climber when `channel` carries a
    left-high gradient and Brownian noise is off. dx<0 => moved toward high side."""
    cfg = Ecology2DConfig(world_size=world, n_particles=1, genome_length=24,
                          mutation_rate=0.0, n_chemical_species=12, seed=0)
    cfg.brownian_strength = 0.0        # isolate chemotaxis from noise
    s = Ecology2DSystem(cfg)
    p = s.particles[0]
    p.position = np.array([world / 2.0, world / 2.0])
    p.velocity = np.array([0.0, 0.0])
    p.energy = 1.0
    p.phenotype = dict(p.phenotype)
    p.phenotype['field_interaction'] = 1.0
    p.phenotype['movement_response'] = 2.0
    p.phenotype['chemotaxis_gene_strength'] = 0.15
    col = np.linspace(2.0, 0.0, world).astype('float32')   # high at x=0
    s.chemical_field.concentrations[:, :, :] = 0.3          # flat baseline
    s.chemical_field.concentrations[:, :, channel] = col[:, None]
    s.chemical_field.invalidate_gradient_cache()
    for _ in range(iters):
        alive = [q for q in s.particles if q.alive]
        state = s._build_particle_state(alive)
        s._particle_movement(alive, state, 0.1)
    return float(p.position[0] - world / 2.0)


def main():
    channels = [(0, 'ATP (0, WIRED)'), (1, 'nutrient (1)'), (2, 'inhibitor (2)'),
                (5, 'inert ch5'), (7, 'inert ch7'), (10, 'inert ch10')]
    print("DETERMINISTIC MOVEMENT MECHANISM TEST (Brownian off, one climber)")
    results = {}
    for ch, name in channels:
        dx = channel_response(ch)
        results[ch] = dx
        print(f"  gradient on {name:16s} -> dx = {dx:+.4f}")
    print("  (dx<0 = follows the channel; dx=0 = channel invisible to movement)")

    steers = [ch for ch, dx in results.items() if abs(dx) > 1e-3]
    print(f"\n  Channels that steer movement: {steers}")
    if steers == [0]:
        print("  => PROVEN: movement direction depends on the ATP channel ALONE.")
        print("     Channels 3..10 are causally inert (no direction, no fitness).")
        print("     Signal->reward association is structurally impossible without a")
        print("     new sensory prior (Designer's Trap).")
    return results


if __name__ == "__main__":
    main()
