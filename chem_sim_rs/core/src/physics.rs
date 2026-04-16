//! Force integration: brownian noise + spring-on-bonds + soft repulsion.
//!
//! All forces use toroidal-space distance via `spatial::delta_toroidal`.

use crate::atom::Atom;
use crate::spatial::{delta_toroidal, dist_sq_toroidal, SpatialHash};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct PhysicsConfig {
    pub world_size:    f32,
    pub r_spring:      f32,  // rest length of backbone bond
    pub k_spring:      f32,  // spring constant
    pub r_repulse:     f32,  // soft-core repulsion radius
    pub k_repulse:     f32,  // repulsion strength
    pub brownian:      f32,  // noise stddev per sqrt(dt)
    pub max_velocity:  f32,
    pub damping:       f32,  // velocity damping each step (0..1, 1=no damping)
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            world_size:   400.0,
            r_spring:     12.0,
            k_spring:     0.30,
            r_repulse:    12.0,
            k_repulse:    0.25,
            brownian:     0.50,
            max_velocity: 2.0,
            damping:      0.95,
        }
    }
}

/// Apply forces to all alive atoms, then integrate position.
/// Operates on the full atom slice (caller is responsible for filtering dead).
pub fn step<R: Rng>(
    atoms:   &mut [Atom],
    spatial: &SpatialHash,
    cfg:     &PhysicsConfig,
    dt:      f32,
    rng:     &mut R,
) {
    let n = atoms.len();
    let mut force = vec![[0.0f32, 0.0f32]; n];

    // ── 1. Brownian noise ─────────────────────────────────────────────────
    // Wiener increment: stddev = brownian * sqrt(dt)
    let noise_scale = cfg.brownian * dt.sqrt();
    for i in 0..n {
        if !atoms[i].alive {
            continue;
        }
        let nx: f32 = rng.gen_range(-1.0..=1.0);
        let ny: f32 = rng.gen_range(-1.0..=1.0);
        force[i][0] += nx * noise_scale;
        force[i][1] += ny * noise_scale;
    }

    // ── 2. Spring forces on backbone bonds ────────────────────────────────
    // Walk each bond once: only when (a.id < a.backbone_next) to avoid double-count.
    for i in 0..n {
        if !atoms[i].alive {
            continue;
        }
        if let Some(bid) = atoms[i].backbone_next {
            let j = bid.as_index();
            if j >= n || !atoms[j].alive {
                continue;
            }
            let d = delta_toroidal(atoms[i].pos, atoms[j].pos, cfg.world_size);
            let dist = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-4);
            let stretch = dist - cfg.r_spring;
            let f = cfg.k_spring * stretch / dist;
            let fx = f * d[0];
            let fy = f * d[1];
            force[i][0] += fx;
            force[i][1] += fy;
            force[j][0] -= fx;
            force[j][1] -= fy;
        }
    }

    // ── 3. Spring force on transient template bonds (lighter than backbone) ─
    // Holds template-paired atoms close enough for R_link to find them.
    for i in 0..n {
        if !atoms[i].alive {
            continue;
        }
        if let Some(bid) = atoms[i].template_bond {
            // Avoid double counting: only fire when i < bid.as_index()
            if (bid.as_index()) <= i {
                continue;
            }
            let j = bid.as_index();
            if j >= n || !atoms[j].alive {
                continue;
            }
            let d = delta_toroidal(atoms[i].pos, atoms[j].pos, cfg.world_size);
            let dist = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-4);
            // Template bonds want a shorter rest distance (atoms paired side-by-side)
            let rest = cfg.r_spring * 0.6;
            let stretch = dist - rest;
            let f = cfg.k_spring * 0.5 * stretch / dist;
            force[i][0] += f * d[0];
            force[i][1] += f * d[1];
            force[j][0] -= f * d[0];
            force[j][1] -= f * d[1];
        }
    }

    // ── 4. Soft repulsion via spatial hash ────────────────────────────────
    let r2 = cfg.r_repulse * cfg.r_repulse;
    for i in 0..n {
        if !atoms[i].alive {
            continue;
        }
        let pos_i = atoms[i].pos;
        let neighbors = spatial.nearby(pos_i);
        for nid in neighbors {
            let j = nid.as_index();
            if j == i || j >= n || !atoms[j].alive {
                continue;
            }
            let dsq = dist_sq_toroidal(pos_i, atoms[j].pos, cfg.world_size);
            if dsq <= 0.0 || dsq >= r2 {
                continue;
            }
            let dist = dsq.sqrt();
            let overlap = cfg.r_repulse - dist;
            let f = cfg.k_repulse * overlap / dist;
            let d = delta_toroidal(pos_i, atoms[j].pos, cfg.world_size);
            // Force on i pushes away from j (i.e., -d direction)
            force[i][0] -= f * d[0];
            force[i][1] -= f * d[1];
        }
    }

    // ── 5. Integrate ───────────────────────────────────────────────────────
    let max_v = cfg.max_velocity;
    for i in 0..n {
        if !atoms[i].alive {
            continue;
        }
        let a = &mut atoms[i];
        a.vel[0] = (a.vel[0] + force[i][0]) * cfg.damping;
        a.vel[1] = (a.vel[1] + force[i][1]) * cfg.damping;
        // Clip velocity
        let v2 = a.vel[0] * a.vel[0] + a.vel[1] * a.vel[1];
        if v2 > max_v * max_v {
            let s = max_v / v2.sqrt();
            a.vel[0] *= s;
            a.vel[1] *= s;
        }
        a.pos[0] = (a.pos[0] + a.vel[0] * dt).rem_euclid(cfg.world_size);
        a.pos[1] = (a.pos[1] + a.vel[1] * dt).rem_euclid(cfg.world_size);
    }
}
