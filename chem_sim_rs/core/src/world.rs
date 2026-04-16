//! World container and main simulation loop.

use crate::atom::{Atom, AtomId, BaseType};
use crate::physics::{self, PhysicsConfig};
use crate::rules::{self, ReactionConfig, ReactionStats};
use crate::spatial::SpatialHash;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

#[derive(Debug, Clone)]
pub struct WorldConfig {
    pub world_size:    f32,
    pub seed:          u64,
    pub physics:       PhysicsConfig,
    pub reaction:      ReactionConfig,
    /// Cell size for spatial hash. Should be ≥ max(reaction_radius, r_repulse).
    pub cell_size:     f32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        let physics  = PhysicsConfig::default();
        let reaction = ReactionConfig::default();
        let cell = physics.r_repulse.max(reaction.reaction_radius);
        Self {
            world_size: physics.world_size,
            seed:       42,
            physics,
            reaction,
            cell_size: cell,
        }
    }
}

#[derive(Debug)]
pub struct World {
    pub config:    WorldConfig,
    pub atoms:     Vec<Atom>,
    pub time_step: u64,
    pub stats:     ReactionStats,
    rng:           Pcg64Mcg,
    spatial:       SpatialHash,
}

impl World {
    pub fn new(config: WorldConfig) -> Self {
        let world_size = config.world_size;
        let cell_size  = config.cell_size;
        Self {
            config,
            atoms:     Vec::new(),
            time_step: 0,
            stats:     ReactionStats::default(),
            rng:       Pcg64Mcg::seed_from_u64(0),
            spatial:   SpatialHash::new(world_size, cell_size),
        }
    }

    pub fn seeded(config: WorldConfig) -> Self {
        let mut w = Self::new(config.clone());
        w.rng = Pcg64Mcg::seed_from_u64(config.seed);
        w
    }

    /// Add an atom with no bonds. Its role (Free, InChain, etc.) is derived
    /// from its bond configuration, which is set by subsequent operations.
    pub fn add_atom(&mut self, ctype: BaseType, pos: [f32; 2]) -> AtomId {
        let id = AtomId(self.atoms.len() as u32);
        self.atoms.push(Atom::new(id, ctype, pos));
        id
    }

    /// Inject `n_per_type` free monomers of each base, randomly placed.
    pub fn inject_free_monomers(&mut self, n_per_type: usize) {
        let ws = self.config.world_size;
        for ct in 0..4u8 {
            for _ in 0..n_per_type {
                let x: f32 = self.rng.gen_range(0.0..ws);
                let y: f32 = self.rng.gen_range(0.0..ws);
                self.add_atom(BaseType::from_u8(ct), [x, y]);
            }
        }
    }

    /// Insert a chain of given sequence at center `(cx, cy)`, atoms separated
    /// by `spacing` on the x-axis. Backbone bonds are created between adjacent
    /// atoms, automatically making each atom's role "in chain".
    pub fn inject_chain(&mut self, sequence: &[BaseType], cx: f32, cy: f32, spacing: f32) {
        let n = sequence.len();
        let start_x = cx - (n as f32 - 1.0) * spacing * 0.5;
        let mut prev: Option<AtomId> = None;
        for (i, &ct) in sequence.iter().enumerate() {
            let jitter: f32 = self.rng.gen_range(-0.5..0.5);
            let pos = [start_x + i as f32 * spacing, cy + jitter];
            let id = self.add_atom(ct, pos);
            if let Some(p) = prev {
                rules::bond_backbone_next(&mut self.atoms, p, id);
            }
            prev = Some(id);
        }
    }

    /// Single simulation step:
    ///   1. Rebuild spatial hash
    ///   2. Reaction phase (random order, first-match per atom)
    ///   3. Physics phase (forces + integration)
    pub fn step(&mut self, dt: f32) {
        self.spatial.clear();
        for a in self.atoms.iter() {
            if a.alive {
                self.spatial.insert(a.id, a.pos);
            }
        }

        let n = self.atoms.len();
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.rng.gen_range(0..=i);
            order.swap(i, j);
        }

        let mut reacted = vec![false; n];
        for &i in order.iter() {
            if !self.atoms[i].alive || reacted[i] {
                continue;
            }
            let aid = self.atoms[i].id;
            let pos = self.atoms[i].pos;
            let mut neighbors = self.spatial.nearby(pos);
            neighbors.retain(|&nid| !reacted[nid.as_index()]);
            let fired = rules::try_react(
                &mut self.atoms,
                aid,
                &neighbors,
                self.config.world_size,
                &self.config.reaction,
                &mut self.stats,
                &mut self.rng,
                dt,
            );
            if fired {
                reacted[i] = true;
                if let Some(t) = self.atoms[i].template_bond {
                    let ti = t.as_index();
                    if ti < n {
                        reacted[ti] = true;
                    }
                }
            }
        }

        physics::step(
            &mut self.atoms,
            &self.spatial,
            &self.config.physics,
            dt,
            &mut self.rng,
        );

        self.time_step += 1;
    }
}
