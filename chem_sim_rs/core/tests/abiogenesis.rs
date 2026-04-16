//! Abiogenesis test: start with PURE free monomers, no seeded templates.
//! If the chemistry is complete, the system should bootstrap chain formation
//! from R_polymerize alone, then accelerate via R_pair (template catalysis).
//!
//! Pass criteria (long simulation):
//!   1. R_polymerize fires (free monomers actually bond)
//!   2. R_pair fires (initial dimers serve as templates)
//!   3. R_release fires (replication completes)
//!   4. Replication events > 0
//!   5. Chains exist at the end (system is sustained, not just bursting and dying)

use chem_sim_core::atom::BaseType;
use chem_sim_core::snapshot::build_snapshot;
use chem_sim_core::world::{World, WorldConfig};
use std::time::Instant;

#[test]
fn abiogenesis_from_pure_monomers() {
    let mut cfg = WorldConfig::default();
    cfg.world_size = 200.0;
    cfg.physics.world_size = 200.0;
    cfg.cell_size = cfg.physics.r_repulse.max(cfg.reaction.reaction_radius);
    cfg.seed = 42;

    let mut world = World::seeded(cfg);
    // Just free monomers, NO seeded template chain.
    world.inject_free_monomers(500); // 500 per type × 4 = 2000 atoms

    let n_steps = 30_000usize;
    let dt = 1.0f32;

    eprintln!("ABIOGENESIS: {} atoms, no seeded chain. Watching for self-organization...", world.atoms.len());

    let t0 = Instant::now();
    for s in 0..n_steps {
        world.step(dt);
        if s % 2000 == 0 || s == n_steps - 1 {
            let snap = build_snapshot(&world.atoms, &world.stats, world.time_step);
            eprintln!(
                "  step {:5}: chains={:4} poly={:5} pair={:5} link={:4} rep={:4} elapsed={:.1}s",
                s,
                snap.n_chains,
                snap.stats.polymerize_events,
                snap.stats.pair_events,
                snap.stats.link_events,
                snap.stats.replication_events,
                t0.elapsed().as_secs_f32()
            );
        }
    }

    let final_snap = build_snapshot(&world.atoms, &world.stats, world.time_step);
    eprintln!("\nFINAL SNAPSHOT:");
    eprintln!("  Total atoms:    {}", final_snap.n_atoms);
    eprintln!("  Free monomers:  {}", final_snap.n_free);
    eprintln!("  Chains:         {}", final_snap.n_chains);
    eprintln!("  Chain lengths:  {:?}", final_snap.chain_lengths);
    eprintln!("  Replications:   {}", final_snap.stats.replication_events);
    eprintln!("  Polymerizations:{}", final_snap.stats.polymerize_events);
    eprintln!("  Pair events:    {}", final_snap.stats.pair_events);
    eprintln!("  Link events:    {}", final_snap.stats.link_events);
    eprintln!("  Diversity:      {:.3}", final_snap.diversity);
    if let Some(s) = &final_snap.dominant_sequence {
        eprintln!("  Dominant seq:   {}", s);
    }

    // Pass criteria
    assert!(
        final_snap.stats.polymerize_events > 0,
        "R_polymerize never fired — free monomers can't bond spontaneously"
    );
    assert!(
        final_snap.stats.pair_events > 0,
        "R_pair never fired — bootstrapped dimers didn't serve as templates"
    );
    assert!(
        final_snap.stats.replication_events > 0,
        "No complete replication — abiogenesis pathway broken"
    );
    assert!(
        final_snap.n_chains > 0,
        "All chains died at end — system not sustainable even with R_polymerize"
    );
}

#[test]
fn _bonus_demonstrate_silly_outcomes() {
    // This is a quick demo, not an assertion — check that the system reaches a
    // non-trivial steady state at the end, not just bursting and dying.
    use chem_sim_core::atom::BaseType;
    let mut cfg = WorldConfig::default();
    cfg.world_size = 200.0;
    cfg.physics.world_size = 200.0;
    cfg.cell_size = cfg.physics.r_repulse.max(cfg.reaction.reaction_radius);
    cfg.seed = 99;

    let mut world = World::seeded(cfg);
    world.inject_free_monomers(500);
    for _ in 0..30000 {
        world.step(1.0);
    }
    let snap = build_snapshot(&world.atoms, &world.stats, world.time_step);
    eprintln!("\n[demo seed=99] chains={} rep={} diversity={:.3}",
              snap.n_chains, snap.stats.replication_events, snap.diversity);
}
