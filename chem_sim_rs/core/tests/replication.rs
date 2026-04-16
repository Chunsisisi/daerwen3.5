//! Validation test: starting with a single AUGC chain template + free monomers,
//! at least one complete daughter chain must form within N steps.

use chem_sim_core::atom::BaseType;
use chem_sim_core::snapshot::{active_replication_count, build_snapshot};
use chem_sim_core::world::{World, WorldConfig};

// Note: After the state-removal refactor, atoms have no `state` field.
// Roles are derived from bond configuration. Tests no longer need to
// pass AtomState to add_atom or inject_chain.

#[test]
fn replication_emerges_from_template_plus_monomers() {
    let mut cfg = WorldConfig::default();
    cfg.world_size = 120.0;
    cfg.physics.world_size = 120.0;
    cfg.cell_size = cfg.physics.r_repulse.max(cfg.reaction.reaction_radius);
    cfg.seed = 42;
    cfg.reaction.complementary = false; // faithful replication for this test

    let mut world = World::seeded(cfg);

    // Seed: a single AUGC template at the center
    let template = vec![BaseType::A, BaseType::U, BaseType::G, BaseType::C];
    world.inject_chain(&template, 60.0, 60.0, 12.0);

    // Add 200 free monomers per type (800 total) for raw material
    world.inject_free_monomers(200);

    // Run
    let n_steps = 5000usize;
    let dt = 1.0f32;
    let mut max_active_rep = 0usize;
    for s in 0..n_steps {
        world.step(dt);
        if s % 500 == 0 {
            let cur_active = active_replication_count(&world.atoms);
            if cur_active > max_active_rep {
                max_active_rep = cur_active;
            }
            let snap = build_snapshot(&world.atoms, &world.stats, world.time_step);
            eprintln!(
                "step {:5}: alive={} chains={} pair={} link={} release={} rep={} active_rep={}",
                world.time_step,
                snap.n_atoms,
                snap.n_chains,
                snap.stats.pair_events,
                snap.stats.link_events,
                snap.stats.release_events,
                snap.stats.replication_events,
                cur_active,
            );
        }
    }

    let final_snap = build_snapshot(&world.atoms, &world.stats, world.time_step);
    eprintln!("FINAL: {:#?}", final_snap);

    assert!(
        world.stats.pair_events > 0,
        "no pair events fired at all — R_pair is broken",
    );
    assert!(
        world.stats.link_events > 0,
        "no link events fired — R_link can't find adjacent paired atoms",
    );
    assert!(
        world.stats.release_events > 0,
        "no release events fired — R_release condition never satisfied",
    );
    assert!(
        world.stats.replication_events > 0,
        "no full daughter chain ever completed — replication does not occur",
    );
}
