//! Stress test: large world, many atoms, long simulation.
//! Designed to make CPU usage visible in Task Manager.

use chem_sim_core::atom::BaseType;
use chem_sim_core::snapshot::build_snapshot;
use chem_sim_core::world::{World, WorldConfig};
use std::time::Instant;

#[test]
fn stress_visible_cpu() {
    let mut cfg = WorldConfig::default();
    cfg.world_size = 400.0;
    cfg.physics.world_size = 400.0;
    cfg.cell_size = cfg.physics.r_repulse.max(cfg.reaction.reaction_radius);
    cfg.seed = 42;

    let mut world = World::seeded(cfg);

    // 4000 free monomers (1000 per type) + 5 templates
    world.inject_free_monomers(1000);
    for i in 0..5 {
        let cx = 80.0 + i as f32 * 60.0;
        world.inject_chain(
            &[BaseType::A, BaseType::U, BaseType::G, BaseType::C, BaseType::A, BaseType::U],
            cx,
            200.0,
            12.0,
        );
    }

    let n_steps = 50_000usize;
    let dt = 1.0f32;

    eprintln!(
        "STRESS: {} atoms, {} steps. Watch Task Manager for ~30 seconds of single-core 100%.",
        world.atoms.len(),
        n_steps
    );

    let t0 = Instant::now();
    for s in 0..n_steps {
        world.step(dt);
        if s % 5000 == 0 {
            let snap = build_snapshot(&world.atoms, &world.stats, world.time_step);
            eprintln!(
                "  step {:6}: chains={:4} rep={:5} elapsed={:.1}s",
                s,
                snap.n_chains,
                snap.stats.replication_events,
                t0.elapsed().as_secs_f32()
            );
        }
    }
    let elapsed = t0.elapsed().as_secs_f32();
    let snap = build_snapshot(&world.atoms, &world.stats, world.time_step);

    eprintln!("\n=== STRESS RESULT ===");
    eprintln!("  Total steps:        {}", n_steps);
    eprintln!("  Total atoms:        {}", world.atoms.len());
    eprintln!("  Wall time:          {:.2}s", elapsed);
    eprintln!("  Steps/sec:          {:.0}", n_steps as f32 / elapsed);
    eprintln!("  Atoms × steps / sec:{:.2e}", world.atoms.len() as f32 * n_steps as f32 / elapsed);
    eprintln!("  Replications:       {}", snap.stats.replication_events);
    eprintln!("  Pair events:        {}", snap.stats.pair_events);
    eprintln!("  Link events:        {}", snap.stats.link_events);
    eprintln!("  Final chains:       {}", snap.n_chains);
    eprintln!("  Sequence diversity: {:.3}", snap.diversity);
    if let Some(s) = &snap.dominant_sequence {
        eprintln!("  Dominant sequence:  {}", s);
    }
    eprintln!("=====================");

    assert!(snap.stats.replication_events > 0, "stress test should produce replications");
}
