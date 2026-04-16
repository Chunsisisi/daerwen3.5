//! Parallel stress test: run N independent simulations across all logical cores.
//! Designed to saturate ~80-100% total CPU usage and demonstrate aggregate throughput.

use chem_sim_core::atom::BaseType;
use chem_sim_core::snapshot::build_snapshot;
use chem_sim_core::world::{World, WorldConfig};
use rayon::prelude::*;
use std::time::Instant;

#[test]
fn stress_parallel_full_cpu() {
    let n_workers = num_cpus_logical();
    let n_steps_per_worker = 30_000usize;
    let n_atoms_per_world  = 4000usize;

    eprintln!(
        "PARALLEL STRESS: {} workers × {} steps × ~{} atoms",
        n_workers, n_steps_per_worker, n_atoms_per_world
    );
    eprintln!("Watch Task Manager: total CPU should jump to ~80-100%.");

    let t0 = Instant::now();

    // Run n_workers independent simulations in parallel; each gets its own seed.
    let results: Vec<_> = (0..n_workers)
        .into_par_iter()
        .map(|wid| {
            let mut cfg = WorldConfig::default();
            cfg.world_size = 400.0;
            cfg.physics.world_size = 400.0;
            cfg.cell_size = cfg.physics.r_repulse.max(cfg.reaction.reaction_radius);
            cfg.seed = 42 + wid as u64 * 7919;

            let mut world = World::seeded(cfg);
            world.inject_free_monomers(n_atoms_per_world / 4);
            for i in 0..5 {
                let cx = 80.0 + i as f32 * 60.0;
                world.inject_chain(
                    &[
                        BaseType::A, BaseType::U, BaseType::G, BaseType::C,
                        BaseType::A, BaseType::U,
                    ],
                    cx,
                    200.0,
                    12.0,
                );
            }

            let dt = 1.0f32;
            for _ in 0..n_steps_per_worker {
                world.step(dt);
            }

            let snap = build_snapshot(&world.atoms, &world.stats, world.time_step);
            (wid, snap.stats.replication_events, snap.n_chains, snap.diversity)
        })
        .collect();

    let elapsed = t0.elapsed().as_secs_f32();
    let total_steps = n_workers * n_steps_per_worker;
    let total_atom_steps = total_steps * n_atoms_per_world;

    eprintln!("\n=== PARALLEL STRESS RESULT ===");
    eprintln!("  Workers:                {}", n_workers);
    eprintln!("  Total simulated steps:  {}", total_steps);
    eprintln!("  Total atom-steps:       {:.2e}", total_atom_steps as f32);
    eprintln!("  Wall time:              {:.2}s", elapsed);
    eprintln!("  Aggregate steps/sec:    {:.0}", total_steps as f32 / elapsed);
    eprintln!("  Aggregate atom-steps/s: {:.2e}", total_atom_steps as f32 / elapsed);
    eprintln!();
    eprintln!("  Per-worker results:");
    for (wid, reps, chains, div) in &results {
        eprintln!(
            "    worker {:2}: replications={:5}  final_chains={:3}  diversity={:.3}",
            wid, reps, chains, div
        );
    }
    eprintln!("==============================");
    let total_reps: u64 = results.iter().map(|r| r.1).sum();
    eprintln!("  Total replications across all workers: {}", total_reps);

    assert!(total_reps > 0);
}

fn num_cpus_logical() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}
