//! Classical molecular evolution experiments — E1 through E6.
//!
//! Each experiment is independently testable via:
//!     cargo test --release e1_lethal_mutagenesis -- --nocapture
//!     cargo test --release e2_quasispecies -- --nocapture
//!     ... etc.
//!
//! All experiments use parallel seeds via rayon to keep wall time low.
//! Theoretical predictions are noted in each experiment's comments so the
//! reader can judge whether the system reproduces the classical result.

use chem_sim_core::atom::BaseType;
use chem_sim_core::snapshot::build_snapshot;
use chem_sim_core::world::{World, WorldConfig};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

// ── Helpers ──────────────────────────────────────────────────────────

fn world_with(seed: u64, mu: f32, world_size: f32) -> World {
    let mut cfg = WorldConfig::default();
    cfg.world_size = world_size;
    cfg.physics.world_size = world_size;
    cfg.cell_size = cfg.physics.r_repulse.max(cfg.reaction.reaction_radius);
    cfg.seed = seed;
    cfg.reaction.mutation_rate = mu;
    World::seeded(cfg)
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let m = values.iter().sum::<f32>() / values.len() as f32;
    let v = values.iter().map(|x| (x - m).powi(2)).sum::<f32>() / values.len() as f32;
    (m, v.sqrt())
}

fn hamming(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).filter(|(x, y)| x != y).count() + (a.len().abs_diff(b.len()))
}

// ════════════════════════════════════════════════════════════════════════
// E1: Lethal Mutagenesis
// ─ Theory: at sufficiently high mutation rate, every replication introduces
//   errors; the population loses sequence information and collapses.
// ─ Measure: chain count vs μ
// ─ Expected: monotonic decline in chain count as μ rises past ~0.1-0.3
// ════════════════════════════════════════════════════════════════════════

#[test]
fn e1_lethal_mutagenesis() {
    let mus = [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00];
    let n_seeds = 5;
    let n_steps = 3000usize;

    let tasks: Vec<(f32, u64)> = mus
        .iter()
        .flat_map(|&mu| (0..n_seeds).map(move |s| (mu, s as u64 * 7 + 42)))
        .collect();

    eprintln!("\n=== E1: Lethal Mutagenesis ===");
    eprintln!("  Tasks: {} ({} mu × {} seeds), {} steps each", tasks.len(), mus.len(), n_seeds, n_steps);
    let t0 = Instant::now();

    let results: Vec<(f32, u32, u64, u64)> = tasks
        .into_par_iter()
        .map(|(mu, seed)| {
            let mut w = world_with(seed, mu, 120.0);
            w.inject_chain(
                &[BaseType::A, BaseType::U, BaseType::G, BaseType::C],
                60.0,
                60.0,
                12.0,
            );
            w.inject_free_monomers(150);
            for _ in 0..n_steps {
                w.step(1.0);
            }
            let snap = build_snapshot(&w.atoms, &w.stats, w.time_step);
            (
                mu,
                snap.n_chains as u32,
                snap.stats.replication_events,
                snap.stats.mutation_events,
            )
        })
        .collect();

    eprintln!("  elapsed: {:.1}s", t0.elapsed().as_secs_f32());

    // Aggregate by μ
    let mut groups: BTreeMap<u32, Vec<(u32, u64, u64)>> = BTreeMap::new();
    for &(mu, n_chains, reps, muts) in &results {
        groups.entry((mu * 1000.0) as u32).or_default().push((n_chains, reps, muts));
    }

    eprintln!();
    eprintln!("  {:>8}  {:>14}  {:>10}  {:>10}  {:>8}", "μ", "chains(±σ)", "rep_mean", "mut_mean", "verdict");
    eprintln!("  {}", "-".repeat(60));
    for (mu_key, data) in &groups {
        let mu = *mu_key as f32 / 1000.0;
        let chains: Vec<f32> = data.iter().map(|x| x.0 as f32).collect();
        let (m, s) = mean_std(&chains);
        let mean_r = data.iter().map(|x| x.1 as f64).sum::<f64>() / data.len() as f64;
        let mean_m = data.iter().map(|x| x.2 as f64).sum::<f64>() / data.len() as f64;
        let verdict = if m > 100.0 { "stable" } else if m < 20.0 { "COLLAPSED" } else { "weak" };
        eprintln!("  {:>8.2}  {:>7.1} ± {:>4.1}  {:>10.0}  {:>10.0}  {:>8}", mu, m, s, mean_r, mean_m, verdict);
    }
    eprintln!();
    eprintln!("  Theory: chain count should decline monotonically as μ → 1.0");
}

// ════════════════════════════════════════════════════════════════════════
// E2: Quasispecies Distribution
// ─ Theory (Eigen 1971): selection acts not on a single sequence but on a
//   "cloud" around a master sequence. Cloud width grows with μ; past the
//   error threshold, the cloud collapses (no master).
// ─ Measure: Hamming distance distribution from dominant sequence
// ════════════════════════════════════════════════════════════════════════

#[test]
fn e2_quasispecies() {
    let mus = [0.0, 0.01, 0.05, 0.10, 0.20, 0.35];
    let n_seeds = 5;
    let n_steps = 3000usize;

    let tasks: Vec<(f32, u64)> = mus
        .iter()
        .flat_map(|&mu| (0..n_seeds).map(move |s| (mu, s as u64 * 7 + 42)))
        .collect();

    eprintln!("\n=== E2: Quasispecies Distribution ===");
    eprintln!("  Tasks: {} ({} mu × {} seeds), {} steps each", tasks.len(), mus.len(), n_seeds, n_steps);
    let t0 = Instant::now();

    // Each result: (μ, mean_hamming, sample_count)
    let results: Vec<(f32, f32, usize)> = tasks
        .into_par_iter()
        .map(|(mu, seed)| {
            let mut w = world_with(seed, mu, 120.0);
            // 5-base alternating template gives Hamming resolution
            w.inject_chain(
                &[BaseType::A, BaseType::U, BaseType::A, BaseType::U, BaseType::A],
                60.0,
                60.0,
                12.0,
            );
            w.inject_free_monomers(150);
            for _ in 0..n_steps {
                w.step(1.0);
            }
            let snap = build_snapshot(&w.atoms, &w.stats, w.time_step);
            let total: usize = snap.sequence_counts.values().sum();
            if total == 0 {
                return (mu, 0.0, 0);
            }
            let dominant = match &snap.dominant_sequence {
                Some(s) => s.clone(),
                None => return (mu, 0.0, 0),
            };
            let mut sum_h: f32 = 0.0;
            let mut count = 0usize;
            for (seq, &cnt) in &snap.sequence_counts {
                let h = hamming(seq, &dominant) as f32;
                sum_h += h * cnt as f32;
                count += cnt;
            }
            let mean_h = if count > 0 { sum_h / count as f32 } else { 0.0 };
            (mu, mean_h, count)
        })
        .collect();

    eprintln!("  elapsed: {:.1}s", t0.elapsed().as_secs_f32());

    let mut groups: BTreeMap<u32, Vec<(f32, usize)>> = BTreeMap::new();
    for &(mu, h, c) in &results {
        groups.entry((mu * 1000.0) as u32).or_default().push((h, c));
    }

    eprintln!();
    eprintln!("  {:>8}  {:>16}  {:>12}", "μ", "mean_hamming", "n_chains_avg");
    eprintln!("  {}", "-".repeat(46));
    for (mu_key, data) in &groups {
        let mu = *mu_key as f32 / 1000.0;
        let h_vals: Vec<f32> = data.iter().map(|x| x.0).collect();
        let (mh, sh) = mean_std(&h_vals);
        let avg_n: f32 = data.iter().map(|x| x.1 as f32).sum::<f32>() / data.len() as f32;
        eprintln!("  {:>8.2}  {:>7.3} ± {:>5.3}  {:>12.1}", mu, mh, sh, avg_n);
    }
    eprintln!();
    eprintln!("  Theory: mean Hamming distance should grow with μ until error threshold,");
    eprintln!("          then drop sharply as the cloud collapses.");
}

// ════════════════════════════════════════════════════════════════════════
// E3: Gause's Competitive Exclusion
// ─ Theory: two species sharing the same niche cannot stably coexist;
//   different niches → coexistence.
// ─ Setup: two competing templates with overlapping or distinct base usage
// ─ Measure: final population fractions + exclusion strength
// ════════════════════════════════════════════════════════════════════════

#[test]
fn e3_gause_exclusion() {
    // 4 pairs: BC = "AU", CB = "UA" (same atoms, different order);
    //          BB = "AA", CC = "UU" (use different atom types entirely).
    //          BC vs BB: partial overlap.
    let pairs: [(&[BaseType], &[BaseType], &str); 4] = [
        (&[BaseType::A, BaseType::U], &[BaseType::U, BaseType::A], "AU vs UA  (same niche)"),
        (&[BaseType::A, BaseType::A], &[BaseType::U, BaseType::U], "AA vs UU  (different)"),
        (&[BaseType::A, BaseType::U], &[BaseType::A, BaseType::A], "AU vs AA  (partial)"),
        (&[BaseType::A, BaseType::U], &[BaseType::U, BaseType::U], "AU vs UU  (partial)"),
    ];

    let n_seeds = 8;
    let n_steps = 3000usize;

    eprintln!("\n=== E3: Gause's Competitive Exclusion ===");
    eprintln!("  {} pairs × {} seeds × {} steps", pairs.len(), n_seeds, n_steps);
    let t0 = Instant::now();

    for (i, (seq_a, seq_b, label)) in pairs.iter().enumerate() {
        let tasks: Vec<u64> = (0..n_seeds).map(|s| s as u64 * 7 + 42 + i as u64 * 100).collect();
        let results: Vec<(f32, f32)> = tasks
            .into_par_iter()
            .map(|seed| {
                let mut w = world_with(seed, 0.01, 120.0);
                w.inject_chain(seq_a, 40.0, 60.0, 12.0);
                w.inject_chain(seq_b, 80.0, 60.0, 12.0);
                w.inject_free_monomers(100);
                for _ in 0..n_steps {
                    w.step(1.0);
                }
                let snap = build_snapshot(&w.atoms, &w.stats, w.time_step);
                let total: f32 = snap.sequence_counts.values().sum::<usize>() as f32;
                if total == 0.0 {
                    return (0.0, 0.0);
                }
                let key_a: String = seq_a.iter().map(|c| c.label()).collect();
                let key_b: String = seq_b.iter().map(|c| c.label()).collect();
                let fa = snap.sequence_counts.get(&key_a).copied().unwrap_or(0) as f32 / total;
                let fb = snap.sequence_counts.get(&key_b).copied().unwrap_or(0) as f32 / total;
                (fa, fb)
            })
            .collect();

        let fas: Vec<f32> = results.iter().map(|x| x.0).collect();
        let fbs: Vec<f32> = results.iter().map(|x| x.1).collect();
        let (mfa, sfa) = mean_std(&fas);
        let (mfb, sfb) = mean_std(&fbs);
        let exclusion: f32 = results.iter().map(|x| x.0.max(x.1)).sum::<f32>() / results.len() as f32;
        let coexist = results.iter().filter(|x| {
            let a = x.0;
            let b = x.1;
            let r = a / (a + b + 1e-9);
            r > 0.3 && r < 0.7
        }).count();

        eprintln!();
        eprintln!("  {}", label);
        eprintln!("    A frac: {:.3} ± {:.3}   B frac: {:.3} ± {:.3}", mfa, sfa, mfb, sfb);
        eprintln!("    exclusion (max f): {:.3}   coexist trials: {}/{}",
            exclusion, coexist, n_seeds);
        let verdict = if exclusion > 0.7 { "EXCLUSION" } else { "COEXISTENCE" };
        eprintln!("    → {}", verdict);
    }
    eprintln!();
    eprintln!("  elapsed: {:.1}s", t0.elapsed().as_secs_f32());
    eprintln!("  Theory: same-niche pairs should show exclusion (one wins);");
    eprintln!("          different-niche pairs should coexist near 50/50.");
}

// ════════════════════════════════════════════════════════════════════════
// E4: Logistic Growth
// ─ Theory: N(t) = K / (1 + (K/N0 - 1) e^(-r t))  →  S-curve
// ─ Measure: chain count at multiple checkpoints, estimate r and K
// ════════════════════════════════════════════════════════════════════════

#[test]
fn e4_logistic_growth() {
    let checkpoints = [50, 100, 200, 300, 500, 800, 1200, 1800, 2500, 3000usize];
    let n_seeds = 8;

    eprintln!("\n=== E4: Logistic Growth ===");
    let t0 = Instant::now();

    let series: Vec<Vec<f32>> = (0..n_seeds)
        .into_par_iter()
        .map(|s| {
            let seed = s as u64 * 7 + 42;
            let mut w = world_with(seed, 0.01, 120.0);
            w.inject_chain(
                &[BaseType::A, BaseType::U, BaseType::A, BaseType::U],
                60.0,
                60.0,
                12.0,
            );
            w.inject_free_monomers(150);
            let mut measurements = Vec::with_capacity(checkpoints.len());
            let mut cp_idx = 0;
            for step in 1..=*checkpoints.last().unwrap() {
                w.step(1.0);
                if cp_idx < checkpoints.len() && step == checkpoints[cp_idx] {
                    let snap = build_snapshot(&w.atoms, &w.stats, w.time_step);
                    measurements.push(snap.n_chains as f32);
                    cp_idx += 1;
                }
            }
            measurements
        })
        .collect();

    eprintln!("  elapsed: {:.1}s", t0.elapsed().as_secs_f32());

    eprintln!();
    eprintln!("  {:>8}  {:>14}  {:>10}", "step", "chains(±σ)", "growth/step");
    eprintln!("  {}", "-".repeat(40));
    let mut means = Vec::with_capacity(checkpoints.len());
    for i in 0..checkpoints.len() {
        let vals: Vec<f32> = series.iter().map(|s| s[i]).collect();
        let (m, s) = mean_std(&vals);
        means.push(m);
        let g = if i == 0 { 0.0 } else { (m - means[i - 1]) / means[i - 1].max(1.0) };
        eprintln!("  {:>8}  {:>7.1} ± {:>4.1}  {:>+10.3}", checkpoints[i], m, s, g);
    }
    let k_est: f32 = means[6..].iter().sum::<f32>() / (means.len() - 6) as f32;
    let r_est: f32 = if means[0] > 0.0 {
        (means[4] / means[0].max(1.0)).ln() / (checkpoints[4] as f32 - checkpoints[0] as f32)
    } else {
        0.0
    };
    eprintln!();
    eprintln!("  Estimated carrying capacity K ≈ {:.0}", k_est);
    eprintln!("  Estimated intrinsic growth rate r ≈ {:.4}/step", r_est);
    eprintln!("  Theory: should fit N(t) = K / (1 + (K/N0 - 1) e^(-rt))");
}

// ════════════════════════════════════════════════════════════════════════
// E5: Length-Rate Tradeoff (limited — current rate_break favors dimers)
// ─ Theory (Holland schema theorem flavor): shorter templates replicate
//   faster because (a) fewer atoms to recruit, (b) fewer bonds to maintain.
// ─ Measure: replications-per-template-per-step over short time window
// ════════════════════════════════════════════════════════════════════════

#[test]
fn e5_length_rate_tradeoff() {
    // Test lengths: 2, 3, 4, 5, 6
    let lengths = [2usize, 3, 4, 5, 6];
    let n_seeds = 5;
    let n_steps = 1500usize; // short, to measure initial replication rate

    eprintln!("\n=== E5: Length-Rate Tradeoff ===");
    let t0 = Instant::now();

    let tasks: Vec<(usize, u64)> = lengths
        .iter()
        .flat_map(|&len| (0..n_seeds).map(move |s| (len, s as u64 * 7 + 42)))
        .collect();

    let results: Vec<(usize, u64)> = tasks
        .into_par_iter()
        .map(|(len, seed)| {
            let mut w = world_with(seed, 0.01, 120.0);
            // Build alternating AU…AU template of given length
            let template: Vec<BaseType> = (0..len)
                .map(|i| if i % 2 == 0 { BaseType::A } else { BaseType::U })
                .collect();
            w.inject_chain(&template, 60.0, 60.0, 12.0);
            w.inject_free_monomers(150);
            for _ in 0..n_steps {
                w.step(1.0);
            }
            (len, w.stats.replication_events)
        })
        .collect();

    eprintln!("  elapsed: {:.1}s", t0.elapsed().as_secs_f32());

    let mut by_len: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    for &(len, reps) in &results {
        by_len.entry(len).or_default().push(reps);
    }

    eprintln!();
    eprintln!("  {:>8}  {:>20}  {:>14}", "length", "replications(±σ)", "rate/step");
    eprintln!("  {}", "-".repeat(48));
    for (len, vals) in &by_len {
        let v: Vec<f32> = vals.iter().map(|&x| x as f32).collect();
        let (m, s) = mean_std(&v);
        eprintln!("  {:>8}  {:>10.1} ± {:>5.1}  {:>14.4}", len, m, s, m / n_steps as f32);
    }
    eprintln!();
    eprintln!("  Theory: replication rate per step should DECREASE with length");
    eprintln!("          (more atoms to recruit per cycle).");
}

// ════════════════════════════════════════════════════════════════════════
// E6: Niche Partitioning (long-term coexistence)
// ─ Theory: stable coexistence requires distinct ecological niches.
// ─ Setup: same as E3 but longer time and time-fraction-of-coexistence metric.
// ════════════════════════════════════════════════════════════════════════

#[test]
fn e6_niche_partitioning() {
    let pairs: [(&[BaseType], &[BaseType], &str); 3] = [
        (&[BaseType::A, BaseType::U], &[BaseType::U, BaseType::A], "AU vs UA  (same niche)"),
        (&[BaseType::A, BaseType::A], &[BaseType::U, BaseType::U], "AA vs UU  (different)"),
        (&[BaseType::A, BaseType::A], &[BaseType::A, BaseType::U], "AA vs AU  (partial)"),
    ];

    let n_seeds = 8;
    let n_steps = 4000usize;
    let sample_interval = 200usize;

    eprintln!("\n=== E6: Niche Partitioning ===");
    let t0 = Instant::now();

    for (i, (seq_a, seq_b, label)) in pairs.iter().enumerate() {
        let tasks: Vec<u64> = (0..n_seeds).map(|s| s as u64 * 7 + 42 + i as u64 * 100).collect();
        let results: Vec<(f32, f32, f32)> = tasks
            .into_par_iter()
            .map(|seed| {
                let mut w = world_with(seed, 0.02, 120.0);
                w.inject_chain(seq_a, 40.0, 60.0, 12.0);
                w.inject_chain(seq_b, 80.0, 60.0, 12.0);
                w.inject_free_monomers(100);
                let key_a: String = seq_a.iter().map(|c| c.label()).collect();
                let key_b: String = seq_b.iter().map(|c| c.label()).collect();
                let mut coexist_steps = 0usize;
                let mut total_samples = 0usize;
                for step in 1..=n_steps {
                    w.step(1.0);
                    if step % sample_interval == 0 {
                        total_samples += 1;
                        let snap = build_snapshot(&w.atoms, &w.stats, w.time_step);
                        let na = *snap.sequence_counts.get(&key_a).unwrap_or(&0);
                        let nb = *snap.sequence_counts.get(&key_b).unwrap_or(&0);
                        if na >= 1 && nb >= 1 {
                            coexist_steps += 1;
                        }
                    }
                }
                let snap = build_snapshot(&w.atoms, &w.stats, w.time_step);
                let total: f32 = snap.sequence_counts.values().sum::<usize>() as f32;
                let fa = snap.sequence_counts.get(&key_a).copied().unwrap_or(0) as f32 / total.max(1.0);
                let fb = snap.sequence_counts.get(&key_b).copied().unwrap_or(0) as f32 / total.max(1.0);
                let coexist_frac = coexist_steps as f32 / total_samples.max(1) as f32;
                (coexist_frac, fa, fb)
            })
            .collect();

        let cf: f32 = results.iter().map(|x| x.0).sum::<f32>() / results.len() as f32;
        let fa: f32 = results.iter().map(|x| x.1).sum::<f32>() / results.len() as f32;
        let fb: f32 = results.iter().map(|x| x.2).sum::<f32>() / results.len() as f32;
        let verdict = if cf > 0.6 { "STABLE COEXISTENCE" } else if cf > 0.3 { "ALTERNATING" } else { "EXCLUSION" };
        eprintln!();
        eprintln!("  {}", label);
        eprintln!("    coexist time fraction: {:.3}   final A: {:.3}   B: {:.3}", cf, fa, fb);
        eprintln!("    → {}", verdict);
    }
    eprintln!();
    eprintln!("  elapsed: {:.1}s", t0.elapsed().as_secs_f32());
    eprintln!("  Theory: 'different' should sustain coexistence longer than 'same'.");
}
