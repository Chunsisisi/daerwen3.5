//! Python bindings for chem_sim_rs.
//!
//! Minimum viable API for DAERWEN integration (Level 1):
//!   - replicate_chain: template copy + mutation
//!   - phenotypes_from_chain: uniform composition-based gene expression
//!   - ChainWorld: full chemistry simulation (for Level 2+)
//!
//! All chain sequences are represented as lists of u8 (0=A, 1=U, 2=G, 3=C).

use chem_sim_core::atom::{BaseType, COMPLEMENT};
use chem_sim_core::world::{World, WorldConfig};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

// ── Helpers ──────────────────────────────────────────────────────────

fn bases_from_u8_list(values: &[u8]) -> Vec<BaseType> {
    values.iter().map(|&v| BaseType::from_u8(v)).collect()
}

fn u8_list_from_bases(bases: &[BaseType]) -> Vec<u8> {
    bases.iter().map(|b| b.as_u8()).collect()
}

// ── Replication (Level 1 core) ───────────────────────────────────────

/// Template-copy a parent chain to produce an offspring, with mutation.
///
/// Args:
///     parent: list of u8 (0=A, 1=U, 2=G, 3=C)
///     mutation_rate: per-base probability of random substitution
///     complementary: if True, use Watson-Crick pairs (A↔U, G↔C)
///     seed: optional PRNG seed
///
/// Returns:
///     list of u8 representing the daughter chain
#[pyfunction]
#[pyo3(signature = (parent, mutation_rate, complementary=false, seed=None))]
fn replicate_chain(
    parent: Vec<u8>,
    mutation_rate: f32,
    complementary: bool,
    seed: Option<u64>,
) -> Vec<u8> {
    let mut rng = match seed {
        Some(s) => Pcg64Mcg::seed_from_u64(s),
        None => Pcg64Mcg::from_entropy(),
    };
    let mut offspring: Vec<u8> = Vec::with_capacity(parent.len());
    for &base in parent.iter() {
        let base = base % 4;
        let template = if complementary {
            COMPLEMENT[base as usize].as_u8()
        } else {
            base
        };
        let out = if rng.gen::<f32>() < mutation_rate {
            rng.gen_range(0..4u8)
        } else {
            template
        };
        offspring.push(out);
    }
    offspring
}

// ── Phenotype expression (Level 1 core) ───────────────────────────────

/// Compute 12 DAERWEN phenotypes from chain composition using a single
/// uniform mapping principle: phenotype = lerp(min, max, normalized_freq).
///
/// For random genomes, every base frequency ≈ 0.25, so every phenotype
/// lands at the midpoint of its range, providing a neutral starting point
/// for selection to push values in either direction.
#[pyfunction]
fn phenotypes_from_chain(py: Python<'_>, chain: Vec<u8>) -> PyResult<PyObject> {
    let n = chain.len().max(1) as f32;
    let mut counts = [0u32; 4];
    for b in chain.iter() {
        counts[(b % 4) as usize] += 1;
    }
    let fa = counts[0] as f32 / n;
    let fu = counts[1] as f32 / n;
    let fg = counts[2] as f32 / n;
    let fc = counts[3] as f32 / n;

    // norm: 0.25 (random) → 0.5; 0.0 → 0.0; 0.5+ → 1.0
    let norm = |freq: f32| (2.0 * freq).min(1.0);
    let lerp = |lo: f32, hi: f32, f: f32| lo + (hi - lo) * norm(f);

    let dict = PyDict::new_bound(py);
    // Hard parameters (purines A, G)
    dict.set_item("field_interaction",        lerp(-1.0, 1.0, fg))?;
    dict.set_item("replication_threshold",    lerp( 0.5, 4.0, fa))?;
    dict.set_item("interaction_mode",         lerp(-1.0, 1.0, fa))?;
    dict.set_item("conversion_threshold",     lerp( 0.0, 1.0, fg))?;
    dict.set_item("interaction_threshold",    lerp( 0.0, 1.0, fa))?;
    dict.set_item("cooperation_threshold",    lerp( 0.0, 1.0, fg))?;
    // Soft parameters (pyrimidines U, C)
    dict.set_item("movement_response",        lerp( 0.0, 2.0, fu))?;
    dict.set_item("aging_resistance",         lerp( 0.0, 2.0, fc))?;
    dict.set_item("replication_energy_split", lerp( 0.3, 0.7, fu))?;
    dict.set_item("atp_absorption_rate",      lerp( 0.05, 0.25, fc))?;
    dict.set_item("inhibitor_sensitivity",    lerp( 0.0, 0.1, fu))?;
    dict.set_item("chemotaxis_gene_strength", lerp( 0.0, 0.15, fc))?;
    Ok(dict.into())
}

/// Count A/U/G/C bases in a chain. Returns (n_a, n_u, n_g, n_c).
#[pyfunction]
fn base_counts(chain: Vec<u8>) -> (u32, u32, u32, u32) {
    let mut counts = [0u32; 4];
    for b in chain.iter() {
        counts[(b % 4) as usize] += 1;
    }
    (counts[0], counts[1], counts[2], counts[3])
}

/// Convert a chain (u8 list) to its human-readable string: "AUGCAU...".
#[pyfunction]
fn chain_to_string(chain: Vec<u8>) -> String {
    chain.iter().map(|b| "AUGC".chars().nth((b % 4) as usize).unwrap()).collect()
}

// ── ChainWorld (Level 2 foundation) ──────────────────────────────────
// Simulation object for later phases: full chem_sim runs, multiple
// chains, ATP atoms, metabolism. Right now only exposes basic lifecycle.

#[pyclass]
struct ChainWorld {
    inner: World,
}

#[pymethods]
impl ChainWorld {
    #[new]
    #[pyo3(signature = (world_size=120.0, seed=42))]
    fn new(world_size: f32, seed: u64) -> Self {
        let mut cfg = WorldConfig::default();
        cfg.world_size = world_size;
        cfg.physics.world_size = world_size;
        cfg.cell_size = cfg.physics.r_repulse.max(cfg.reaction.reaction_radius);
        cfg.seed = seed;
        Self {
            inner: World::seeded(cfg),
        }
    }

    #[getter]
    fn time_step(&self) -> u64 {
        self.inner.time_step
    }

    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.atoms.iter().filter(|a| a.alive).count()
    }

    /// Inject a pre-formed chain of given sequence at center (cx, cy).
    #[pyo3(signature = (sequence, cx=60.0, cy=60.0, spacing=12.0))]
    fn inject_chain(&mut self, sequence: Vec<u8>, cx: f32, cy: f32, spacing: f32) {
        let bases = bases_from_u8_list(&sequence);
        self.inner.inject_chain(&bases, cx, cy, spacing);
    }

    /// Add `n_per_type` free monomers of each of the 4 base types.
    fn inject_free_monomers(&mut self, n_per_type: usize) {
        self.inner.inject_free_monomers(n_per_type);
    }

    /// Advance one time step.
    #[pyo3(signature = (dt=1.0))]
    fn step(&mut self, dt: f32) {
        self.inner.step(dt);
    }

    /// Advance `n` time steps.
    #[pyo3(signature = (n, dt=1.0))]
    fn step_many(&mut self, n: usize, dt: f32) {
        for _ in 0..n {
            self.inner.step(dt);
        }
    }

    /// Get all current chain sequences as a list of u8 lists.
    fn get_chains(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty_bound(py);
        for a in self.inner.atoms.iter() {
            if !a.alive {
                continue;
            }
            if a.backbone_prev.is_none() && a.backbone_next.is_some() {
                let mut seq: Vec<u8> = Vec::new();
                let mut cur = a.id.as_index();
                let mut len = 0;
                loop {
                    if cur >= self.inner.atoms.len() {
                        break;
                    }
                    let c = &self.inner.atoms[cur];
                    seq.push(c.ctype.as_u8());
                    len += 1;
                    match c.backbone_next {
                        Some(next) => cur = next.as_index(),
                        None => break,
                    }
                    if len > 10_000 {
                        break;
                    }
                }
                list.append(PyList::new_bound(py, &seq))?;
            }
        }
        Ok(list.into())
    }

    /// Compact statistics snapshot: dict with pair_events, link_events,
    /// release_events, replication_events, polymerize_events, break_events,
    /// mutation_events, n_atoms, n_chains.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("pair_events",         self.inner.stats.pair_events)?;
        dict.set_item("link_events",         self.inner.stats.link_events)?;
        dict.set_item("release_events",      self.inner.stats.release_events)?;
        dict.set_item("replication_events",  self.inner.stats.replication_events)?;
        dict.set_item("polymerize_events",   self.inner.stats.polymerize_events)?;
        dict.set_item("break_events",        self.inner.stats.break_events)?;
        dict.set_item("mutation_events",     self.inner.stats.mutation_events)?;
        let alive: Vec<_> = self.inner.atoms.iter().filter(|a| a.alive).collect();
        dict.set_item("n_atoms",  alive.len())?;
        let n_chains = alive.iter()
            .filter(|a| a.backbone_prev.is_none() && a.backbone_next.is_some())
            .count();
        dict.set_item("n_chains", n_chains)?;
        Ok(dict.into())
    }
}

// ── Module definition ────────────────────────────────────────────────

#[pymodule]
fn chem_sim(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replicate_chain, m)?)?;
    m.add_function(wrap_pyfunction!(phenotypes_from_chain, m)?)?;
    m.add_function(wrap_pyfunction!(base_counts, m)?)?;
    m.add_function(wrap_pyfunction!(chain_to_string, m)?)?;
    m.add_class::<ChainWorld>()?;
    Ok(())
}
