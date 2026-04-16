//! Lightweight statistical snapshot for inspection / serialization.

use crate::atom::Atom;
use crate::rules::ReactionStats;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub time_step: u64,
    pub n_atoms:   usize,
    pub n_free:    usize,
    pub n_chains:  usize,
    pub chain_lengths: HashMap<usize, usize>,
    pub sequence_counts: HashMap<String, usize>,
    pub dominant_sequence: Option<String>,
    pub diversity:         f64,
    pub stats: ReactionStats,
}

pub fn build_snapshot(atoms: &[Atom], stats: &ReactionStats, time_step: u64) -> Snapshot {
    let alive: Vec<&Atom> = atoms.iter().filter(|a| a.alive).collect();
    let n_atoms = alive.len();
    // "Free" for snapshot purposes: any atom with no backbone bonds.
    // (We don't subdivide into truly free vs replicating-but-unlinked.)
    let n_free  = alive.iter().filter(|a| a.has_no_backbone()).count();

    let mut chain_lengths: HashMap<usize, usize> = HashMap::new();
    let mut sequence_counts: HashMap<String, usize> = HashMap::new();
    let mut n_chains = 0;
    for a in alive.iter() {
        // A chain head: no backbone_prev, has backbone_next.
        if a.backbone_prev.is_none() && a.backbone_next.is_some() {
            let mut seq = String::new();
            let mut cur_idx = a.id.as_index();
            let mut len = 0;
            loop {
                let cur = &atoms[cur_idx];
                seq.push(cur.ctype.label());
                len += 1;
                match cur.backbone_next {
                    Some(next) => cur_idx = next.as_index(),
                    None => break,
                }
                if len > 10_000 {
                    break;
                }
            }
            *chain_lengths.entry(len).or_insert(0) += 1;
            *sequence_counts.entry(seq).or_insert(0) += 1;
            n_chains += 1;
        }
    }

    let total: usize = sequence_counts.values().sum();
    let diversity: f64 = if total > 1 {
        sequence_counts
            .values()
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.ln()
            })
            .sum()
    } else {
        0.0
    };
    let dominant_sequence = sequence_counts
        .iter()
        .max_by_key(|(_, c)| *c)
        .map(|(s, _)| s.clone());

    Snapshot {
        time_step,
        n_atoms,
        n_free,
        n_chains,
        chain_lengths,
        sequence_counts,
        dominant_sequence,
        diversity,
        stats: stats.clone(),
    }
}

/// Number of atoms currently participating in a replication event
/// (i.e., have a template_bond).
pub fn active_replication_count(atoms: &[Atom]) -> usize {
    atoms.iter().filter(|a| a.alive && a.is_replicating()).count()
}

/// Type-frequency histogram of currently alive atoms.
pub fn type_histogram(atoms: &[Atom]) -> [usize; 4] {
    let mut h = [0usize; 4];
    for a in atoms.iter().filter(|a| a.alive) {
        h[a.ctype as usize] += 1;
    }
    h
}
