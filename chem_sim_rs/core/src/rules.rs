//! Reaction rules. Each atom can fire at most one rule per step (first-match
//! by descending priority): R_link > R_release > R_pair > R_stale > R_break > R_polymerize.
//!
//! All conditions are pure structural queries on bonds — there is no `state`
//! field to drift out of sync. Adding or removing a bond automatically updates
//! an atom's role, with no manual synchronization step required.

use crate::atom::{Atom, AtomId, BaseType, COMPLEMENT};
use crate::spatial::dist_sq_toroidal;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionConfig {
    /// Whether replication produces a faithful copy (same type) or a
    /// Watson-Crick complementary copy (A↔U, G↔C).
    pub complementary: bool,
    /// Per-base mutation probability when R_pair fires.
    pub mutation_rate: f32,
    /// Reaction radius (atoms must be within this distance to react).
    pub reaction_radius: f32,
    /// Per-step probability of each rule (multiplied by dt).
    pub rate_pair:    f32,
    pub rate_link:    f32,
    pub rate_release: f32,
    pub rate_stale:   f32,
    pub rate_break:   f32,
    pub rate_polymerize: f32,
}

impl Default for ReactionConfig {
    fn default() -> Self {
        Self {
            complementary:   false,
            mutation_rate:   0.01,
            reaction_radius: 12.0,
            rate_pair:        0.30,
            rate_link:        0.70,
            rate_release:     0.15,
            rate_stale:       0.05,
            rate_break:       0.002,
            // Template catalysis claim: R_pair is ~300x faster than
            // spontaneous polymerization. This is a physical-chemistry
            // assertion, not a tunable knob.
            rate_polymerize:  0.001,
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ReactionStats {
    pub pair_events:        u64,
    pub link_events:        u64,
    pub release_events:     u64,
    pub stale_events:       u64,
    pub break_events:       u64,
    pub polymerize_events:  u64,
    pub mutation_events:    u64,
    pub replication_events: u64,
}

/// Try to fire one rule for atom at `aid`, returning true if any rule fired.
pub fn try_react<R: Rng>(
    atoms:     &mut Vec<Atom>,
    aid:       AtomId,
    neighbors: &[AtomId],
    world_size: f32,
    cfg:       &ReactionConfig,
    stats:     &mut ReactionStats,
    rng:       &mut R,
    dt:        f32,
) -> bool {
    let i = aid.as_index();
    if i >= atoms.len() || !atoms[i].alive {
        return false;
    }

    // Snapshot atom-i fields up front (cheap copies) to avoid borrow conflicts
    // when mutating the atom alongside a neighbor later.
    let a_pos         = atoms[i].pos;
    let a_ctype       = atoms[i].ctype;
    let a_template    = atoms[i].template_bond;
    let a_back_prev   = atoms[i].backbone_prev;
    let a_back_next   = atoms[i].backbone_next;
    let a_in_chain    = atoms[i].in_chain();
    let a_is_free     = atoms[i].is_free();
    let a_replicating = atoms[i].is_replicating();
    let a_no_backbone = atoms[i].has_no_backbone();

    let r2 = cfg.reaction_radius * cfg.reaction_radius;

    // ── Priority 1: R_link ─────────────────────────────────────────────
    // I have a template_bond. My template has a backbone neighbor.
    // Find a neighbor with a template_bond to that neighbor, whose matching
    // backbone slot is empty, and bond us.
    if a_replicating {
        let t_i = a_template.unwrap().as_index();
        if t_i < atoms.len() && atoms[t_i].alive {
            // Forward direction: my template has a next; I want a next.
            if let Some(t_next_id) = atoms[t_i].backbone_next {
                if a_back_next.is_none() {
                    for &bid in neighbors {
                        let j = bid.as_index();
                        if j == i || j >= atoms.len() || !atoms[j].alive {
                            continue;
                        }
                        let b = &atoms[j];
                        if b.template_bond == Some(t_next_id)
                            && b.backbone_prev.is_none()
                            && dist_sq_toroidal(a_pos, b.pos, world_size) <= r2
                            && rng.gen::<f32>() < cfg.rate_link * dt
                        {
                            bond_backbone_next(atoms, aid, bid);
                            stats.link_events += 1;
                            return true;
                        }
                    }
                }
            }
            // Reverse direction: my template has a prev; I want a prev.
            if let Some(t_prev_id) = atoms[t_i].backbone_prev {
                if a_back_prev.is_none() {
                    for &bid in neighbors {
                        let j = bid.as_index();
                        if j == i || j >= atoms.len() || !atoms[j].alive {
                            continue;
                        }
                        let b = &atoms[j];
                        if b.template_bond == Some(t_prev_id)
                            && b.backbone_next.is_none()
                            && dist_sq_toroidal(a_pos, b.pos, world_size) <= r2
                            && rng.gen::<f32>() < cfg.rate_link * dt
                        {
                            bond_backbone_next(atoms, bid, aid);
                            stats.link_events += 1;
                            return true;
                        }
                    }
                }
            }
        }
    }

    // ── Priority 2: R_release (productive) ────────────────────────────
    // I have a template_bond, and either side is anchored in a chain (so
    // letting go produces a stable chain on each side).
    if a_replicating {
        let b_id = a_template.unwrap();
        let j = b_id.as_index();
        if j < atoms.len() && atoms[j].alive {
            let b_in_chain = atoms[j].in_chain();
            let productive = a_in_chain && b_in_chain;
            if productive && rng.gen::<f32>() < cfg.rate_release * dt {
                unbond_template(atoms, aid);
                // If a daughter-side chain has completed (all InChain, no
                // template_bonds, length ≥ 2), count as a replication event.
                if check_complete_chain(atoms, b_id, 2) {
                    stats.replication_events += 1;
                }
                stats.release_events += 1;
                return true;
            }
        }
    }

    // ── Priority 3: R_pair ────────────────────────────────────────────
    // An in-chain atom without a current template recruits a matching Free
    // monomer to start a new daughter atom.
    if a_in_chain
        && !a_replicating
        && rng.gen::<f32>() < cfg.rate_pair * dt
    {
        let required_ctype = if cfg.complementary {
            COMPLEMENT[a_ctype.as_u8() as usize]
        } else {
            a_ctype
        };
        for &bid in neighbors {
            let j = bid.as_index();
            if j == i || j >= atoms.len() || !atoms[j].alive {
                continue;
            }
            let b = &atoms[j];
            // Free = no bonds of any kind
            if b.is_free()
                && b.ctype == required_ctype
                && dist_sq_toroidal(a_pos, b.pos, world_size) <= r2
            {
                if rng.gen::<f32>() < cfg.mutation_rate {
                    let mutated = BaseType::from_u8(rng.gen_range(0..4));
                    atoms[j].ctype = mutated;
                    stats.mutation_events += 1;
                }
                bond_template(atoms, aid, bid);
                stats.pair_events += 1;
                return true;
            }
        }
    }

    // ── Priority 4: R_stale ───────────────────────────────────────────
    // Mutual-stuck release. If I have a template_bond but neither I nor my
    // partner has any chain bonds, R_link can never fire — break template to
    // let both atoms re-enter the free pool.
    if a_replicating && a_no_backbone {
        let b_id = a_template.unwrap();
        let j = b_id.as_index();
        if j < atoms.len() && atoms[j].alive && atoms[j].has_no_backbone() {
            if rng.gen::<f32>() < cfg.rate_stale * dt {
                unbond_template(atoms, aid);
                stats.stale_events += 1;
                return true;
            }
        }
    }

    // ── Priority 5: R_break ───────────────────────────────────────────
    // Spontaneous backbone break. Both sides must be pure chain atoms
    // (not currently serving as templates).
    if a_in_chain && !a_replicating && a_back_next.is_some() {
        let next_id = a_back_next.unwrap();
        let j = next_id.as_index();
        if j < atoms.len() && atoms[j].alive
            && !atoms[j].is_replicating()
            && rng.gen::<f32>() < cfg.rate_break * dt
        {
            unbond_backbone_next(atoms, aid);
            // No state to fix up — roles are derived from bonds.
            stats.break_events += 1;
            return true;
        }
    }

    // ── Priority 6: R_polymerize ─────────────────────────────────────
    // Spontaneous bond formation between two free monomers — the background
    // chemistry that lets the system bootstrap from pure monomers and recover
    // after all chains have broken.
    if a_is_free && rng.gen::<f32>() < cfg.rate_polymerize * dt {
        for &bid in neighbors {
            let j = bid.as_index();
            if j == i || j >= atoms.len() || !atoms[j].alive {
                continue;
            }
            let b = &atoms[j];
            if b.is_free() && dist_sq_toroidal(a_pos, b.pos, world_size) <= r2 {
                bond_backbone_next(atoms, aid, bid);
                stats.polymerize_events += 1;
                return true;
            }
        }
    }

    false
}

// ── Bond-mutation helpers ────────────────────────────────────────────

#[inline]
pub fn bond_backbone_next(atoms: &mut [Atom], a: AtomId, b: AtomId) {
    let (ai, bi) = (a.as_index(), b.as_index());
    debug_assert!(ai != bi, "bond_backbone_next: self-bond");
    debug_assert!(atoms[ai].backbone_next.is_none());
    debug_assert!(atoms[bi].backbone_prev.is_none());
    atoms[ai].backbone_next = Some(b);
    atoms[bi].backbone_prev = Some(a);
}

#[inline]
pub fn unbond_backbone_next(atoms: &mut [Atom], a: AtomId) {
    let ai = a.as_index();
    if let Some(bid) = atoms[ai].backbone_next {
        let bi = bid.as_index();
        atoms[ai].backbone_next = None;
        atoms[bi].backbone_prev = None;
    }
}

#[inline]
pub fn bond_template(atoms: &mut [Atom], a: AtomId, b: AtomId) {
    let (ai, bi) = (a.as_index(), b.as_index());
    debug_assert!(ai != bi, "bond_template: self-bond");
    debug_assert!(atoms[ai].template_bond.is_none());
    debug_assert!(atoms[bi].template_bond.is_none());
    atoms[ai].template_bond = Some(b);
    atoms[bi].template_bond = Some(a);
}

#[inline]
pub fn unbond_template(atoms: &mut [Atom], a: AtomId) {
    let ai = a.as_index();
    if let Some(bid) = atoms[ai].template_bond {
        let bi = bid.as_index();
        atoms[ai].template_bond = None;
        atoms[bi].template_bond = None;
    }
}

/// Walk from `start` toward the chain head and forward, checking that every
/// atom is in the chain, has no template_bond, and length ≥ `min_len`.
pub fn check_complete_chain(atoms: &[Atom], start: AtomId, min_len: usize) -> bool {
    let mut cur = start.as_index();
    let mut steps = 0;
    while let Some(prev) = atoms.get(cur).and_then(|a| a.backbone_prev) {
        cur = prev.as_index();
        steps += 1;
        if steps > 10_000 {
            return false;
        }
    }
    let mut len = 0;
    loop {
        if cur >= atoms.len() {
            return false;
        }
        let a = &atoms[cur];
        // A complete chain atom: alive, has at least one backbone bond (or is
        // a single-atom fragment that still counts as "in chain" only if len
        // ends up ≥ min_len), no active template.
        if !a.alive || a.template_bond.is_some() {
            return false;
        }
        len += 1;
        match a.backbone_next {
            Some(next) => cur = next.as_index(),
            None => break,
        }
        if len > 10_000 {
            return false;
        }
    }
    len >= min_len
}
