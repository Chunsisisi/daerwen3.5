//! chem_sim_core
//!
//! Continuous-space artificial chemistry with chain-based template replication.
//!
//! Concepts inspired by Hutton (2002) "Evolvable Self-Replicating Molecules in an
//! Artificial Chemistry" — independently re-implemented from the paper's stated
//! mechanism, not from the original C++ source. The chain-based replication state
//! machine extends Hutton's dimer model to arbitrary-length molecules and is the
//! original work of this project.
//!
//! License: AGPL-3.0-or-later (compatible with GPL-3.0 of the squirm3 reference).

pub mod atom;
pub mod physics;
pub mod rules;
pub mod snapshot;
pub mod spatial;
pub mod world;

pub use atom::{Atom, AtomId, BaseType, COMPLEMENT, N_BASES};
pub use snapshot::Snapshot;
pub use world::{World, WorldConfig};
