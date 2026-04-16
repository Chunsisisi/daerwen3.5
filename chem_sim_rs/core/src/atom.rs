//! Atom data structures.
//!
//! Design principle: an atom's "state" is fully derived from its bond
//! configuration. There is NO separate `state` field — that was a redundant
//! abstraction inherited from the prototype that allowed bond reality and
//! state to drift, producing "phantom" states that broke the chemistry.
//!
//! All role queries (`is_free`, `in_chain`, `is_replicating`) are pure
//! functions of `backbone_prev`, `backbone_next`, and `template_bond`.

use serde::{Deserialize, Serialize};

pub const N_BASES: u8 = 4;

/// RNA-style base type. Layout matches A=0, U=1, G=2, C=3 with Watson-Crick
/// complementary pairs A↔U and G↔C.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BaseType {
    A = 0,
    U = 1,
    G = 2,
    C = 3,
}

impl BaseType {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v % N_BASES {
            0 => Self::A,
            1 => Self::U,
            2 => Self::G,
            _ => Self::C,
        }
    }
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
    #[inline]
    pub fn complement(self) -> Self {
        match self {
            Self::A => Self::U,
            Self::U => Self::A,
            Self::G => Self::C,
            Self::C => Self::G,
        }
    }
    #[inline]
    pub fn label(self) -> char {
        match self {
            Self::A => 'A',
            Self::U => 'U',
            Self::G => 'G',
            Self::C => 'C',
        }
    }
}

/// Lookup table form of complement, for symmetry with paper notation.
pub const COMPLEMENT: [BaseType; 4] = [BaseType::U, BaseType::A, BaseType::C, BaseType::G];

/// Stable handle to an atom in the world. Indices are not reused after death.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AtomId(pub u32);

impl AtomId {
    pub const NONE: AtomId = AtomId(u32::MAX);
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }
    #[inline]
    pub fn as_index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub id:    AtomId,
    pub ctype: BaseType,
    pub pos:   [f32; 2],
    pub vel:   [f32; 2],
    /// Persistent backbone bond toward the previous atom in the chain (head direction).
    pub backbone_prev: Option<AtomId>,
    /// Persistent backbone bond toward the next atom in the chain (tail direction).
    pub backbone_next: Option<AtomId>,
    /// Transient template bond (replication-time only). Symmetric.
    pub template_bond: Option<AtomId>,
    pub alive: bool,
}

impl Atom {
    pub fn new(id: AtomId, ctype: BaseType, pos: [f32; 2]) -> Self {
        Self {
            id,
            ctype,
            pos,
            vel: [0.0, 0.0],
            backbone_prev: None,
            backbone_next: None,
            template_bond: None,
            alive: true,
        }
    }

    // ── Derived role queries (the only "state" — read from bonds) ────────

    /// No bonds of any kind: a free monomer drifting in solution.
    #[inline]
    pub fn is_free(&self) -> bool {
        self.backbone_prev.is_none()
            && self.backbone_next.is_none()
            && self.template_bond.is_none()
    }

    /// Has at least one backbone bond (is part of a chain).
    #[inline]
    pub fn in_chain(&self) -> bool {
        self.backbone_prev.is_some() || self.backbone_next.is_some()
    }

    /// Has a template bond — currently participating in a replication event,
    /// either as template (also in_chain) or as a daughter atom being assembled.
    #[inline]
    pub fn is_replicating(&self) -> bool {
        self.template_bond.is_some()
    }

    /// True iff the atom has no backbone bonds. Free monomers AND unlinked
    /// paired daughter atoms both satisfy this — distinguish via `is_replicating`.
    #[inline]
    pub fn has_no_backbone(&self) -> bool {
        self.backbone_prev.is_none() && self.backbone_next.is_none()
    }

    /// Human-readable label for debugging.
    pub fn label(&self) -> String {
        let role = if self.is_free() {
            "Free"
        } else if self.is_replicating() && self.in_chain() {
            "Template"
        } else if self.is_replicating() {
            "Paired"
        } else {
            "InChain"
        };
        format!("{}:{}", self.ctype.label(), role)
    }
}
