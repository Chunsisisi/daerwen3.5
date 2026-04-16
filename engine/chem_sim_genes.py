"""
Chemistry-grounded gene expression layer.

This module is now a thin Python wrapper around the actual Rust chem_sim
implementation (compiled via maturin/pyo3 from chem_sim_rs/python/).

If the Rust module is unavailable for some reason, falls back to a pure-
Python emulation so the rest of DAERWEN keeps working — but the fallback
is slower and approximate. Real production runs should use the Rust module.

API:
  express_phenotypes_from_composition(genome) -> dict
  replicate_chain_with_mutation(parent, rng, mutation_rate, complementary) -> ndarray
  base_frequencies(genome) -> dict
  chain_to_string(genome) -> str
"""
from __future__ import annotations
from typing import Dict
import numpy as np

# ── Try to use the Rust chem_sim module ──────────────────────────────
_USING_RUST = False
try:
    import chem_sim as _rust
    _USING_RUST = True
except ImportError:
    _rust = None


# Base alphabet: 0=A, 1=U, 2=G, 3=C
N_BASES = 4
COMPLEMENT = {0: 1, 1: 0, 2: 3, 3: 2}


def is_using_rust() -> bool:
    return _USING_RUST


def base_frequencies(genome: np.ndarray) -> Dict[str, float]:
    """Return normalized A/U/G/C frequencies in the genome."""
    n = max(1, len(genome))
    return {
        'A': float(np.sum(genome == 0)) / n,
        'U': float(np.sum(genome == 1)) / n,
        'G': float(np.sum(genome == 2)) / n,
        'C': float(np.sum(genome == 3)) / n,
    }


def express_phenotypes_from_composition(
    genome: np.ndarray,
    config: object | None = None,
) -> Dict[str, float]:
    """
    Compute the 12 DAERWEN phenotypes from base composition.

    All formulas use the same uniform shape:
        normalized = clip(2 * freq, 0, 1)
        value = MIN + (MAX - MIN) * normalized

    For random genomes, every base has freq ≈ 0.25, so normalized ≈ 0.5,
    placing every phenotype at the midpoint of its range.
    """
    if _USING_RUST:
        # Convert numpy array to a list of u8 for Rust
        chain_list = [int(b) % 4 for b in genome]
        return dict(_rust.phenotypes_from_chain(chain_list))

    # ── Python fallback ────────────────────────────────────────────────
    f = base_frequencies(genome)

    def norm(freq: float) -> float:
        return min(1.0, 2.0 * freq)

    def lerp(low: float, high: float, freq: float) -> float:
        return low + (high - low) * norm(freq)

    return {
        'field_interaction':         lerp(-1.0,  1.0, f['G']),
        'replication_threshold':     lerp( 0.5,  4.0, f['A']),
        'interaction_mode':          lerp(-1.0,  1.0, f['A']),
        'conversion_threshold':      lerp( 0.0,  1.0, f['G']),
        'interaction_threshold':     lerp( 0.0,  1.0, f['A']),
        'cooperation_threshold':     lerp( 0.0,  1.0, f['G']),
        'movement_response':         lerp( 0.0,  2.0, f['U']),
        'aging_resistance':          lerp( 0.0,  2.0, f['C']),
        'replication_energy_split':  lerp( 0.3,  0.7, f['U']),
        'atp_absorption_rate':       lerp( 0.05, 0.25, f['C']),
        'inhibitor_sensitivity':     lerp( 0.0,  0.1, f['U']),
        'chemotaxis_gene_strength':  lerp( 0.0,  0.15, f['C']),
    }


def replicate_chain_with_mutation(
    parent_genome: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.01,
    complementary: bool = False,
) -> np.ndarray:
    """
    Template-style copy with optional Watson-Crick complementarity.

    Uses Rust chem_sim if available (faster, deterministic seed via numpy rng).
    Falls back to pure Python if not.
    """
    parent = np.asarray(parent_genome, dtype=np.int8)
    if _USING_RUST:
        # Generate a deterministic seed from the rng so Rust is reproducible
        seed = int(rng.integers(0, 2**63 - 1))
        chain_list = [int(b) % 4 for b in parent]
        offspring_list = _rust.replicate_chain(
            chain_list,
            mutation_rate=float(mutation_rate),
            complementary=complementary,
            seed=seed,
        )
        return np.asarray(offspring_list, dtype=np.int8)

    # ── Python fallback ────────────────────────────────────────────────
    offspring = np.empty_like(parent)
    for i, base in enumerate(parent):
        target = COMPLEMENT[int(base)] if complementary else int(base)
        if rng.random() < mutation_rate:
            offspring[i] = int(rng.integers(0, N_BASES))
        else:
            offspring[i] = target
    return offspring


def chain_to_string(genome: np.ndarray) -> str:
    """Human-readable chain representation."""
    if _USING_RUST:
        return _rust.chain_to_string([int(b) % N_BASES for b in genome])
    return ''.join('AUGC'[int(b) % N_BASES] for b in genome)
