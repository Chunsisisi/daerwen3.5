"""
Chemistry-grounded gene expression layer.

Replaces the 12 hand-crafted gene→phenotype formulas in core.py's
`_express_genes` with a single uniform principle:

    phenotype_i = min_i + range_i * f_i(base_frequencies)

where f_i is a simple linear combination of base frequencies (A/U/G/C)
in the gene chain. The only designer choice is "which base controls which
phenotype" — much smaller than the previous "12 different math functions".

Conceptually, this treats each particle's genome as a chemical chain of
A/U/G/C atoms (matching chem_sim_rs semantics). Replication uses
template-style copying with mutation. Phenotype is composition.

This is the MINIMUM VIABLE integration of chem_sim concepts into DAERWEN.
A future version (pyo3 binding to chem_sim_rs) would replace this Python
emulation with the actual Rust chemistry.
"""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np


# Base alphabet: 0=A, 1=U, 2=G, 3=C  (matches chem_sim_rs)
N_BASES = 4
COMPLEMENT = {0: 1, 1: 0, 2: 3, 3: 2}  # A↔U, G↔C


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
        normalized = clip(2 * freq, 0, 1)         # 0.25 random → 0.5 mid
        value = MIN + (MAX - MIN) * normalized

    For random genomes, every base has freq ≈ 0.25, so normalized ≈ 0.5,
    which is the midpoint of every phenotype range. This means a population
    of random genotypes starts at the *centre* of phenotype space, and
    selection can push values toward either extreme (more or less of a base
    in the genome).

    There is no `tanh`, `sin`, `sigmoid`, or `std` selectively applied to
    specific segments — those choices were the Designer's Trap. The single
    uniform principle here is: phenotype = position-on-axis(base frequency).

    Designer choice that remains: which base controls which phenotype.
    A future fully-emergent version would derive this from chemistry too.
    """
    f = base_frequencies(genome)

    def norm(freq: float) -> float:
        """0.25 (random) → 0.5; 0.0 → 0.0; 0.5+ → 1.0"""
        return min(1.0, 2.0 * freq)

    def lerp(low: float, high: float, freq: float) -> float:
        return low + (high - low) * norm(freq)

    return {
        # Hard parameters (driven by purines A, G)
        'field_interaction':         lerp(-1.0,  1.0, f['G']),
        'replication_threshold':     lerp( 0.5,  4.0, f['A']),
        'interaction_mode':          lerp(-1.0,  1.0, f['A']),
        'conversion_threshold':      lerp( 0.0,  1.0, f['G']),
        'interaction_threshold':     lerp( 0.0,  1.0, f['A']),
        'cooperation_threshold':     lerp( 0.0,  1.0, f['G']),
        # Soft parameters (driven by pyrimidines U, C)
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

    In chem_sim's R_pair, each new-strand atom is recruited based on the
    template-strand atom: faithful (same type) or complementary (A↔U, G↔C).
    Mutations occur at the recruitment step.

    Faithful mode produces direct copies (with errors).
    Complementary mode requires two replication rounds to recover the
    original sequence, doubling effective mutation pressure per generation.
    """
    parent = np.asarray(parent_genome, dtype=np.int8)
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
    return ''.join('AUGC'[int(b) % N_BASES] for b in genome)
