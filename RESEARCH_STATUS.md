# Research Status & Open Bottlenecks

> Last updated: **2026-04-16**
> Status: **Active research with publicly acknowledged architectural limit**

---

## Why this document exists

This project makes ambitious claims (a physics-grounded substrate for AGI). Some of those claims are validated. Some are aspirational. At least one fundamental architectural limit is now clearly visible.

**Rather than hiding the limit, this document records it.** The intent is twofold:

1. To give anyone landing on the repository an honest picture of where research stands today
2. To make the open problem visible enough that collaborators (or future-me) can pick it up

This document will be updated as work progresses. The git history of the file is itself the record of how the project has evolved.

---

## What is currently validated

| Component | Status | Evidence |
|-----------|--------|----------|
| **2D ecology engine** (Python, daerwen3.5/engine/) | ✅ Works | runs at 24/7, supports external inputs, emits structured output |
| **Continual learning resilience** | ⚠ Partial | very low forgetting (0.014); but absolute task scores low (F-grade benchmark) |
| **Noise / catastrophe robustness** | ✅ Validated | 0.88-0.98 noise resilience, 1.0 catastrophe recovery |
| **chem_sim_rs (Rust)** — template-directed replication | ✅ Validated | abiogenesis from pure free monomers: 14k complete replications |
| **Gene expression with reduced designer-prior** | ✅ Implemented | 12 hand-crafted formulas replaced with one uniform composition-based mapping (less Designer's Trap) |
| **Performance (single-thread)** | ✅ Validated | ~1.5M atom-steps/sec in chem_sim_rs |

## What is not validated

| Claim | Reality |
|-------|---------|
| "Chemistry-grounded genes" | ✅ Partially true (chem_sim_rs handles template replication of gene chains) |
| "Chemistry-grounded ecology" | ❌ False — energy, metabolism, particle bodies are still Python abstractions |
| "Full physical substrate for AGI" | ❌ Aspirational — see *The Chemistry Substrate Bottleneck* below |
| "Beats neural-net continual-learning benchmarks" | ❌ Currently F-grade on standard benchmarks (we explain why) |
| "Pure emergence with zero designer priors" | ❌ Impossible in principle; we minimize but cannot eliminate priors |

---

## The Chemistry Substrate Bottleneck

**This is the open architectural problem.**

`chem_sim_rs` is a rule-based artificial chemistry inspired by Hutton's squirm3 (2002). It successfully demonstrates:

- Spontaneous bond formation from free monomers
- Template-directed copying (the abiogenesis pathway)
- Self-sustaining chemical equilibrium
- Mutation and inheritance of sequence information

But it has six explicit reaction rules (`R_pair`, `R_link`, `R_release`, `R_stale`, `R_break`, `R_polymerize`) with hand-tuned rate constants. **This works for the narrow case of chain replication, but does not scale to general biochemistry.**

### A concrete failure case: photosynthesis

The simplest representative biological process — converting light + water + CO₂ to sugar + O₂ — requires:

- An external energy source (photons)
- Differentiated atom roles (C/O/H, not just A/U/G/C placeholders)
- Multi-step reaction networks (chlorophyll absorbs → electron transport → ATP synthesis → carbon fixation)
- Energy stored in covalent bonds and released later

`chem_sim_rs` cannot do any of this without adding more hand-coded rules. And there are thousands of such enzyme-catalyzed reactions in even the simplest cell. **Adding one rule per process re-introduces exactly the Designer's Trap we set out to avoid.**

### Why this matters

DAERWEN's central philosophical commitment is to minimize designer-imposed priors. Adding a rule for every biological process violates this commitment. The substrate, as currently designed, is therefore a **demonstration of one mechanism**, not a general substrate.

---

## Phase C: Physics-grounded chemistry (the path forward)

The principled solution is to step down one level of abstraction: replace explicit reaction rules with continuous **potential energy surfaces**.

In this model:

- Atoms are particles with mass, position, velocity, partial charge, and type
- Pairwise potentials (Morse, Lennard-Jones, Coulomb) define all interactions
- Bonds form and break smoothly when atoms cross energy thresholds
- Catalysis emerges because chain geometries lower activation energies for nearby reactions
- Photosynthesis emerges because photons carry energy that can be stored in bonds
- All biology becomes implicit in physics

### What this requires

- **Reading**: Allen & Tildesley *Computer Simulation of Liquids*, Frenkel & Smit *Understanding Molecular Simulation*, the ReaxFF and MARTINI papers, Hutton 2007 (functional cell)
- **Implementation**: a coarse-grained molecular dynamics engine, written from scratch in Rust on top of the existing `chem_sim_rs` workspace skeleton
- **Validation**: must show template catalysis (or some equivalent emergent self-replication) **without** any rule explicitly encoding it

### Honest assessment

This is **6 to 12 months of focused research-level work**, with no guarantee that emergent template catalysis will appear under any tunable parameter regime. Many groups have attempted variants of this — most ended up adding some abstraction (Hutton 2007 added cells; Penny added membranes; Fontana used lambda calculus). A genuinely abstraction-free demonstration would be publishable in a top-tier venue, but the probability of getting there is realistically 10–20% per attempt.

### Why pursue it anyway

- The DAERWEN narrative ("physics-grounded substrate underneath symbolic intelligence") only fully holds if the substrate is actually physics-grounded
- Even partial progress (cells, membranes, energy carriers) would be valuable
- Negative results — "we tried X, here is what we learned about why it doesn't work" — are still scientific contributions

---

## Pragmatic interim architecture

Until Phase C is complete (or proven infeasible), the project uses a **layered architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Python ecology layer (daerwen3.5/engine/core.py)               │
│  - Particles, energy fields, ATP, metabolism, ecological dynamics│
│  - Energy is abstracted as floating-point numbers                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────▼────────────────┐
                │  Gene expression layer        │
                │  (engine/chem_sim_genes.py)   │
                │  Uniform composition mapping  │
                │  ← reduced Designer's Trap    │
                └──────────────┬────────────────┘
                               │
                ┌──────────────▼────────────────┐
                │  chem_sim_rs (Rust)           │
                │  Template replication of      │
                │  RNA-like chain genes         │
                │  ← chemistry-grounded         │
                └───────────────────────────────┘
```

This is honest about which parts are chemistry and which parts remain abstract.

---

## Recent changes (chronological log)

- **2026-04-16**: this document created
- **2026-04-16**: identified architectural ceiling of `chem_sim_rs` (cannot scale to general biology)
- **2026-04-16**: replaced 12 hand-crafted gene formulas with uniform composition-based mapping; viable smoke test (population survives, evolution accelerates per-generation, smaller equilibrium population)
- **2026-04-16**: classical molecular evolution experiments (E1-E6) implemented in Rust; only E4 (logistic growth) reproduces the textbook result; others fail because spontaneous polymerization dominates over template-directed replication
- **2026-04-16**: abiogenesis validated — chemistry bootstraps from pure free monomers (14,000 complete replications, 736 stable chains in equilibrium)
- **2026-04-16**: gene expression state-machine refactored to derive role purely from bond configuration (eliminated entire class of state-bond-drift bugs)
- **2026-04-16**: chem_sim_rs Rust workspace created; Hutton 2002 squirm3 concepts independently re-implemented under AGPL-3.0 (compatible with squirm3's GPL-3.0)
- **2026-04-16**: initial public release of daerwen3.5 with Zenodo DOI [10.5281/zenodo.19604736](https://doi.org/10.5281/zenodo.19604736)

---

## Open call

If you have expertise in any of:

- Molecular dynamics simulation (especially reactive force fields)
- Artificial chemistry / artificial life from first principles
- Origin-of-life modelling
- Self-organising systems / autocatalytic networks
- Rust systems programming

…and find any of this interesting, please open an issue or get in touch via GitHub. This is a long-haul research effort and collaboration is welcome.

---

## Funding & affiliation

**None.** This project is self-funded and self-directed. The author (Hou Zehao) is an independent researcher, not affiliated with any university or company. All work is done with AI agent assistance under human direction. There are no commercial obligations and no institutional pressures shaping the research direction.

This means progress is bounded by one person's available time. It also means the project can take long-shot research bets that institutional projects typically cannot.

---

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

The chem_sim_rs Rust implementation is similarly AGPL-3.0-or-later, compatible with the GPL-3.0 of Tim Hutton's [squirm3](https://github.com/timhutton/squirm3) which inspired the rule-based chemistry approach.
