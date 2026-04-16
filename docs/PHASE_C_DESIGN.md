# Phase C Design: Physics-Grounded Artificial Chemistry

> **Status**: Initial design proposal · 2026-04-16
> **Purpose**: Replace `chem_sim_rs`'s explicit reaction rules with continuous potentials so that all chemistry — bond formation, breaking, catalysis, replication — emerges from physics rather than from designer-coded rules.
> **Scope**: 6-12 months of focused implementation work (estimated). Success probability ~10-20% for the decisive milestone (emergent template catalysis).

---

## 1. Executive Summary

### The plan in one paragraph

Build a coarse-grained molecular dynamics (MD) engine in Rust. Atoms are point particles with mass, position, velocity, type, and partial charge. Three pairwise potentials govern all interactions: **Morse** (covalent bonding), **Lennard-Jones** (van der Waals + steric repulsion), and **Coulomb** (electrostatic). Time integration via **velocity-Verlet**. Temperature controlled by a **Langevin thermostat**. Bonds are not stored — they are *detected* post-hoc from the topology of strongly-bound atom pairs. The decisive validation is whether template-directed self-replication emerges in such a system **without any rule explicitly encoding it**.

### Key technical commitments

- **Integrator**: velocity-Verlet (symplectic, energy-conserving in NVE; the standard choice)
- **Potentials**: Morse + Lennard-Jones + Coulomb (the minimal set that captures covalent + dispersion + electrostatic chemistry)
- **Neighbor finding**: cell list (O(N) average; appropriate for our scale)
- **Cutoffs**: LJ at 2.5σ (truncated and shifted, the de facto standard); Coulomb via reaction-field cutoff (Ewald is overkill at our scale)
- **Thermostat**: Langevin (handles the dissipative reactive system better than Nose-Hoover)
- **Bond detection**: post-hoc analysis only (distance + relative-velocity criterion); never used to drive dynamics
- **Atom representation**: Structure-of-Arrays (SoA) for SIMD-friendly access; 4-8 atom types parameterized

### Open questions that need user input

1. **Atom alphabet size**: 2 types (minimal viable), 4 types (RNA-like A/U/G/C), or arbitrary?
2. **System scale**: 200 atoms (fast iteration) or 2000 atoms (more emergent richness, ~100x slower)?
3. **Acceptance level for Designer's Trap**: zero (truly only Morse + LJ + Coulomb, derive everything) or limited (allow per-type-pair Morse parameters, which is itself a design choice)?
4. **Time horizon**: How long is the user willing to wait per simulation? This bounds the maximum integration time (and thus the slowest emergent behaviors observable).

---

## 2. Theoretical Foundation

### 2.1 Why physics, not rules

Real chemistry has no "reaction rules" stored as if-then logic. Reactions occur because the potential energy of the system can be lowered by bond rearrangement, and there exist pathways across the energy landscape where the activation barrier is low enough relative to thermal energy. Catalysts (enzymes, templates) are not special chemical entities — they are molecules whose three-dimensional structure happens to lower the activation barrier for a specific reaction by physically aligning the reactants.

If we want catalysis to emerge rather than be encoded, we must work at this level: continuous potentials, no explicit reactions, no priority-ordered rules.

### 2.2 Velocity-Verlet integration

The standard symplectic integrator for MD. Update sequence per step:

```
1.  v(t + Δt/2) = v(t) + (Δt/2) · a(t)              # half-kick
2.  x(t + Δt)   = x(t) + Δt · v(t + Δt/2)           # drift
3.  Compute a(t + Δt) from x(t + Δt)                 # force evaluation
4.  v(t + Δt)   = v(t + Δt/2) + (Δt/2) · a(t + Δt)  # half-kick
```

This is mathematically identical to (and computationally cheaper than) the original position-Verlet. **Symplectic** means it preserves phase-space volume, which gives bounded energy oscillation rather than secular drift — critical for long simulations.

**Time step selection**: must be small relative to the highest-frequency vibration. For Morse bonds with our parameters, expected highest frequency is ω = √(2 D_e a² / m). With Δt ≈ 1/(20 ω), this gives a usable step somewhere in the range 0.001-0.005 in reduced units. Conservative choice: start with Δt = 0.002.

**Stability test**: in NVE (no thermostat), total energy should oscillate with bounded amplitude — drift > 1% over 10⁵ steps means Δt is too large.

### 2.3 The three potentials

#### Morse (covalent bonding)

```
V_Morse(r) = D_e · (1 − exp(−a(r − r_e)))²
F_Morse(r) = −2 a D_e · (1 − exp(−a(r − r_e))) · exp(−a(r − r_e)) · (r̂)
```

Parameters per atom-pair-type:
- `D_e`: dissociation energy (well depth)
- `r_e`: equilibrium bond length
- `a`: well width (controls vibrational frequency)

Why Morse and not harmonic spring: **bonds must be able to break**. Harmonic potentials have infinite walls; Morse has a finite dissociation energy `D_e`. When kinetic energy exceeds `D_e`, the atoms separate — exactly the physics we want.

In real chemistry, `D_e` is fixed by the electronic structure of the atom pair. We cannot derive it from anything more fundamental at this abstraction level — **this is the irreducible design choice**: the table of `D_e` values for each atom-type-pair encodes our chemistry.

#### Lennard-Jones (van der Waals + steric repulsion)

```
V_LJ(r) = 4ε · [(σ/r)¹² − (σ/r)⁶]
F_LJ(r) = 24ε · [2(σ/r)¹² − (σ/r)⁶] · (r̂ / r)
```

Parameters:
- `ε`: well depth (~10× weaker than Morse `D_e`)
- `σ`: characteristic distance (zero crossing of V; minimum at r = 2^(1/6)σ ≈ 1.12σ)

This handles two things simultaneously: short-range steric repulsion (the r⁻¹² term — atoms can't overlap) and long-range dispersion attraction (the r⁻⁶ term — slight attraction at moderate distances).

**Cutoff**: standard 2.5σ truncated-and-shifted (the LJTS variant). At r = 2.5σ, V_LJ is already <2% of well depth — truncation introduces negligible error and saves substantial compute.

#### Coulomb (electrostatic)

```
V_C(r) = k_e · q_i · q_j / r
F_C(r) = k_e · q_i · q_j · (r̂ / r²)
```

Where `q_i` is the partial charge on atom `i`. **This is the key to template specificity**: differential electrostatic patterns mean some atom-type pairs attract from a distance, others repel. Watson-Crick complementarity in real biology is essentially a hydrogen-bond pattern, which at our coarse-grained level reduces to electrostatic attraction between atoms with appropriate partial charges.

**Long-range issue**: Coulomb is unscreened in vacuum. For 100-1000 atoms in a small box, full pairwise Coulomb is feasible (O(N²) but still fast). For larger systems, reaction-field cutoff or Ewald summation. **Decision**: start with full pairwise (simpler, correct), optimize later if needed.

### 2.4 Thermostat

Pure NVE (microcanonical, no thermostat) drifts due to numerical error and is inappropriate for reactive systems where energy is constantly redistributed. We need NVT (constant temperature).

**Choice: Langevin thermostat.** Each atom gets a stochastic friction force:

```
F_total = F_potential − γ m v + √(2 γ m k_B T / Δt) · η
```

Where `η` is a unit-normal random vector. This gives correct Boltzmann distribution at temperature T while providing the dissipation that reactive systems need.

Why Langevin over Nose-Hoover or Berendsen:
- **Berendsen**: not statistically correct (suppresses fluctuations)
- **Nose-Hoover**: deterministic, can have ergodicity problems in non-stiff systems
- **Langevin**: correct, ergodic, simple, robust

Friction coefficient γ should be small enough not to dominate dynamics (~ 0.1 / time_unit is a reasonable starting point).

---

## 3. Architecture

### 3.1 Atom data structure

Structure-of-Arrays (SoA) for SIMD-friendly access:

```rust
pub struct AtomSystem {
    pub positions:   Vec<[f32; 3]>,    // or [f32; 2] for 2D
    pub velocities:  Vec<[f32; 3]>,
    pub forces:      Vec<[f32; 3]>,    // accumulated each step
    pub atom_types:  Vec<u8>,           // 0..N_TYPES
    pub charges:     Vec<f32>,          // partial charge, may be derived from type
    pub masses:      Vec<f32>,          // typically per-type, replicated for cache locality
}
```

Per-atom-pair parameters (shared across all atoms of given types):

```rust
pub struct PairTable {
    // Morse parameters per (type_a, type_b) pair
    pub morse_de:   Vec<f32>,    // size N_TYPES × N_TYPES
    pub morse_a:    Vec<f32>,
    pub morse_re:   Vec<f32>,

    // LJ parameters per pair
    pub lj_epsilon: Vec<f32>,
    pub lj_sigma:   Vec<f32>,

    // Coulomb is just q_i × q_j, no pair table needed
}
```

### 3.2 Force calculation pipeline

```
1.  Clear forces[i] = 0 for all i
2.  Build / update neighbor list (cell list)
3.  For each (i, j) neighbor pair within cutoff:
       compute Morse + LJ + Coulomb force
       accumulate into forces[i] and forces[j]  (Newton's 3rd: -F into j)
4.  Add Langevin friction + noise per atom
5.  Velocity-Verlet update of positions and velocities
```

### 3.3 Neighbor list strategy

**Cell list** (uniform grid, cell size = global cutoff). For each atom:
1. Hash position to a cell
2. Iterate over the atom's cell + 26 neighboring cells (or 8 in 2D)
3. Compute pairwise interactions with each candidate

This gives O(N) total work per step (vs O(N²) naive). For N ~ 1000, this is the difference between minutes-per-step and microseconds-per-step.

We don't need full Verlet lists (skin distance + cell list) because our system is reactive — bonds form/break, atoms can move significantly. Rebuild the cell list every step, accept the cost.

### 3.4 Integrator

Already specified: velocity-Verlet, see §2.2.

### 3.5 Bond detection (post-hoc only)

**Critical principle: bonds are emergent, not stored.**

For analysis (snapshot statistics, visualization, replication detection), we identify bonds by:
1. Two atoms are within `1.3 × r_e` of each other for their atom-pair-type
2. Their relative velocity is below a threshold (they're not just passing by)
3. Their pairwise interaction energy is below `−0.5 × D_e` (deeply in the well)

This produces a graph of "currently bonded" pairs which we use for:
- Counting "molecules" (connected components)
- Detecting "replication events" (chain duplication topology)
- Visualization (drawing lines between bonded atoms)

**The dynamics never read this graph.** Forces are always computed from continuous potentials. This is the principled difference from `chem_sim_rs`, where bonds were stored data structures driving rule-based behavior.

---

## 4. Module Layout

Extends the existing `chem_sim_rs` Cargo workspace. New crate `chem_phys` alongside the existing `core`:

```
chem_sim_rs/
├── Cargo.toml                  workspace root
├── core/                       existing rule-based chemistry (kept as v0.1)
└── chem_phys/                  NEW — physics-based MD chemistry (Phase C)
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── system.rs           AtomSystem (SoA storage)
    │   ├── pair_table.rs       per-type-pair parameters
    │   ├── potentials.rs       Morse, LJ, Coulomb (functions, no state)
    │   ├── neighbor.rs         cell list neighbor finding
    │   ├── integrator.rs       velocity-Verlet
    │   ├── thermostat.rs       Langevin
    │   ├── forces.rs           main force computation pipeline
    │   ├── bonds.rs            post-hoc bond detection (analysis only)
    │   └── snapshot.rs         statistics + serialization
    └── tests/
        ├── conservation.rs     energy conservation in NVE
        ├── thermalization.rs   reaches correct T under Langevin
        ├── dimerization.rs     two atom types form dimers
        ├── chains.rs           longer chains form spontaneously
        └── catalysis.rs        the decisive test (template effect emerges)
```

### Dependencies

```toml
[dependencies]
rand     = "0.8"          # already used
rand_pcg = "0.3"          # deterministic PRNG
serde    = { version = "1", features = ["derive"] }
serde_json = "1"
fxhash   = "0.2"          # for cell-list hashmap

[dev-dependencies]
rayon    = "1"            # parallel testing across seeds
approx   = "0.5"          # float comparisons in tests
```

No external MD library — we implement from scratch. This is intentional: existing libraries (LAMMPS bindings, etc.) come with their own designer choices baked in.

### Integration with existing code

- `chem_sim_rs/core` (rule-based) stays — it's the validated v0.1 substrate
- `chem_sim_rs/chem_phys` is independent
- Eventually a third crate `chem_unified` could provide a common interface so DAERWEN's Python layer can call either

---

## 5. Validation Strategy

Four sequential phases. Each must pass before the next begins.

### Phase B — Sanity tests (1-2 weeks)

Goal: prove the MD machinery itself is correct.

| Test | Pass criterion |
|------|---------------|
| Single-atom in box | conservation of energy (drift < 0.1% / 10⁵ steps in NVE) |
| LJ liquid | reaches expected pressure & RDF for given (T, ρ) |
| Pair of attractive atoms | form a vibrating dimer, oscillation frequency ≈ √(2 D_e a² / m) |
| Langevin thermalization | ⟨KE⟩ converges to (3/2) k_B T per atom (3D) within 10% |

If any of these fails, the implementation has a bug. No physics question yet, just engineering.

### Phase C-1 — Dimerization (2-4 weeks)

Goal: with 2 atom types and Morse attraction between them, do dimers form and persist?

Setup: 100 type-A + 100 type-B atoms, Morse_AB strong, Morse_AA & Morse_BB weak (just slight attraction), LJ everywhere, no Coulomb.

Pass: at equilibrium, > 50 of the 100 possible AB pairs are bonded (by post-hoc detection). Pair half-life > 10⁴ steps.

This is the trivial case. If it fails, MD parameters are wrong.

### Phase C-2 — Chain formation (4-8 weeks)

Goal: longer-than-dimer chains form.

Setup: same as above but with a "linker" type C that has two attractive sides. Or: use partial charges so that chains of alternating polarity form via Coulomb in addition to Morse.

Pass: at equilibrium, average molecule size > 3, with non-trivial size distribution.

This tests whether the geometry of bonded atoms creates new bonding sites — a precondition for chain growth.

### Phase D — Template catalysis (8-16 weeks)

**The decisive test.**

Setup: a pre-formed chain of, say, length 6 (sequence ABABAB) is placed in a soup of free A and B monomers. We measure:
- (a) baseline rate of A-B bond formation in the soup
- (b) rate of A-B bond formation **within geometric proximity** of the existing chain

If template catalysis is real, (b) should be measurably higher than (a) — the chain's geometry guides monomers into bonding configurations they would not otherwise reach as easily.

**Pass criterion**: (b) / (a) > 5× over a statistically significant sample of independent runs.

If we see this, we have **physics-grounded catalysis with no rule encoding it**. This is the publishable result.

If we don't see it, we learn that our potential parameters don't yield the geometric specificity needed. Try varying:
- Stiffer Morse (higher `a`) for sharper geometric constraints
- Asymmetric Morse parameters across types
- Larger partial charges for stronger Coulomb-driven alignment
- Smaller temperature (less thermal noise)

This is a **scientific search**, not engineering. No guarantee of success.

---

## 6. Honest Risk Assessment

### What might fail

1. **Phase B**: implementation bugs. Probability of passing eventually: 95%. Just engineering.
2. **Phase C-1**: dimerization. 85% probability of passing — if our Morse parameters are reasonable, dimers form trivially.
3. **Phase C-2**: chains. 50% probability of straightforward success. Chain formation often requires careful potential design (real polymers have very specific bond angles and dihedral constraints; we have only pairwise).
4. **Phase D**: template catalysis. **15-25% probability of clean success.** This is where most prior efforts have plateaued. Hutton's 2007 work needed cell membranes to make replication tractable; pure free-monomer template catalysis from MD has been reported in special cases but is fragile.

### Computational feasibility

Single workstation, 20 cores. Estimated throughput at 1000 atoms:
- Force calculation per step: ~1 ms (rayon-parallelized cell list)
- Integration: <0.1 ms
- Total: ~1.5 ms / step
- Steps per minute: ~40,000
- Steps per day: ~50 million

This is enough for the long-time-scale exploration phase D requires. **Compute is not the bottleneck** — algorithm design is.

### Cost of failure

Even if Phase D fails, we will have:
- A working MD engine in Rust (engineering output, reusable)
- Clear negative results in a specific parameter regime (scientific output)
- Detailed documentation of what didn't work and why (for the next attempt)

These are publishable in their own right, in venues like *Artificial Life* or *Journal of Theoretical Biology*. Not Nature, but real contributions.

---

## 7. Open Questions for User Decision

These are questions only the project lead can answer, and they shape everything downstream.

1. **Atom alphabet**: Start with 2 types (cleanest, easiest to reason about) or 4 types (matches current chem_sim_rs A/U/G/C and DAERWEN's gene mapping)?
   - **Recommendation**: 2 types for Phase B/C-1; expand to 4 if catalysis hunt requires it.

2. **Dimensionality**: 2D (faster, easier visualization, matches DAERWEN's main engine) or 3D (more realistic geometry, allows real folding)?
   - **Recommendation**: 2D throughout. DAERWEN is 2D; real catalysis can emerge in 2D (cellular automata of various sorts have shown this).

3. **Thermostat strength**: How much should we damp dynamics?
   - **Decision needed before C-2**.

4. **Periodic boundaries vs walls**: torus (no boundary effects, easier to reason about) or hard walls (more realistic but introduces edge artifacts)?
   - **Recommendation**: torus, like DAERWEN.

5. **Acceptance level for type-pair parameter table**: do we accept that each pair (A, B) has its own (D_e, a, r_e, ε, σ) — itself a designer choice — or do we try to derive these from atomic properties (charge, size)?
   - **Honest answer**: Some designer choice is unavoidable here. A "fully derived" parameter table would require quantum chemistry, which is out of scope. The minimum-prior version is: assign each atom type a (charge, size) pair; derive all interaction parameters as combinations of these. This still has designer choice in the (charge, size) tuple per type, but it's much less than a full pair table.
   - **Decision needed**: how much to push on this.

6. **Validation timeline pressure**: do you want me to optimize for "first observable phenomenon as quickly as possible" (i.e., aggressive Phase B in 2 weeks then jump to C) or "rock-solid foundation first" (Phase B with extensive tests, 4-6 weeks)?

7. **Visualization**: do you want a Rust-side visualization (using some Rust GUI like egui or bevy) or rely on output JSON + Python plotting?
   - **Recommendation**: JSON output + Python matplotlib. Visualization is for humans to interpret, not for the simulation.

---

## 8. Reading Material List

**Verified directly via WebFetch during this design phase:**

- *Verlet integration* (Wikipedia) — confirmed velocity-Verlet update equations and stability properties.
- *Morse potential* (Wikipedia) — confirmed V(r), parameter meanings, why Morse vs harmonic.
- *Lennard-Jones potential* (Wikipedia) — confirmed 12-6 form, ε/σ semantics, 2.5σ cutoff convention.

**Required reading for implementation (not yet personally verified, listed for the implementer):**

- **Allen & Tildesley, *Computer Simulation of Liquids*** (Oxford University Press, 1987 1st ed; 2017 2nd ed). The MD bible. Read chapters 1-4 (basic MD), chapter 6 (intermolecular potentials), chapter 7 (sampling).

- **Frenkel & Smit, *Understanding Molecular Simulation* (2nd ed, 2002)**. Companion to Allen & Tildesley with stronger statistical mechanics treatment. Read chapters 4 (MD), 6 (MC), 14 (rare events / reactive sampling).

- **van Duin, Dasgupta, Lorant, Goddard III (2001)**, "ReaxFF: A Reactive Force Field for Hydrocarbons", *J Phys Chem A* 105:9396. The original reactive force field paper. Read for: how bond order interpolation works, charge equilibration in reactive systems. Note: ReaxFF itself is too heavyweight for our scale, but the *concepts* (smooth bond formation, geometric specificity from charges) inform our design.

- **Marrink et al. (2007)**, "The MARTINI Force Field: Coarse Grained Model for Biomolecular Simulations", *J Phys Chem B* 111:7812. The standard for biological coarse-grained MD. Read for: how to choose the level of coarse-graining, what bead types capture chemical character.

- **Hutton, T. J. (2007)**, "Evolvable Self-Reproducing Cells in a Two-Dimensional Artificial Chemistry", *Artificial Life* 13(1):11-30. The most directly relevant prior work. Hutton showed cells can form, divide, and evolve in a 2D artificial chemistry — but he used reaction rules, not MD. Reading him will tell us where rule-based hits its ceiling.

- **Schuster, P., Eigen, M.** (1977-1979) — the *hypercycle* papers in *Naturwissenschaften*. Background on what self-replication looks like at the chemical-network level. Inform our expectations for what catalysis we might see.

- **Pross, A., *What is Life?: How Chemistry Becomes Biology*** (Oxford, 2012). Conceptual framework for thinking about the chemistry → biology transition. Read for inspiration, not for techniques.

- **rust-md ecosystem**: `lumol` is a research-grade MD engine in Rust (https://github.com/lumol-org/lumol). Read its codebase for: how Rust code organizes force calculation, neighbor lists, integrators. Don't depend on it (we want our own primitives), but learn from it.

---

## 9. Glossary

- **MD**: Molecular Dynamics. Simulating the time evolution of atoms via Newton's laws and pairwise forces.
- **NVE**: ensemble with conserved Number of particles, Volume, Energy. The natural ensemble of pure Newtonian dynamics.
- **NVT**: ensemble with conserved N, V, Temperature. Requires a thermostat.
- **NPT**: ensemble with conserved N, Pressure, Temperature. Requires a barostat. Not relevant for us.
- **Symplectic integrator**: an integrator that preserves the symplectic structure of phase space, giving bounded energy error rather than secular drift. Velocity-Verlet is the standard simple choice.
- **Thermostat**: algorithm that adjusts velocities to maintain temperature. Berendsen, Nose-Hoover, Langevin are the common choices.
- **Cutoff**: the distance beyond which we set pairwise forces to zero. Controls accuracy vs compute trade-off.
- **Reduced units**: simulation units chosen so that ε = σ = m = 1 for the dominant species. Removes spurious precision issues. Real units recovered by post-hoc scaling.
- **Cell list / linked list**: spatial data structure for finding neighboring atoms in O(N) time per step.
- **Verlet list**: enhancement to cell lists that caches the neighbor list across multiple steps using a "skin" distance. Faster for non-reactive systems; not used here because our atoms move significantly between steps.
- **Bond order**: a continuous measure (typically 0 to 1+) of how "bonded" two atoms are. Used in ReaxFF to smoothly transition between non-bonded and bonded states.
- **Designer's Trap**: the project's term for arbitrary priors injected into a simulation by the designer. The project's central commitment is to minimize these.
- **Template catalysis**: the phenomenon where an existing molecular structure (template) increases the rate of formation of a similar structure nearby. The decisive test for Phase D.
- **Abiogenesis**: the spontaneous emergence of self-replicating systems from non-living chemistry. The implicit goal of the entire physics-grounded chemistry direction.

---

## Closing Note

This design is **a starting point**, not a final blueprint. Implementation will surface problems we cannot anticipate; the design will be revised as we learn. The git history of this file is the record of those revisions.

The most important decision is not technical — it is whether to commit the months of focused work this requires. The project lead (Hou Zehao) has indicated willingness; this document attempts to make the path concrete enough to act on.

*Document prepared 2026-04-16. Source-verified (Wikipedia, this date) for MD basics, Morse, Lennard-Jones; relies on prior knowledge for Hutton 2007, ReaxFF, MARTINI, and the broader artificial-life literature.*
