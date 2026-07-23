# Research Status & Open Bottlenecks

> Last updated: **2026-04-17**
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
| **Beats physics baseline** | ✅ Validated | **+16.3% improvement** over mutation-disabled physics baseline on gradient-alignment tasks (2026-04-17) |
| **Low forgetting** | ✅ Validated | avg_forgetting 0.010 on 10-task CL benchmark; ~100× lower than vanilla MLP on Split MNIST (0.992) — note: different paradigms, not a direct comparison |
| **Noise / catastrophe robustness** | ✅ Validated | 0.83-0.98 noise resilience, 1.0 catastrophe recovery |
| **chem_sim_rs (Rust)** — template-directed replication | ✅ Validated | abiogenesis from pure free monomers: 14k complete replications |
| **Gene expression with reduced designer-prior** | ✅ Implemented | 12 hand-crafted formulas replaced with one uniform composition-based mapping (less Designer's Trap). Legacy archived in `engine/_legacy_gene_expression.py`. |
| **Rust pyo3 integration** | ✅ Validated | DAERWEN Python now calls Rust chem_sim via maturin-built wheel in the chu conda env |
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

- **2026-07-19 (root-cause fix)**: Added opt-in `rich_channels` — each particle gets
  an independently-evolvable gene per otherwise-inert channel (attractive+nutritious
  if >0, repulsive+toxic if <0), turning ~8 dead channels into causally active ones
  that steer movement and affect survival. Payoff (Finding 12): 6 environments that
  the default engine cannot remember at all (decode 0.167 = chance, RSA 1.00) become
  a decodable, decaying genetic memory (decode 0.625, RSA 0.31). This directly
  attacks the root bottleneck behind low memory capacity and no associative learning
  (Findings 4/7/9b). Test: `tests/honest/memory_capacity_rich.py`. Default off.
  But it does NOT unlock associative recall (Finding 12b): re-running the sign-flip
  test under rich_channels (incl. a pure `cue_channels` cue that is sensable but not
  food) still shows no association. Refined diagnosis — recall is blocked by the
  learning rule (positional/survival selection never makes a predictive cue
  necessary), not by sensing. So: memory *capacity* is a substrate problem (fixed);
  associative *recall* is a learning-rule problem (open, would need within-lifetime
  plasticity). Test: `tests/honest/memory_association_rich.py`.
- **2026-07-19 (recall achieved, with a caveat)**: Added opt-in `lifetime_plasticity`
  — a reward-modulated Hebbian rule lets each particle learn a cue-weight within its
  life. Re-running the sign-flip test, the *learned* weight consistently tracks the
  cue→reward contingency (median L−R +0.081, 5/6 seeds; genetics alone was −0.079,
  2/6). So associative recall — the half population genetics could not do — is now
  demonstrated (Finding 13). Caveat: modest magnitude, and it is a **hand-designed
  learning rule (an added prior)**, so recall is shown *achievable/engineered*, not
  *emergent* from the substrate. Default off.
- **2026-07-22 (learning itself emerges)**: Made the learning rate a heritable,
  mutating gene (`evolvable_plasticity`) instead of a hand-set constant. In a world
  that changes faster than generations (oscillating reward), the learning-rate gene
  evolves UP from random init (≈2.3) and the population learns to follow the cue; in
  a STATIC world it decays to ≈0.46 (genetic assimilation); in a noise world it
  plateaus mid-range. So the *capacity/decision to learn* emerges from selection —
  the classic Baldwin effect, reproduced from the substrate (Finding 14). Remaining
  prior: the reward-modulated Hebbian *form* is still hand-designed. Test:
  `tests/honest/evolution_of_learning.py`. Default off.
- **2026-07-22 (third step — honest negative)**: Tried to make the learning
  *mechanism* itself emerge: `chemical_learning` gives each particle an internal
  molecule evolving by a generic bilinear reaction with 5 evolvable coefficients;
  if evolution wired up a coincidence-detector, learning would emerge from
  chemistry. It did NOT beat a no-learning control at tracking a moving reward
  (Finding 15). Reason (sharpens the whole arc): positional selection + local
  reproduction let the population track resources *without* individual learning, so
  the fitness value of learning is too low for its mechanism to self-assemble. The
  emergence loop's remaining open problem is now precisely stated: a task where an
  individual must use a within-lifetime learned association that local reproduction
  cannot shortcut. Test: `tests/honest/emergent_learning_chem.py`. Default off.
- **2026-07-19**: **First positive, properly-controlled memory result** (borrowed
  neuroscience method — see [`tests/honest/FINDINGS.md`](tests/honest/FINDINGS.md)
  Finding 9). A leave-one-seed-out cross-validated linear decoder reads *which of 3
  environments the population was recently in* from the **heritable gene-frequency
  vector** at 0.75 (chance 0.33), decaying to chance by ~800 steps — a real
  forgetting curve; shuffle control at chance. So "population genetics is memory"
  holds empirically in the **storage/retention** sense (distinct from associative
  recall, which Finding 6 shows does not emerge). New test:
  `tests/honest/population_decoding.py`. Follow-ups (Findings 9b–11): the memory is
  **low-capacity** — it stores a few coarse regimes but collapses to ~chance for a
  fine 6-way distinction (RSA: several environments genetically identical); it shows
  genuine **"savings"** (a latent trace speeds relearning of a forgotten environment,
  a recall-adjacent positive); but shows **no "emotional weighting"** (memory depth
  is flat across selection strength). Tests: `memory_forgetting_deep.py`,
  `memory_savings.py`, `memory_emotional_weighting.py`.
- **2026-07-17**: Added three **opt-in** engine capabilities from the findings
  (defaults unchanged, so existing results/behaviour are preserved): `seed` for
  reproducibility (Finding 1); `orthogonal_expression` to fix the 3-DOF phenotype
  collapse when independent traits are needed (Finding 7/8); `carrying_capacity`
  to bound runaway growth (Finding 5). Also added `tests/honest/` — a reproducible,
  multi-seed, distribution-based test suite. The hippocampus/memory research goal
  and positioning are unchanged; these are tools for pursuing it more rigorously.
- **2026-07-15**: Constructive follow-up (see [`tests/honest/FINDINGS.md`](tests/honest/FINDINGS.md)
  Finding 6). Added opt-in `evolvable_sensing` — a pathway that lets movement
  respond to an arbitrary signal channel via an evolvable `signal_affinity` gene
  (default off = byte-identical to before). Tested whether evolution can learn a
  signal→reward association, with a sign-flip control (reward co-located vs
  opposite to the signal). **No sign flip across four regimes** (tail-gene,
  independent modifier locus, high leverage, survival-critical reward): the gene
  is co-opted by orthogonal pressures (composition hitchhiking, then dispersal),
  never by the reward contingency. Removing the sensory prior is necessary but not
  sufficient; the barrier is layered (non-orthogonal gene encoding + weak
  chemotaxis vs noise + position/fitness decoupling).
- **2026-07-14**: Independent reproducible re-verification (see
  [`tests/honest/FINDINGS.md`](tests/honest/FINDINGS.md)). Added
  `Ecology2DConfig.seed` — the engine previously used an unseeded RNG, so no
  prior number was reproducible. Re-tested headline claims with seeded, 8-seed,
  paired stats: evolution beats physics by a **median +0.054 absolute (~+9%),
  not +16.3%** (that was one draw from a wide distribution). The advantage
  reduces to enriching **one gene** (`field_interaction`); the `mutation_rate=0`
  "physics baseline" already contains natural selection, so it is not "no
  evolution". Proved deterministically that **movement direction depends on the
  ATP channel alone** (channels 3–10 causally inert) — the proposed
  "signal–reward decoupling" capability is structurally impossible without a new
  sensory prior. Also: the ecology has **no carrying capacity** (population
  diverges under sustained energy input).
- **2026-04-17**: Avalanche CL comparison (Split MNIST, vanilla MLP + Naive strategy): neural net baseline shows 99.2% forgetting rate (catastrophic); DAERWEN's native benchmark shows 1.0% forgetting rate (~100× lower). Paradigm mismatch noted — vanilla NN does image classification, DAERWEN does population ecology; direct comparison not meaningful, but forgetting characteristic is dramatically different.
- **2026-04-17**: First stable positive improvement over physics baseline: **+16.3%** (previous OLD-gene runs were between -35% and +6%). The switch to uniform composition-based gene expression (less designer-trap) ALSO happens to make the system learn above physics — not worse, slightly better.
- **2026-04-17**: Removed Level 2 chem_sim energy substrate code (use_chem_sim_energy flag + 7 config params). Confirmed via fair A/B test that chem_sim alone cannot sustain the system (ATP atoms polymerize into uneatable chains). Kept chem_sim_rs as the gene-expression layer (Level 1, works).
- **2026-04-17**: Cleaned up the gene-expression dual-track. 12-formula legacy moved to `engine/_legacy_gene_expression.py` (archived, not imported). Uniform composition-based mapping is now the only path. `use_chem_sim_genes` flag removed from Ecology2DConfig.
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
