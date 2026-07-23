# Memory and Learning in DAERWEN: an emergence-vs-design map

> A synthesis of the honest, reproducible investigation in `tests/honest/`
> (Findings 4–15). It does **not** claim a closed loop. It maps precisely which
> memory/learning properties *emerge* from the population-genetic substrate, which
> had to be *engineered in*, and — for the one that resists emergence — *why*.

Status: research note, 2026-07-22. All results are seeded, multi-seed, with
shuffle/no-learning controls; every capability discussed is an **opt-in** engine
flag whose default is off (the default engine is byte-for-byte unchanged).

---

## 1. The question

DAERWEN's core hypothesis is *"the distribution of genes across a population is a
form of memory"*, framed as an artificial hippocampus, on a philosophy of
**minimal designer priors, maximal emergence**. Two questions follow:

1. Which hippocampal memory properties does this substrate actually have?
2. Can the mechanisms of memory/learning *emerge* from the substrate, or must they
   be designed in — and if the latter, how far down can the prior be pushed?

We answer both by borrowing the quantitative methods neuroscience uses to measure
memory in real neural populations (population decoding, forgetting curves, savings,
representational similarity), and by evolutionary experiments (evolution of
learning, Baldwin effect).

## 2. The scorecard

| Property | Emerges? | Evidence |
|---|---|---|
| **Storage / retention** (a decodable trace of the recent environment) | ✅ emergent | Finding 9: a linear decoder reads the recent environment from the *heritable* gene-frequency vector at 0.75 (chance 0.33), shuffle control at chance |
| **Forgetting curve** (trace decays to chance) | ✅ emergent | Finding 9: 0.75 → 0.33 over ~800 steps |
| **Savings** (a decayed memory speeds relearning) | ✅ emergent | Finding 10: experienced pop re-reaches the remembered state faster than naive (median savings +0.08) |
| **Capacity** (how many environments) | ⚠️ low; expandable by design | Finding 9b: 6-way collapses to chance (environments genetically collinear); Finding 12: `rich_channels` lifts it (6 unremembered environments become decodable, 0.63) |
| **Emotional weighting** (stronger events remembered longer) | ❌ | Finding 11: memory depth flat across selection strength |
| **Associative recall** (cue → stored memory) | ❌ from genetics; ✅ engineered | Findings 6/12b: no sign-flip under pure selection; Finding 13: a hand-designed plasticity rule produces it (modest) |
| **The decision to learn** (whether/how much to learn) | ✅ emergent | Finding 14: an evolvable learning-rate gene climbs in a fast-changing world, decays under genetic assimilation in a static one (Baldwin, 1896) |
| **The learning mechanism** (the update rule itself) | ❌ | Finding 15: a generic evolvable reaction did not self-assemble into a coincidence detector that beats a no-learning control |

## 3. The narrative

**Storage memory emerges cheaply.** Population selection alone leaves a heritable,
decodable, time-decaying trace of recent experience, and even shows savings on
relearning. In the *storage/retention* sense, "population genetics is memory" is
empirically true, with proper controls. This is the strongest positive result.

**But it is low-capacity and lossy.** The substrate has few causally-independent
axes (movement follows only the ATP gradient; fitness reads only ATP/nutrient/
inhibitor), so many environments leave near-identical genetic signatures and cannot
be told apart (Finding 9b). Giving the inert channels evolvable sense+metabolism
genes (`rich_channels`) expands capacity — previously un-memorable environments
become decodable memories (Finding 12) — but this is an engineered substrate
change, not emergence.

**Associative recall does not emerge from selection.** No matter how the sensory
substrate is enriched (Findings 6, 12b — four+ variants, pure cues, survival-
critical rewards), evolution never learns to *use a cue that predicts reward*. The
tell is always the same: the population reaches the reward by **positional selection
+ local reproduction**, which works without following any cue, so the cue never
becomes necessary and its gene just drifts.

**Recall can be engineered.** A hand-written reward-modulated Hebbian rule
(`lifetime_plasticity`) gives individuals within-lifetime associative learning; the
learned weight tracks the cue-reward contingency (Finding 13). Modest, and by
adding a designed prior.

**The *decision* to learn emerges.** Making the learning rate a heritable gene
(`evolvable_plasticity`) and placing the population in a world that changes faster
than generations, selection itself amplifies learning (gene climbs to ≈2.3) and, in
a static world, assimilates the solution genetically and lets learning decay
(≈0.46). This is the Baldwin effect, reproduced from the substrate (Finding 14) —
the learning rate was never set by hand.

**The *mechanism* of learning does not emerge.** Replacing the hand-written rule
with a generic bilinear internal reaction whose coefficients evolve (`chemical_
learning`), evolution did not wire up a coincidence detector that beats a
no-learning control at tracking a moving reward (Finding 15).

## 4. The central obstacle

Every negative above has one root cause:

> **Reproduction-driven positional selection is a competent but shallow learner.**
> In a spatial ecology where organisms reproduce locally, the population tracks and
> solves resource problems by differential survival + local reproduction — it
> "flows" with resources — *without any individual needing to learn*. This makes
> individual associative learning unnecessary, so it neither emerges nor (when
> engineered in) contributes much.

This is a tension inherent to the project's own premise. Population genetics *is* a
learner; but *because* it is a good population-level learner, it removes the
selective pressure for individual-level learning to arise. The substrate that gives
you emergent population memory is the same substrate that suppresses emergent
individual recall.

This is a genuine, honest scientific statement about the **limits** of "population
genetics as intelligence": excellent for spatial/survival adaptation, structurally
hostile to the emergence of individual cognition.

## 5. The precise open problem

To make individual learning *emerge*, one must design a task/environment where:

1. an individual must **use a within-lifetime learned association** to survive or
   reproduce, and
2. **population reproduction cannot shortcut it** — the answer cannot be reached by
   "whoever is in the right place survives and reproduces there".

Candidate directions (each hard, because spatial reproduction shortcuts most things):
- a cue whose **meaning is only knowable by individual experience** and changes
  within a lifetime (not readable from spatial position);
- **temporal/sequential** structure an individual must remember (a population cannot
  "flow" through a sequence);
- **decoupling reproduction from the task** so that being in the right place does
  not automatically convert into local offspring.

## 6. Honest caveats and standing limits

- **The prior floor.** Every capability here is opt-in machinery *we added*.
  Emergence can be pushed down a level (rule → rate → …) but not to zero; there is
  an unavoidable floor of priors (a template reaction, a set of channels). This is
  consistent with, not a violation of, the project's stated philosophy — but it
  means "closed emergence loop" is not an achievable endpoint, only a direction.
- **Analogy limits.** "Population genetics = hippocampus" holds in *some* measurable
  respects (storage, forgetting, savings) and fails in others (associative recall,
  emotional weighting, high capacity). It is a partial, quantified analogy, not an
  identity.
- **Integration.** DAERWEN is an isolated substrate emitting memory-like statistics;
  nothing here connects it to a downstream decision-maker that *uses* the memory.
  Until it does, "memory module for AGI" remains architectural intent.
- **Compute.** The rich-channel / plasticity / chemistry loops are per-particle
  Python and heavy; several runs were killed and had to be reduced. Scaling any of
  this requires vectorization.
- **Magnitudes and seeds.** Positive effects are real (controlled) but often modest,
  at 4–8 seeds. They establish existence and direction, not precise magnitudes.

## 7. What stands

- A reproducible honest test suite (`tests/honest/`) and a raw findings log
  (`FINDINGS.md`).
- Verified, controlled positive results: emergent storage memory, a forgetting
  curve, savings, and the emergent *decision* to learn (Baldwin).
- Verified negatives with mechanistic explanations: no emergent associative recall,
  no emotional weighting, low capacity, and no emergent learning *mechanism* — all
  traced to the reproduction shortcut.
- Engineering improvements, all opt-in: reproducible seeding, a carrying capacity,
  an orthogonal phenotype encoding, rich channels, and three learning modes.

The honest headline is not "we built a hippocampus". It is: **population genetics
gives you emergent, lossy, low-capacity *storage* memory and can even evolve the
*decision* to learn, but it structurally suppresses the emergence of individual
*associative recall* — because reproduction keeps making learning unnecessary.**
That is a real result, and it names exactly the problem the next step must solve.
