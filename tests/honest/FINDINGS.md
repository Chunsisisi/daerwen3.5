# Honest verification findings

> Produced by an independent, reproducible re-test of DAERWEN 3.5's headline
> claims. Every number here is regenerated from seeded runs; the scripts are in
> this folder. The goal is neither to boost nor to dismiss the project — it is to
> replace single-run, non-reproducible numbers with distributions, and to state
> exactly what the system does and does not do.

Date: 2026-07-14. Environment: conda `chu`, numpy 2.4.4, CPU, Rust `chem_sim` present.

---

## TL;DR

| Claim (README / RESEARCH_STATUS) | Verified reality |
|---|---|
| "+16.3% over physics baseline" | **Real but overstated.** Paired, 8-seed: evolution beats physics by a **median +0.054 absolute alignment (~+9% relative), IQR [+0.014, +0.124]**. One seed still goes negative. The single "+16.3%" was one lucky draw from a wide distribution. |
| "Reproducible results" | **Was false, now fixed.** The engine used an unseeded `default_rng()`; no reported number could be reproduced. Added `Ecology2DConfig.seed`; same seed now gives byte-identical trajectories. |
| Physics baseline = "no evolution" | **Misleading.** The `mutation_rate=0` arm still runs full Darwinian *selection on standing variation* — survivors reach ~100% positive `field_interaction` with zero mutation. It is "no new mutations", not "no evolution". |
| The advantage = learning | **Trivial mechanism.** The boundary-task advantage is explained by enriching **one scalar gene** (`field_interaction`) that aligns chemotaxis with the reward gradient. corr(field_interaction, score) = 0.52; both means shift together (+0.127 / +0.121). Not spatial memory, not multi-modal integration. |
| Future strength: "signal–reward decoupling" (associate an arbitrary signal with reward) | **Structurally impossible in the current engine.** Movement direction is a function of the ATP gradient *alone*; metabolism reads only ATP/nutrient/inhibitor. Channels 3–10 are causally inert. See Finding 4. |
| — (undocumented) | **No carrying capacity.** Fed constant energy, the population grows without bound (140 → 18,039 in 50 steps). |

---

## Note on the engine options added by this work (defaults unchanged)

This investigation added three **opt-in** capabilities; engine defaults are
unchanged, so all existing results and the project's numbers are preserved:
- `seed` — reproducibility (Finding 1). Default `None` = old random behavior.
- `orthogonal_expression=True` — fixes the 3-DOF phenotype collapse (Finding 7)
  when independent traits are wanted. Default `False` = original composition map.
- `carrying_capacity=<int>` — bounds runaway growth (Finding 5). Default `None`
  = unbounded (original behavior).

The findings below were measured with the relevant option toggled on where noted;
they describe what the engine *can* do, not a change to its defaults.

## Method (what makes this "honest")

1. **Seeded.** `Ecology2DConfig(seed=...)` threads into the one RNG that drives all
   ecology stochasticity. Verified: same seed → identical `(alive, energy, gen,
   replications)`; different seed → diverges.
2. **Distributions, not points.** Every metric runs over 8 seeds; we report median
   + IQR + range, never a single run.
3. **Paired.** Evolution and physics arms use the *same* seed, so their difference
   is a per-world delta. We report the **difference**, never the ratio — the old
   benchmark's `(sys-base)/max(base,1e-6)` produced a fake "+53,000,000%" when the
   baseline was ~0.
4. **Extinction is a failure, not an exclusion.** Runs that die are counted, not
   silently dropped from an average.
5. **Anti-overfit guard.** Discriminating tests ship with a positive control; a
   null result is only accepted if the control clearly fires. (This guard already
   caught one of *our own* tests being too weak to conclude — see git history of
   `movement_probe.py`.)

---

## Finding 1 — Reproducibility was broken (now fixed)

`core.py` created `self.rng = np.random.default_rng()` with no seed, so the
"reproducible results" claim could not hold. Fix: `Ecology2DConfig.seed` (default
`None` = old behavior). Proof: two `seed=42` runs → identical `(95, 100.249865,
gen 46, 479 repl)`; `seed=7` → `(84, 106.467204, gen 14, 297)`.

## Finding 2 — Evolution beats physics, but modestly

`honest_harness.py`, boundary task, 8 seeds:

```
evolution score : median +0.606  IQR[+0.593, +0.637]
physics score   : median +0.584  IQR[+0.488, +0.643]
paired diff     : median +0.054  IQR[+0.014, +0.124]  range[-0.047, +0.415]
extinction      : evo 0%  physics 0%
```

The IQR of the paired difference is strictly positive, so the effect is real and
reproducible — but small. Physics already reaches 0.584; evolution adds ~9% on top.
Physics does ~91% of the work.

## Finding 3 — The advantage is one-parameter tuning, not rich learning

`trait_attribution` (6 seeds): evolution raises mean `field_interaction`
0.413 → 0.540 (+0.127); boundary score rises 0.514 → 0.635 (+0.121) in lockstep,
corr = 0.52. Crucially, `fraction(field_interaction > 0) ≈ 1.00 in BOTH arms` —
selection on standing variation (present even at mutation=0) does the bulk of the
work; new mutation only nudges the magnitude. The task rewards "climb the ATP
gradient"; the single gene that controls that response is what evolution tunes.
This is the minimal possible adaptation, not memory or integration.

## Finding 4 — The substrate is blind to any non-ATP signal (Designer's Trap)

`movement_probe.py` — deterministic, one engineered maximal gradient-climber,
Brownian noise off, gradient on one channel at a time:

```
gradient on ATP (0, WIRED)  -> dx = -0.0227   (follows it)
gradient on nutrient (1)    -> dx =  0.0000
gradient on inhibitor (2)   -> dx =  0.0000
gradient on inert ch5/7/10  -> dx =  0.0000
```

Movement direction depends on the ATP channel **alone**. Fitness (metabolism)
reads only ATP/nutrient/inhibitor. Therefore channels 3–10 cannot influence
behavior *or* survival — no arbitrary signal can ever be associated with reward.
The "signal–reward decoupling" benchmark named as the system's future strength is
structurally impossible here without adding a new sensory pathway — i.e. a new
designer prior, the exact Designer's Trap the project sets out to minimize.

## Finding 5 — No carrying capacity

Under a constantly re-imposed ATP band (constant energy inflow), population went
140 → 18,039 in 50 steps and kept growing. The ecology had no density-dependent
brake; "equilibrium population" only existed because the default solar input is
tiny and ATP decays. Any sustained external energy made it diverge.

**Fix available (opt-in):** `Ecology2DConfig.carrying_capacity=<int>` adds an
unbiased random cull that bounds the population without perturbing gene
frequencies. With it set, the same constant-ATP-band scenario holds flat at the
cap instead of diverging. Default stays `None` (unbounded) to preserve old behavior.

## Finding 6 — Constructive test: we removed the sensory prior; association still did NOT emerge

Finding 4 says the substrate cannot *sense* a non-ATP signal. So we removed that
prior: added an opt-in `evolvable_sensing` pathway (`Ecology2DConfig.evolvable_sensing`)
— movement now also responds to a designated signal channel via an evolvable
`signal_affinity`, verified to steer that channel while leaving default behavior
byte-identical. Then we asked whether **evolution can learn** to follow a signal
that predicts reward, using a pre-registered anti-overfit control: reward is
placed either co-located with the signal or on the opposite side, and genuine
learning must show a **sign flip** of the evolved `signal_affinity`.

Result across four increasingly-favorable variants — **no sign flip in any of them**:

| Variant | signal_affinity: reward-correlated → anti-correlated | Verdict |
|---|---|---|
| tail-codon gene | +0.60 → +0.50 (both +) | no flip |
| independent modifier locus | −0.49 → −0.70 (both −) | no flip |
| + high leverage (low Brownian) | −0.77 → −0.57 (both −) | no flip |
| + survival-critical reward (starvation ATP) | −0.76 → −0.64 (both −) | no flip |

The control caught **two distinct confounds** that a naive "the gene moved!"
reading would have mis-sold as learning:
1. **Composition hitchhiking** (tail-codon variant): the "independent" gene was
   derived from genome bases, which also feed global base frequency, which drives
   fitness-relevant phenotypes (atp_absorption/aging/chemotaxis are all `f(C-freq)`).
   Selecting C-rich genomes dragged `signal_affinity` positive as a byproduct
   (C-freq 0.23→0.40 rose in lockstep with the gene). **The gene→phenotype map is
   non-orthogonal: no trait can be encoded independently.**
2. **Dispersal co-option** (later variants): with a truly independent gene, the
   signal peak is a steep-gradient crowded spot; `signal_affinity` is driven
   *negative* regardless of reward because moving away from the crowd is generically
   good foraging. The gene became a dispersal knob, not a predictor.

Conclusion: removing the sensory prior is necessary but **not sufficient**. The
substrate does not convert "signal predicts reward" into "follow signal" under any
tested regime — the evolvable gene is co-opted by orthogonal pressures. This does
not prove associative learning is impossible in principle, but shows the current
architecture (degenerate gene encoding + weak chemotaxis vs Brownian + ecological
dynamics that decouple position from reproductive success) does not support it.

### Final check: all three fixes together, multi-seed — still no association

We combined every fix that could plausibly help — `evolvable_sensing` + the
independent `signal_gene` locus + `orthogonal_expression` (healthy 1200-cap
populations) + survival-critical reward — and ran the sign-flip test over 4 seeds.
For genuine association, evolved `signal_affinity` should be systematically higher
when reward tracks the signal (left) than when it is anti-correlated (right):

```
seed0 left-right = +0.102     seed2 left-right = -0.077
seed1 left-right = -0.532     seed3 left-right = -0.454     median = -0.266
```

The sign is essentially random across seeds (founder drift), median negative — **no
association**. Meanwhile `reward_occupancy ≈ 0.5` in every condition and seed: the
population reliably *reaches* the reward, just never *via the signal*.

**Why — the mechanistic root cause.** In this substrate reward is obtained by
*being at a location*, and the population arrives there by selection-on-position:
whoever is near the reward survives and reproduces locally, so occupancy builds up
without any predictive cue. Following an arbitrary signal is therefore never
*necessary* to get the reward — the signal is redundant — so there is no selective
pressure to learn to follow it. Associative learning requires a cue to be
predictively necessary; a spatial ecology with local reproduction never makes it
so. This is the deep reason "population genetics = memory" does not yield
*associative* memory: population genetics selects for where/what you are, not for
learned predictive relationships between arbitrary cues and outcomes.

Reproduce: `tests/honest/association_experiment.py` (and the confound checks in
this file's git history).

## Finding 7 — The "12 phenotypes" have only 3 degrees of freedom

Expressing phenotypes for 4000 random genomes through the real (Rust) mapping and
computing the phenotype correlation matrix:

```
effective dimensionality (participation ratio): 3.00
components for 95% variance: 3 of 12
max |off-diagonal correlation|: 1.000
```

The 12 phenotypes collapse into 4 perfectly-correlated (r = +1.000) triples:

| driving base | phenotypes that are the SAME number (rescaled) |
|---|---|
| G | field_interaction = conversion_threshold = cooperation_threshold |
| A | replication_threshold = interaction_mode = interaction_threshold |
| U | movement_response = inhibitor_sensitivity = replication_energy_split |
| C | aging_resistance = atp_absorption_rate = chemotaxis_gene_strength |

Because each phenotype is `lerp(min, max, 2·freq_of_one_base)` and the four base
frequencies sum to 1, there are only **3 independent phenotypic dimensions**. You
cannot, for example, raise `aging_resistance` without identically raising
`atp_absorption_rate` and `chemotaxis_gene_strength`. Evolution has 3 knobs, not 12.

This is the root cause of the Finding-6 hitchhiking confound, and it is a **more
severe** Designer's Trap than the one it replaced: `RESEARCH_STATUS.md` lists
"12 hand-crafted formulas → one uniform composition mapping" as a *reduction* of
the Designer's Trap, but by tying every phenotype to a single global base
frequency it collapsed a potentially-12-dimensional phenotype space to 3. Fix
direction: encode each phenotype from a **disjoint genome segment** (positional
encoding) instead of global composition, so traits can vary independently.

Reproduce: express `express_phenotypes_from_composition` over random genomes and
inspect the correlation matrix (script in this file's git history).

### Fix implemented and verified: `orthogonal_expression`

Added `Ecology2DConfig.orthogonal_expression` (opt-in; default `False` keeps the
original composition map). When on, each phenotype is read from a **disjoint genome
segment** (positional encoding) instead of global base frequency. Verified on 4000
random genomes:

```
composition (old): effective dimensionality 3.00,  max |cross-corr| 1.000
orthogonal (new) : effective dimensionality 11.96, max |cross-corr| 0.059
```

The 12 phenotypes become nearly independent. **Bonus (not the goal): it also makes
the ecology far more viable** — median living population over 5 seeds after 1500
steps rose from **47 (old) to 902 (new)**, with neither encoding going extinct.
The 3-DOF map was trapping the population in a fragile low equilibrium; 12
independent knobs let evolution reach rich, populous regimes. This is a concrete
step toward the project's own goal (more phenotypic freedom, less designer
ceiling) — the opposite of what the "uniform composition" change achieved.

## Finding 8 — The "beats physics" edge is an artifact of the degenerate encoding

Re-running Finding 2 (evolution vs physics, boundary task, paired) under the fixed
`orthogonal_expression=True`:

| encoding | evo−physics paired diff | populations |
|---|---|---|
| composition (3 DOF, old) | **median +0.054**, IQR [+0.014, +0.124] | small (median 47 alive) |
| orthogonal (12 DOF, new) | **median −0.012**, IQR [−0.023, +0.007] | large (1200–2700 alive) |

Under a proper phenotype space the evolution-over-physics advantage **vanishes**
(IQR straddles zero). This confirms Finding 3 from the other direction: the +0.054
was never real learning. In the 3-DOF map, `field_interaction` (the one gene that
aligns chemotaxis with the boundary reward) is tightly coupled to survival, so
selection was *forced* to raise it — which the benchmark read as "learning". Give
evolution 12 independent knobs and it optimizes actual survival (big healthy
populations) instead of that one metric, and the edge disappears.

So the project's single quantitative "learning beyond physics" result is an
artifact of a broken encoding. Fixing the encoding is a genuine improvement **and**
it removes the headline number. Both facts are true and both matter.

## Finding 9 — POSITIVE: the gene pool genuinely stores a decodable, decaying trace of past environments

Borrowing the standard neuroscience/BCI method (population decoding + forgetting
curve) to test the project's core claim "the gene distribution *is* memory". Three
environments (toxic / starved / abundant), 8 seeds; a leave-one-seed-out
cross-validated logistic decoder predicts *which environment the population was
recently in* from its state vector, at increasing delays into a neutral washout.
(Run with `orthogonal_expression=True` + `carrying_capacity` so populations survive
the stress and can be read — a documented choice that gives the hypothesis its best
shot.)

```
 delay | full(16) | baseFreq(4, HERITABLE) | phenotype(12) | shuffled   (chance 0.333)
     0 |   0.917  |         0.750          |     0.958     |  0.358
   100 |   0.583  |         0.458          |     0.625     |  0.367
   200 |   0.667  |         0.500          |     0.667     |  0.217
   400 |   0.417  |         0.458          |     0.417     |  0.275
   800 |   0.417  |         0.333          |     0.417     |  0.383
```

Two things make this a **genuine, clean positive**:
1. **It is real, not leakage.** The shuffled-label control sits at chance (~0.33–0.38)
   at every delay, and CV is leave-one-*seed*-out, so the decoder generalizes to
   runs it never saw.
2. **It is genuinely genetic, not transient physiology.** The purely *heritable*
   base-frequency vector (4-d — literally "the gene distribution") alone decodes the
   recent environment at **0.750** immediately and decays to **0.333 = chance** by
   delay 800. That decay is a real **forgetting curve** on the heritable substrate.

**So "population genetics is memory" is empirically true — in the storage/retention
sense.** The gene pool holds a readable, time-decaying record of recent experience.
This is the first project claim that survives proper controls.

**Important scope.** This is *storage/retention* memory (a trace of the past that
fades), NOT *associative recall* (cue → stored memory), which Finding 6 showed does
NOT emerge. A biological hippocampus does both; DAERWEN now has evidence for the
storage half only. Caveats: 3 environments, 8 seeds, curve is noisy (non-monotonic
at d=100–200); a sharper study would use more environments and finer delays.

### Finding 9b — but the memory is LOW-CAPACITY and encoding-strength-sensitive

Scaling the test to 6 environments (and lighter params: cap 800, encode 600,
6 seeds) collapses it: heritable 6-way decoding is 0.14–0.31 (chance 0.167) — near
chance. Two causes, disentangled:
- **Encoding strength matters.** A control with the 3 well-separated environments
  under the *same light params* also drops (≈0.28–0.67, noisy) vs 0.75 under the
  strong params of Finding 9. The trace needs enough selection time + population to
  form cleanly.
- **Genuine collinearity (capacity limit).** RSA on the per-environment mean genetic
  states shows they are nearly identical (pairwise r 0.80–1.00; `waste_rich` vs
  `nutrient_rich` r = **1.00**), and the geometry is not reproducible across seeds
  (split-half r ≈ 0). The substrate has too few independent selectable axes
  (cf. Findings 4, 7) to give many environments distinct genetic signatures.

Net: the memory is real (Finding 9) but **low-dimensional** — it reliably stores a
few coarse, well-separated regimes given adequate selection, and does not support
fine multi-way discrimination. This is consistent with the "lossy, abstract"
hippocampus framing, but it is a real capacity ceiling, not a high-fidelity store.

## Finding 10 — POSITIVE (recall side): "savings" — a latent memory speeds relearning

Finding 6 showed no *associative* (arbitrary-cue → memory) recall. This tests a
different, classic recall phenomenon — **savings on relearning** (Ebbinghaus): even
after a memory has decayed below the decoding threshold, re-exposure may recover the
original state faster than a naive first exposure, revealing a latent trace.

Protocol: expose to `starved` → 1000-step neutral washout (memory ~un-decodable) →
re-expose, tracking L2 distance of the heritable base-freq state to the remembered
target S_A; compare to a NAIVE population meeting `starved` for the first time
(8 seeds, `orthogonal_expression`).

```
 re-exp step | EXPERIENCED | NAIVE | savings (naive − exp)
      0       |    0.079    | 0.182 |  +0.103
      160     |    0.056    | 0.126 |  +0.071
      320     |    0.053    | 0.085 |  +0.032     median savings +0.080
```

The experienced population is closer to the remembered state at *every* point —
including re-exposure step 0, i.e. after the washout — and re-acquires it faster
(the gap shrinks as both converge). **Genuine savings: a latent memory persists
below the readout threshold and accelerates relearning of the same environment.**
This is recall-adjacent (facilitated re-acquisition), NOT cue-triggered associative
recall (still absent, Finding 6). Caveat: naive uses a different seed, so per-pair
founder variation is averaged over the 8 seeds rather than perfectly controlled.

## Finding 11 — NEGATIVE: no "emotional weighting" (memory depth ≠ selection strength)

The hippocampus encodes survival-relevant events more strongly and for longer. Test:
does a *stronger* selection pressure leave a *deeper* and *more persistent* genetic
memory? Decode a `stress` (elevated inhibitor) vs `neutral` contrast at three
intensities (0.12 / 0.25 / 0.45), across the forgetting curve (10 seeds).

```
intensity | d0 accuracy | mean-over-delays (persistence)
   0.12    |    0.800    |    0.840
   0.25    |    0.700    |    0.820
   0.45    |    0.800    |    0.820
```

Memory depth and persistence are **flat across selection strength** — no emotional-
weighting effect. Likely because once selection fixes the relevant trait, more
pressure cannot push it further (a ceiling). (Aside: the coarse 2-way "stressed vs
not" stays decodable ~0.8–0.95 out to 700 steps — much longer than the fine 6-way
"which environment", i.e. coarse memories outlast fine ones, which is sensible.)

## Finding 12 — ROOT-CAUSE FIX WORKS: activating inert channels expands memory capacity

Findings 4/7/9b traced the low memory capacity and the absence of associative
learning to one root cause: too few *causally active* channels (only ATP/nutrient/
inhibitor affect behaviour or survival; channels 3–10 are inert). The fix
(`rich_channels`, opt-in) gives each particle an independently-evolvable gene per
inert channel — attractive+nutritious if >0, repulsive+toxic if <0 — so those
channels now steer movement AND affect survival, adding ~8 independent environmental
axes.

Payoff test: 6 environments, each elevating a *different* inert channel; decode which
environment from the heritable state, `rich_channels` OFF vs ON.

```
                          decode (d=0)   d=800    RSA env-similarity
rich_channels OFF (base-freq) 0.167       0.167        1.00   (identical, unremembered)
rich_channels ON  (channel-genes) 0.625   0.375        0.31   (distinct, remembered)
                                  chance = 0.167
```

Without the fix these environments leave **no** genetic trace (decoding exactly at
chance, environments genetically identical). With it they become a **decodable,
heritable, decaying memory** (0.625 ≈ 4× chance; shuffle control 0.11), and the
environments become distinct (RSA 1.00 → 0.31). **So the diagnosed bottleneck was
real and the root-cause fix directly expands memory capacity** — six previously
un-memorable environments become remembered.

Caveats: 4 seeds, light params; `channel_genes` is a modifier-locus vector (a clean
independent inheritance channel, not DNA-encoded); the rich metabolism/sensing loop
is compute-heavy (the full OFF+ON sweep was too slow to finish in one run — OFF and
ON were measured separately).

### Finding 12b — but rich_channels does NOT unlock associative recall (deeper cause found)

We then asked whether the richer substrate also fixes the associative-learning
failure of Finding 6, via the sign-flip test (cue on channel 5, reward correlated
vs anti-correlated), under rich_channels — both coupled (cue is also food) and
decoupled (`cue_channels`: sensable but non-nutritious, pure cue):

```
                          median(left − right) cue-gene   seeds flipping correctly
coupled  (cue = food)          +0.025                       1/6
decoupled (pure cue)           −0.079                       2/6
```

Neither flips: association still does not emerge. The tell is the same as Finding 6
— reward occupancy is similar in both conditions (~0.3–0.7), i.e. **the population
reaches the reward by positional selection (survival of whoever is near it + local
reproduction), which works without following any cue**, so the cue-gene has no
selective pressure and just drifts.

**Refined diagnosis.** Adding channels fixed *capacity* but not *recall*, because
recall was never bottlenecked by sensing. It is blocked by the **learning rule
itself**: population-genetic selection is positional/survival-based and never makes
a predictive cue *necessary*. No amount of sensory richness creates pressure to
learn a cue→reward association. Unlocking associative recall would require a
different learning mechanism (e.g. within-lifetime/Lamarckian plasticity where
following a cue is rewarded within a generation), not more channels. This cleanly
separates the two halves: **storage capacity is a substrate problem (fixed);
associative recall is a learning-rule problem (open).**

## Finding 13 — within-lifetime plasticity produces the associative learning genetics could not

Finding 12b concluded that associative recall is a *learning-rule* problem, not a
sensory one. Test: add opt-in `lifetime_plasticity` — each particle carries a
learned cue-weight updated every step by a reward-modulated Hebbian rule
(`w[c] += lr · Δenergy · local_conc[c]`: a cue coinciding with reward becomes
attractive, one coinciding with starvation aversive), added to its chemotaxis.
Re-run the sign-flip test (pure `cue_channels` cue on ch5, reward correlated vs
anti-correlated), measuring the *learned* weight, 6 seeds, lr=2.0.

```
                          median(left − right) learned weight   seeds with L>R
pure genetic (Finding 12b)     −0.079                              2/6   (no signal)
within-lifetime plasticity     +0.081                              5/6   (consistent)
```

The learned cue-weight is reliably higher when the cue predicts reward than when it
is anti-correlated (5/6 seeds; correlated 0.12–0.23, anti 0.05–0.16). **This is
genuine associative learning — the first mechanism in this project that produces it,
which pure population genetics could not.** Honest scope: the effect is *modest*
(weights ~0.15, a graded difference not a sign reversal — the anti weight stays
weakly positive), lr=2.0 is a chosen value, and — critically — it is achieved by a
**hand-designed learning rule**, i.e. an added prior, not emergence. So associative
recall is demonstrated to be *achievable*, but *engineered in*, not emergent from the
physics/chemistry/genetics substrate. Test: `memory_association_rich.py` with
`lifetime_plasticity=True`.

## Finding 14 — LEARNING EMERGES from selection (Baldwin effect + genetic assimilation)

Finding 13 got recall by hand-setting the learning rate — a prior. Here the rate is
NOT set: it is a heritable, mutating gene (`evolvable_plasticity`), randomly
initialized (negative/zero/positive). The population lives in a world that changes
faster than generations (an oscillating reward patch), so a static genetic
preference cannot track it; only within-lifetime learning to follow the cue can.
Three worlds, 4 seeds; we watch the population-mean learning-rate gene evolve:

```
step   learnable (reward moves, cue marks it)   noise (cue useless)   static (reward fixed)
   0            1.00  (learned cue-w 0.01)            1.01                  1.00 (0.01)
 600            1.70  (0.25)                          1.73                  1.49 (0.23)
1200            2.94  (0.25)                          1.82                  0.67 (0.05)
1800            2.34  (0.25)                          1.66                  0.46 (0.04)
```

Read the columns:
- **learnable (fast-changing, learnable):** the learning-rate gene climbs and stays
  high (≈2.3); the population keeps learning to follow the cue (learned weight ≈0.25).
- **static (fixed target):** learning is useful at first (rises to 1.5) then **decays
  to 0.46** — once genetics assimilates the fixed preference, learning is redundant
  and is selected *down*. This is genetic assimilation.
- **noise (no useful cue):** plateaus mid-range (~1.7) on generic adaptive value.

**So the *capacity to learn* emerges from selection alone** — it rises exactly when
the environment changes faster than generations, and decays when genetics can
assimilate the solution. Crucially it is selected *down* in the static world, which
rules out "plasticity just drifts up" — the effect is demand-driven. This is the
classic evolutionary origin of learning (Baldwin, 1896), reproduced from the
substrate; the learning rate was never set by hand.

**Honest scope — the loop is not fully closed.** What emerged is *whether and how
much* to learn (the rate/existence), driven by selection. The learning-rule
*template* itself — the reward-modulated Hebbian form `Δw ∝ Δenergy · cue` — is still
hand-designed. Fully closing the emergence loop would require that update *form* to
arise from lower-level (e.g. intra-particle chemical) dynamics too. So: storage
memory emerges (Findings 9–10); the *decision to learn* emerges (this finding); the
*mechanism* of learning remains a designed prior. Test:
`tests/honest/evolution_of_learning.py`.

## Finding 15 — the learning MECHANISM did not emerge from generic chemistry (honest negative)

Third step toward closing the emergence loop: instead of a hand-written learning
rule, give each particle an internal molecule `mem` evolving by a *generic bilinear
mass-action reaction* with 5 randomly-initialized, evolvable coefficients
(`d_mem = a·cue·rew + b·cue + c·rew − d·mem`), coupling to cue-following via evolved
`e`. If evolution wires up `a,d,e` into a coincidence-detector, associative learning
would emerge from chemistry. Behavioral test: track a moving reward (4 seeds),
vs a no-learning control.

```
                                   reward-tracking occupancy (step 1500)   evolved a / e
MOVING reward, chemical_learning            0.48                            +0.07 / +0.67
MOVING reward, NO learning (control)        0.68                            —
STATIC reward, chemical_learning            0.33                            −0.10 / −0.49
```

**Chemistry did not beat the no-learning control (0.48 < 0.68): the mechanism did
not emerge functionally.** The reason sharpens the whole arc's central obstacle: a
*moving* reward still does not *require* within-lifetime learning — the population
tracks it by positional selection + local reproduction (the no-learning control
tracks it fine). With learning unnecessary, there is too little fitness pressure to
wire up a coincidence detector from scratch, so the reaction coefficients drift
inconsistently (and are selected *negative* in the static world). This does not
prove emergence is impossible — it shows that in a spatial ecology where local
reproduction lets the population "flow" with resources, the fitness value of
individual associative learning is too low for its *mechanism* to self-assemble.

**What it would take:** a task where an individual must *use a learned association
within its own lifetime* in a way local reproduction cannot shortcut (e.g. a cue
whose meaning is only knowable by within-life experience, with no spatial
give-away). Designing such a task in a reproduction-driven ecology is the real crux
— and is the honest open problem the emergence loop now rests on.

## Finding 16 — POSITIVE: the system's output can be shaped by its consequences (operant, at the population level)

First embodied probe. The "output" (a voltage-like effector reading) is the fraction
of the population occupying a target zone; producing that output triggers a chemical
reward. Per Finding-2's lesson (only chemical/selective pressure changes genes), the
reward is ATP injected proportional to the action, and we compare **contingent**
(reward → the target zone, crediting the actors) vs a **yoked** control (same reward
amount → a random zone, decorrelated from the action). 3 seeds each.

```
step   contingent occupancy   yoked occupancy   (baseline ≈ 0.20)
   0        0.30                   0.29
 400        0.96                   0.20
 800        0.94                   0.15
1200        0.96                   0.22
in-zone turnover per step ≈ 0.99  (both) -> "grew (selection)", not "moved"
```

**Confirmed operant conditioning:** the output rises to ~0.95 under contingent reward
and stays at baseline (~0.20) under yoked — so it is the *action–reward contingency*,
not mere presence of reward, that shapes the output (the yoked control rules out
trivial chemotaxis). This is the first fully ground-truthed, control-validated "yes,
it learns" in this suite.

**But the mechanism is selection, not individual cognition** (Finding-1-style
confound, now measured): in-zone turnover ≈ 0.99 means the target zone is repopulated
by *new* particles almost every window — the population *grows into* the rewarded
configuration by differential reproduction; no individual learns to go there. So
operant shaping holds at the *population* level (you can train the system's collective
output with contingent chemical reward), not the individual level. The harder,
still-open version is a *global/delayed* reward that selection cannot credit to the
action — which would require the individual learning that does not emerge here.

## How to reproduce

```bash
# from daerwen3.5/, with the 'chu' env (numpy + rust chem_sim)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/movement_probe.py          # Finding 4 (seconds)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/honest_harness.py          # Findings 1-2 (~4 min)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/association_experiment.py  # Finding 6 (~6 min)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/population_decoding.py     # Finding 9 (~6 min)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_savings.py          # Finding 10 (~5 min)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_emotional_weighting.py  # Finding 11 (~8 min)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_forgetting_deep.py  # Finding 9-deep + RSA
CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_capacity_rich.py    # Finding 12 (rich_channels payoff)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/memory_association_rich.py # Finding 13 (engineered recall)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/evolution_of_learning.py   # Finding 14 (Baldwin)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/emergent_learning_chem.py  # Finding 15 (mechanism did not emerge)
CUDA_VISIBLE_DEVICES=-1 python tests/honest/operant_voltage.py         # Finding 16 (operant, shaped output)
```

---

## Honest conclusion

DAERWEN's engine is competently built and its `RESEARCH_STATUS.md` is unusually
candid. The strongest positive result is Finding 9: using a standard neuroscience
population-decoding method, the **heritable gene pool genuinely stores a decodable,
time-decaying trace of past environments** — so "population genetics is memory" is
empirically true in the *storage/retention* sense, with proper controls. What does
*not* hold is the rest of the headline "learning" story: the one
quantitative "beats physics" number (a) was a single lucky draw from a wide,
extinction-prone distribution (Finding 2), (b) reduces to tuning one sensor gain
on top of a "physics baseline" that already contains natural selection (Finding 3),
and (c) **vanishes entirely once the degenerate gene encoding is fixed** (Finding 8).
The architecture, as written, **cannot** support the higher-level capabilities the
narrative reaches for: behavior and fitness are wired to a fixed tiny set of
channels (Finding 4), and even after we removed that prior, associative learning
did not emerge under any regime (Finding 6). Underlying much of this is a phenotype
encoding with only 3 effective degrees of freedom (Finding 7) — which we fixed
(`orthogonal_expression`), restoring 12 and greatly improving viability, at the
cost of the headline number it was propping up.

We then *tried to fix the deepest limitation* (Finding 6): we gave the substrate
the missing sensory pathway and a clean independent gene, and still could not get
associative learning to emerge under any of four regimes. So the barrier is not a
single missing feature — it is layered (non-orthogonal gene encoding; chemotaxis
too weak vs Brownian noise; ecological dynamics that decouple position from
reproductive success). Each layer was found by a test whose control refused a
false positive.

Two honest paths forward, which should be separated rather than blended:
- **As artificial life / origin-of-life** (chem_sim, Eigen error-threshold): the
  work is legitimate and worth continuing on its own terms.
- **As a memory/AGI module**: this would require redesigning three coupled things
  at once — an orthogonal gene→phenotype encoding, a sensory pathway strong enough
  to matter against noise, and a task/environment where following a learned signal
  is the difference between reproducing and not. Finding 6 shows that adding any
  one of these alone does nothing. Until all three exist, "learning" numbers
  measure physics + one-gene tuning, not memory.
