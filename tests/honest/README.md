# `tests/honest/` — reproducible, controlled tests

An independent, honest re-examination of DAERWEN's memory/learning claims. Every
test is seeded, multi-seed, and ships with a control (shuffle / yoked / no-learning)
and a baseline so a result cannot be an artifact of a hand-picked metric. The full
narrative and evidence is in [`FINDINGS.md`](FINDINGS.md); the synthesis is in
[`../../docs/notes/MEMORY_EMERGENCE_REPORT.md`](../../docs/notes/MEMORY_EMERGENCE_REPORT.md).

All capabilities exercised here are **opt-in** `Ecology2DConfig` flags; the engine's
defaults are unchanged.

Run any test with the CPU (Rust `chem_sim` optional):

```bash
CUDA_VISIBLE_DEVICES=-1 python tests/honest/<script>.py
```

| Script | Question it answers | Finding |
|---|---|---|
| `honest_harness.py` | Does evolution beat physics? (seeded, paired, multi-seed) | 1–2 |
| `movement_probe.py` | Which channels can steer movement? | 4 |
| `population_decoding.py` | Does the gene pool store a decodable, decaying trace of the environment? | 9 |
| `memory_savings.py` | Does a decayed memory speed relearning (savings)? | 10 |
| `memory_emotional_weighting.py` | Do stronger events leave deeper/longer memory? | 11 |
| `memory_forgetting_deep.py` | Capacity + representational structure (RSA) | 9b |
| `memory_capacity_rich.py` | Does activating inert channels expand capacity? | 12 |
| `association_experiment.py` | Does associative (cue→reward) learning emerge from genetics? | 6, 12b |
| `memory_association_rich.py` | …with richer channels / a designed plasticity rule? | 13 |
| `evolution_of_learning.py` | Does the *decision to learn* evolve? (Baldwin) | 14 |
| `emergent_learning_chem.py` | Can the learning *mechanism* emerge from generic chemistry? | 15 |
| `operant_voltage.py` | Can the system's output be shaped by its consequences? (operant, yoked control) | 16 |

**One-line summary of what these establish:** population genetics gives emergent,
lossy, low-capacity *storage* memory (with forgetting and savings) and can even
evolve the *decision* to learn — but individual *associative recall* does not emerge,
because reproduction-driven positional selection keeps making individual learning
unnecessary. See `FINDINGS.md` for the evidence and the precise open problem.
