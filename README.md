# DAERWEN 3.5

> **An Artificial Hippocampus: Population-Genetic Memory Through Ecological Evolution**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19604736.svg)](https://doi.org/10.5281/zenodo.19604736)
[![Active Research](https://img.shields.io/badge/status-active%20research-brightgreen)](RESEARCH_STATUS.md)

> 📋 **[Research Status & Open Bottlenecks](RESEARCH_STATUS.md)** — validation results, architectural limits, and the path forward.

**DAERWEN** (Darwin-inspired Emergent World Engine) is a **hippocampus module** for larger AI systems. It uses population-level evolution in a 2D ecology to form, maintain, and recall memories — the same way biological hippocampus encodes experience through lossy, abstract, reconstructive storage.

---

## 💫 Mission Statement

**I share Elon Musk's vision: the advancement of civilization matters more than individual recognition.**

This project was built entirely with AI agent assistance. My English is limited, but I believe in the power of collaboration—human and AI working together to explore new paths to AGI.

**If someone is inspired by this work and creates something better, even if this project remains unknown, I will be happy. As long as civilization progresses.** Science has no boundaries.

---

🎯 **Core Philosophy**: Minimize the "Designer's Trap" by reducing arbitrary priors and maximizing emergence. A floor of priors is unavoidable (you can't simulate from the Big Bang) — the discipline is to keep pushing it lower and stay honest about where it sits.

---

## 🧭 Where This Project Stands

This is **a genuinely new research direction**, not a variation on existing ML pipelines. Its theoretical framework has reached **logical closure** — physics → chemistry → genetics → evolutionary learning forms a self-consistent substrate for intelligence to emerge from.

The current implementation is an **initial version in which the designer's fingerprints are still visible**. Even so, we have already observed real learning-like properties: strong resistance to forgetting, stable recovery from catastrophic disturbance, and adaptation to noisy environments. These are not simulated — they are measured (see [Current Status](#-current-status--limitations)).

The ongoing task is to **progressively reduce arbitrary priors** and let more of the system's behavior emerge from the substrate itself, rather than from choices made by the designer.

---

## 🧠 What This System Actually Is

**DAERWEN is an artificial hippocampus** — a memory module that stores, maintains, and recalls experience through population-level evolution, not through databases or neural network weights.

### Why "hippocampus"?

The real hippocampus (in your brain right now):
- Receives **abstract features** from the cortex, not raw pixels
- Stores memories as **sparse, lossy patterns** — not recordings
- Recalls by **reconstructing** from partial cues — which is why memories are creative and sometimes wrong
- Encodes **stronger memories for emotional/dangerous events** — survival-relevant memories persist longer
- Runs **continuously** — even during sleep (consolidation)

DAERWEN does all of these through a completely different mechanism: **population genetics**.

### Population genetics *is* memory

**The distribution of genes across the population is the memory.** An environment that kills particles with gene-X means gene-X disappears from the population forever. That disappearance IS the memory of "gene-X doesn't work here."

- A population that survived a drought **remembers** the drought (drought-resistant genes dominate)
- A population in a toxic zone **remembers** the toxin (toxin-avoidant genes spread)
- Over time, mutations **drift** the memory — old memories fade unless reinforced (forgetting)
- Strong selection events leave **deeper marks** — like emotional memories in real brains

### Input: pre-processed features, not raw data

The real hippocampus never sees raw pixels. Visual information passes through 5-6 cortical layers before reaching it, arriving as abstract concepts ("cat", "kitchen", "morning").

DAERWEN should receive **pre-processed feature vectors** (from CNN or other models), not raw sensor data. The feature vector is injected as **sparse, theta-rhythmic spike bursts** into specific regions of the 2D world — matching how the real hippocampus receives input.

### Output: memory state, not text

DAERWEN's output is a **continuous state vector** derived from population statistics:
- `familiarity`: how similar is the current situation to past experience?
- `threat_level`: did similar situations cause high mortality before?
- `spatial_memory`: where were resources / dangers located?
- `novelty`: is this something the system has never encountered?

These signals are read by downstream modules (decision-making, planning) — not by humans directly.

### DAERWEN is one module, not the whole brain

| Brain Part | Function | DAERWEN? |
|-----------|----------|----------|
| Visual cortex | Process raw images | ❌ Use CNN (ResNet etc.) |
| Auditory cortex | Process raw audio | ❌ Use audio models |
| Thalamus | Route sensory signals | ❌ Separate attention/routing module |
| **Hippocampus** | **Form and recall memories** | **✅ THIS IS DAERWEN** |
| Basal ganglia | Select actions | ❌ Use RL agent |
| Prefrontal cortex | Plan and reason | ❌ Use LLM |

See [`docs/design/HIPPOCAMPUS_ARCHITECTURE.md`](docs/design/HIPPOCAMPUS_ARCHITECTURE.md) for the full architectural design.

---

## 🌟 What Makes This Different?

Unlike neural network memory (replay buffers, weight matrices, embeddings):
- ❌ No training phase — memories form through natural selection in real time
- ❌ No backpropagation — adaptation is Darwinian, not gradient-based
- ❌ No fixed capacity — population can grow or shrink with the environment
- ❌ No catastrophic forgetting — old memories coexist with new ones (forgetting rate 0.010)

Instead:
- ✅ **Population = memory**: gene frequencies encode past experience
- ✅ **Lossy and abstract**: like real hippocampus, stores summaries not recordings
- ✅ **Continuous operation**: runs 24/7, always integrating new experience
- ✅ **Reconstructive recall**: querying memory returns a creative reconstruction, not a playback
- ✅ **Emotional weighting**: stronger selection pressure = deeper memory imprint

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10 or higher
python --version

# Install dependencies
pip install numpy websockets orjson
# Optional: torch (for GPU acceleration)
```

### Run the Simulation
```bash
# Start the ecology engine
python scripts/start_engine.py

# Or CPU-only version
python scripts/start_engine_cpu.py

# Open web UI: http://localhost:8765
```

### Run Tests
```bash
# AI Learning Benchmark (continual learning metrics)
python tests/benchmarks/brutal_benchmark_fast.py

# Unconscious Learning Suite (implicit pattern extraction)
python tests/benchmarks/unconscious_learning_suite.py
```

---

## 📊 Current Status & Limitations

**We choose to be upfront about what this system does and doesn't do.**

### What works (measured on current benchmark v2)

| Capability | Score | Notes |
|-----------|-------|-------|
| **Beats physics baseline** | **+16.3%** | First stable positive result — selection actually produces better-than-random-drift behavior (2026-04-17) |
| **Forgetting resistance** | avg_forgetting ≈ 0.010 | Extremely low — system doesn't "unlearn" past environments. For context, vanilla MLP + Naive CL on Split MNIST forgets 99.2%. Paradigms differ but characteristic is dramatic. |
| **Noise robustness** | 0.83–0.98 | Survives and adapts under strong environmental noise |
| **Catastrophe recovery** | 1.0 | Recovers from mass extinction events |
| **Backward transfer (BWT)** | ±0.01 | Essentially flat — past tasks neither improve nor degrade |

### What does NOT work yet

| Limitation | Current value | What it means |
|-----------|---------------|---------------|
| **Final Average Performance (FAP)** | 0.05–0.11 | System barely beats pure-physics baseline on spatial tasks |
| **Overall grade** | F (35–37 / 100) | On a CL-style benchmark, it fails hard |
| **Task-specific adaptation speed** | Too slow | 200 training steps is nowhere near enough evolutionary cycles |
| **Physics baseline stability** | High variance | Same task, two runs → baselines vary 0.47 ↔ 0.74; benchmark itself is noisy |

### Why the F grade doesn't mean the system is broken

The current benchmark (`brutal_benchmark_fast.py`) was designed to measure *sequential task adaptation* in the tradition of continual-learning literature (Lopez-Paz & Ranzato 2017). That is a good fit for neural networks trained by gradient descent. It is a **poor fit** for an evolutionary ecology:

- Evolution operates on generations, not training steps
- The system's hypothesized strength is **simultaneous multi-modal integration**, not sequential task switching
- Physics baselines dominate alignment scores when particles naturally cluster in gradient wells

**What we're working on next** is a different benchmark — **signal-reward decoupling under simultaneous multi-modal input** — which tests what this system is actually theoretically good at (associating arbitrary sensory signals with energy rewards through evolutionary pressure, rather than hard-coded chemotaxis).

Sample result files live in [`benchmark_results/`](benchmark_results/).

---

## 🤖 Development Approach

**All code, tests, and documentation are written by AI agents under human direction.** The human contributor provides vision, judgment, and course-correction; implementation and iteration are done by AI.

This is not a limitation to hide — it is **part of the experiment**. If physics-grounded ecological intelligence is a viable path, it should be explorable by a single person with AI leverage, not only by funded research groups. The openness of the code is also a record of what AI-assisted research can produce when aimed at first-principles problems.

---

## 🏗️ Architecture Overview

### DAERWEN in a larger AI brain

```
Sensors → CNN (visual cortex) ──→ Feature vector ──→ DAERWEN (hippocampus)
       → Audio model          ──→ Feature vector ─↗        │
       → Other models         ──→ Feature vector ─↗    Memory state
                                                        (5-10 floats)
                                                            │
                                                    Decision module
                                                    (basal ganglia)
                                                            │
                                                      Action output
```

### DAERWEN internal architecture

```
Feature spikes (sparse, theta-rhythmic)
        │
┌───────▼──────────────────────────┐
│  ExternalInput                   │  ← Receives pre-processed features
├──────────────────────────────────┤
│  Ecology Engine (core.py)        │  ← 2D particles + chemistry + genetics
│  - 24/7 continuous evolution     │  ← Population = memory substrate
│  - Gene expression (Rust/pyo3)   │  ← Uniform composition mapping
│  - Chemical field dynamics       │
├──────────────────────────────────┤
│  SystemOutput                    │  ← Memory state vector
│  - Population statistics         │  ← familiarity, threat, spatial, novelty
│  - Chemical field state          │
└──────────────────────────────────┘
```

**Key Design Principle**: 
> DAERWEN is a memory module, not a complete brain. It receives abstract features and outputs memory state. All external interaction goes through `ExternalInput`/`SystemOutput`.

---

## 📚 Documentation

### For Researchers
- [**Hippocampus Architecture Design**](docs/design/HIPPOCAMPUS_ARCHITECTURE.md) — Why DAERWEN is a hippocampus, I/O specification, memory mechanism
- [AGI Vision & Roadmap](docs/AGI_VISION.md) — Emergence-driven path to AGI
- [Genotype→Phenotype First Principles](docs/design/GENOTYPE_TO_PHENOTYPE_FIRST_PRINCIPLES.md) — Multi-layer expression theory
- [Dual-Process AGI Architecture](docs/notes/DUAL_PROCESS_AGI.md) — Brain module integration design

### For Engineers
- [Core Overview](docs/CORE_OVERVIEW.md) — Technical architecture
- [API Reference](docs/API_REFERENCE.md) — ExternalInput/SystemOutput specification
- [MVS Runbook](docs/MVS_RUNBOOK.md) — Minimal Viable System setup
- [Engineering Spec & Guardrails](docs/design/ENGINEERING_SPEC_AND_GUARDRAILS.md) — Invariants and safety rails

---

## 🧪 Current Research Status

### ✅ Completed
- Physics-chemistry-genetics coupled engine
- Unified ExternalInput/SystemOutput interface
- Professional AI learning benchmark suite
- Unconscious learning test framework
- Multi-layer gene expression (basic)

### 🚧 In Progress
- Regulatory gene networks (deeper expression layers)
- Dual-process LLM integration (bridge.py)
- Robustness improvements (edge cases)

### 📋 Roadmap
- Scale to 1000×1000 worlds
- Long-term evolution experiments (10M+ steps)
- Real-world application exploration (adaptive control, drug discovery)

---

## 🤝 Contributing

We welcome contributions! This is an **exploration**, not a finished product.

### Ways to Contribute
- 🐛 **Bug reports**: Found a crash or unexpected behavior?
- 💡 **Ideas**: Suggest improvements or new experiments
- 🔬 **Experiments**: Run tests with different parameters and share results
- 📝 **Documentation**: Improve clarity or add examples
- 🧬 **Code**: Implement features from the roadmap

### Getting Started
1. Fork the repository
2. Create a branch (`git checkout -b feature/your-idea`)
3. Make changes and test
4. Submit a Pull Request

**Code Style**: We value clarity over cleverness. Comment your reasoning, especially for non-obvious decisions.

---

## 🌍 Related Projects

### Inspirations
- **Tierra**: Digital evolution (instruction set)
- **Avida**: Digital organisms (genetic programming)
- **Lenia**: Continuous cellular automata (mathematical beauty)
- **Active Inference**: Free energy minimization (cognitive science)

### Differences
| Feature | DAERWEN | Others |
|---------|---------|--------|
| **Substrate** | Physics + Chemistry + Genetics | Instructions / Cells / Math |
| **Learning** | Population-level adaptation | Individual / None |
| **Control** | Environmental interventions only | Direct / None |
| **Philosophy** | Designer Trap awareness | Implicit |
| **Goal** | Dual-process AGI | Digital life / Art / Theory |

---

## 📖 Citation

If you use DAERWEN in your research, please cite:

```bibtex
@software{hou2026daerwen,
  author       = {Hou, Zehao},
  title        = {DAERWEN: Physics-Grounded Ecological Intelligence},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {3.5.0},
  doi          = {10.5281/zenodo.19604736},
  url          = {https://github.com/Chunsisisi/daerwen3.5}
}
```

Or plain text:

> Hou, Zehao. *DAERWEN: Physics-Grounded Ecological Intelligence* (v3.5.0). Zenodo, 2026. [https://doi.org/10.5281/zenodo.19604736](https://doi.org/10.5281/zenodo.19604736)

---

## 📄 License

GNU Affero General Public License v3.0 - See [LICENSE](LICENSE) for details.

**Why AGPL v3?**
- ✅ Research stays open forever — derivatives must also be open source
- ✅ Covers network use — no "SaaS loophole" (running as a service still requires open sourcing)
- ✅ Fosters open science and civilizational progress over private capture

---

## 🙏 Acknowledgments & Specific Inspirations

This project stands on the shoulders of prior work. We list below *specific* inspirations, with the corresponding code location where the idea landed:

### Theoretical & architectural
- **Active Inference / Free Energy Principle** (Friston, 2010-present) — shaped the "environment-driven adaptation without reward function" philosophy throughout `engine/core.py`
- **Turing's reaction-diffusion systems** (Turing, 1952) — basis for the multi-species chemical dynamics in `engine/core.py`
- **Lenia / Tierra / Avida** — the lineage of emergent digital life; DAERWEN's substrate choice (physics+chemistry+genetics, not pure math/cells/instructions) is defined *against* these

### Benchmark methodology
- **Lopez-Paz & Ranzato (2017)** *Gradient Episodic Memory* — the BWT/FWT/FAP metrics and evaluation matrix pattern in [`tests/benchmarks/brutal_benchmark_fast.py`](tests/benchmarks/brutal_benchmark_fast.py)
- **SICR** *Statistically-Induced Chunking Recall* — inspired Level-1 implicit pattern extraction test in [`tests/benchmarks/unconscious_learning_suite.py`](tests/benchmarks/unconscious_learning_suite.py)
- **CADI / Order Parameters** — chaos-aware design index used as the conceptual basis for the Level-2 test in the same file
- **MetrIntMeas / Swarm Intelligence Metrics** — basis for the Level-3 collective behavior metric in the same file

### Design concepts that shaped this project
- **"Designer's Trap"** — the principle that every arbitrary prior a designer writes into a system becomes a ceiling on emergence. Motivates the minimal-rule substrate philosophy.
- **Dual-Process architecture** — subconscious ecology engine (always-on evolutionary substrate) paired with a conscious LLM layer (on-demand symbolic reasoning). See [`docs/notes/DUAL_PROCESS_AGI.md`](docs/notes/DUAL_PROCESS_AGI.md).

---

## 📬 Contact

- **GitHub Issues**: For bug reports, feature requests, questions, and discussions
- **Pull Requests**: Contributions are welcome!

---

## ⚠️ Disclaimer

This is **research code**, not production software. Expect:
- 🐛 Bugs and edge cases
- 📊 Parameter sensitivity
- 🔧 Breaking changes as we iterate

But we strive for:
- 📖 Clear documentation
- ✅ Reproducible results
- 🤝 Open communication

---

## 🌟 Star History

If you find this project interesting, please consider starring it! ⭐

It helps others discover this work and motivates continued development.

---

**Status**: Active research (as of April 2026)  
**Version**: 3.5 — artificial hippocampus module with validated memory properties (forgetting 0.010, +16.3% over physics baseline, catastrophe recovery 1.0)  
**Maintainer**: Hou Zehao ([@Chunsisisi](https://github.com/Chunsisisi))

---

> "In creating artificial life, we don't just create life — we re-examine the nature of life, the origin of rules, and the possible truth of the universe."
