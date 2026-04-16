# DAERWEN 3.5

> **Physics-Grounded Ecological Intelligence: Exploring AGI Emergence from Minimal Rules**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19604736.svg)](https://doi.org/10.5281/zenodo.19604736)
[![Active Research](https://img.shields.io/badge/status-active%20research-brightgreen)](RESEARCH_STATUS.md)

> 📋 **[Research Status & Open Bottlenecks](RESEARCH_STATUS.md)** — current state of validation, identified architectural limits, and the path forward (Phase C: physics-grounded chemistry).

**DAERWEN** (Darwin-inspired Emergent World Engine) is a research platform exploring how intelligence can emerge from simple physical rules, chemical interactions, and genetic evolution—without hardcoded behaviors.

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

The clearest mental model: **DAERWEN is the subconscious substrate that could let an LLM run the way a human brain does.**

### Population evolution *is* population intelligence

There is no separate "memory module" in this system, by design. **The distribution of genes across the population is the memory.** Behaviors that help survival persist in the gene pool; behaviors that don't are diluted out over generations. This isn't simulated memory — it is emergent memory, the same mechanism biology has used for billions of years. A population of 500 particles carrying evolved genes *is* a stored representation of everything the system has encountered.

### Continuous operation vs. stateless inference

An LLM is fundamentally **stateless**: each inference starts from cold context, driven entirely by the prompt. A human brain is not like this. The subconscious runs 24/7 — heartbeat, hormones, autonomic responses, background consolidation, dreaming. DAERWEN is designed to be that layer:

- **LLM** = conscious, on-demand, symbolic, fast — the tip of the iceberg
- **DAERWEN** = subconscious, always-on, biochemical-ecological, slow — the mass underneath

A continuously evolving ecology doesn't just "remember" its environment — it *is* shaped by it, moment to moment, whether or not anyone is asking questions.

### Output is hormone, not text

The system's outputs are **multi-channel continuous signals** — analogous to hormone levels, muscle tone, emotional state, arousal. Not language. The upper layer (LLM, when present) translates these states into symbolic reasoning when needed, just as a conscious mind translates feelings into words. **Brute text output is the wrong level of abstraction for this substrate.**

### Input is sensory, not text

The central argument: **DAERWEN is designed not to take direct text input.**

Humans don't have text injected into their brains. We see shapes and hear sounds; "text" is a pattern built on top of the visual and auditory channels through learning. The same logic applies here. DAERWEN is intended to receive the full multi-modal sensory bandwidth — spatial patterns (vision-like), temporal oscillations (audio-like), chemical gradients (olfaction-like), contact forces (touch-like) — and it can in principle accept bands humans lack, such as infrared or ultrasonic. If symbolic content reaches the system at all, the more consistent approach is for it to enter through the same sensory channels as everything else.

This is the main reason DAERWEN does not look like any other AI project: it is trying to be the **substrate underneath symbolic intelligence**, not another re-implementation of symbolic intelligence.

---

## 🌟 What Makes This Different?

Unlike traditional AI approaches:
- ❌ No supervised training on datasets
- ❌ No backpropagation or gradient descent
- ❌ No hardcoded behaviors or reward functions

Instead:
- ✅ **Physics-driven**: 2D particle dynamics, chemical diffusion, energy conservation
- ✅ **Chemistry-coupled**: Multi-species chemical reactions shape the environment
- ✅ **Genetics-based**: Multi-layer gene expression (not direct genome→behavior mapping)
- ✅ **Evolutionary learning**: Natural selection at population level
- ✅ **Dual-process architecture**: Subconscious (ecology engine) + Conscious (LLM integration, planned)

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

```
┌─────────────────────────────────────────┐
│  ExternalInput (Environmental Control)  │
└────────────┬────────────────────────────┘
             │
      ┌──────▼──────┐
      │   Engine    │  ← 2D Physics + Chemistry + Genetics
      │  (Core.py)  │  ← 24/7 Continuous Evolution
      └──────┬──────┘
             │
      ┌──────▼──────────┐
      │ SystemOutput    │  ← Visualization + Statistics
      │ (Aggregated)    │  ← Emergence Detection
      └──────┬──────────┘
             │
    ┌────────▼─────────┐
    │  Controllers     │  ← State Aggregator
    │  (Optional)      │  ← Predictive Controller (WIP)
    └──────────────────┘
```

**Key Design Principle**: 
> All external interaction goes through `ExternalInput`/`SystemOutput` interface. No direct manipulation of internal state.

---

## 📚 Documentation

### For Researchers
- [AGI Vision & Roadmap](docs/AGI_VISION.md) — Emergence-driven path to AGI
- [Genotype→Phenotype First Principles](docs/design/GENOTYPE_TO_PHENOTYPE_FIRST_PRINCIPLES.md) — Multi-layer expression theory
- [Dual-Process AGI Architecture](docs/notes/DUAL_PROCESS_AGI.md) — Subconscious engine + conscious LLM design
- [Unconscious Learning Test Suite](tests/benchmarks/unconscious_learning_suite.py) — Pattern extraction without explicit goals

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
**Version**: 3.5 — initial implementation of the closed theoretical framework; learning properties observed, designer's fingerprints still visible  
**Maintainer**: Hou Zehao ([@Chunsisisi](https://github.com/Chunsisisi))

---

> "In creating artificial life, we don't just create life — we re-examine the nature of life, the origin of rules, and the possible truth of the universe."
