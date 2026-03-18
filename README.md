# DAERWEN 3.5

> **Physics-Grounded Ecological Intelligence: Exploring AGI Emergence from Minimal Rules**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**DAERWEN** (Darwin-inspired Emergent World Engine) is a research platform exploring how intelligence can emerge from simple physical rules, chemical interactions, and genetic evolution—without hardcoded behaviors.

🎯 **Core Philosophy**: Avoid the "Designer's Trap" by minimizing arbitrary priors and maximizing emergence.

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
python tests/ai_learning_benchmark_quick.py

# Unconscious Learning Suite (implicit pattern extraction)
python tests/unconscious_learning_suite.py
```

---

## 📊 Validated Capabilities

Our system demonstrates:

### ✅ Continual Learning
- **BWT (Backward Transfer)**: -0.012 (minimal forgetting)
- **FWT (Forward Transfer)**: +0.011 (positive knowledge transfer)
- **FAP (Final Average Performance)**: 0.656

### ✅ Implicit Pattern Extraction
- Level 1 Score: 0.736 (adapts to statistical regularities without explicit goals)

### ✅ Niche Specialization
- Spatial Distribution Score: 0.857 (population self-organizes into ecological niches)

**These are system-level learning metrics** (population adaptation), not individual-level learning.

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

### For Philosophers
- [**Philosophy Overview**](../PHILOSOPHY_OVERVIEW.md) — The Designer's Trap, Dual-Process AGI, Universe Simulation Hypothesis
- [Philosophical Discussion Record (8-hour deep dive)](docs/notes/哲学讨论记录-从设计师陷阱到宇宙本质.md)

### For Researchers
- [AGI Vision & Roadmap](docs/AGI_VISION.md) — Emergence-driven path to AGI
- [Genotype→Phenotype First Principles](docs/types/GENOTYPE_TO_PHENOTYPE_FIRST_PRINCIPLES.md) — Multi-layer expression theory
- [Learning Test Suite Design](tests/unconscious_learning_suite.py) — World's first unconscious learning benchmark

### For Engineers
- [Core Overview](docs/CORE_OVERVIEW.md) — Technical architecture
- [API Reference](docs/API_REFERENCE.md) — ExternalInput/SystemOutput specification
- [MVS Runbook](docs/MVS_RUNBOOK.md) — Minimal Viable System setup

### For Everyone
- [Project Map](../PROJECT_MAP.md) — Navigate 100+ documents
- [Reading Guide](../READING_GUIDE.md) — Tailored paths for different roles

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

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

**Why MIT?**
- ✅ Permissive (allows commercial use)
- ✅ Encourages adoption
- ✅ Fosters open science

---

## 🙏 Acknowledgments

- **Philosophical Foundation**: 8-hour discussion on the nature of design, emergence, and intelligence
- **Testing Methodology**: Inspired by Lopez-Paz & Ranzato (2017) continual learning metrics
- **Architecture**: Influenced by Active Inference and Free Energy Principle

---

## 📬 Contact & Community

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and brainstorming
- **Email**: [Your email if you want]
- **Twitter/X**: [Your handle if you have one]

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

**Status**: Active research (as of March 2026)  
**Version**: 3.5 (Stable branch with optimal parameters)  
**Maintainer**: [Your name/handle]

---

> "In creating artificial life, we don't just create life—we re-examine the nature of life, the origin of rules, and the possible truth of the universe."
> 
> — From the 8-hour philosophical discussion
