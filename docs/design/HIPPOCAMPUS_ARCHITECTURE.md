# DAERWEN as Hippocampus: Architecture Design

> Established: 2026-04-17/18
> Status: Active design direction

---

## Core Positioning

**DAERWEN is NOT the entire brain. DAERWEN IS the hippocampus module
in a larger AI system.**

Previous (wrong) assumption: DAERWEN = complete AGI substrate
Current (correct) understanding: DAERWEN = one specialized module (memory)

---

## Brain Architecture Mapping

### Complete brain structure and DAERWEN's role

| Brain Structure | Function | In AI System | DAERWEN? |
|----------------|----------|-------------|----------|
| **Brainstem** | Basic survival (heartbeat, breathing) | OS/hardware layer | ❌ Not DAERWEN |
| **Cerebellum** | Motor coordination, timing | PID/control algorithms | ❌ Not DAERWEN |
| **Thalamus** | Sensory routing + attention gating | Attention/routing layer | ❌ Separate module needed |
| **Hypothalamus** | Hormone regulation, homeostasis | Internal state controller | ❌ Separate module |
| **Amygdala** | Threat detection, emotional tagging | Fast threat classifier | ❌ Separate module |
| **Hippocampus** | Memory formation, spatial mapping | **✅ THIS IS DAERWEN** | ✅ |
| **Basal Ganglia** | Action selection (go/no-go gate) | RL agent / decision gate | ❌ Separate module |
| **Cerebral Cortex** | Perception, reasoning, planning | CNN + LLM + other models | ❌ Separate modules |

### Why hippocampus fits

DAERWEN's validated capabilities map directly to hippocampal functions:

| Hippocampal Function | DAERWEN Equivalent | Evidence |
|---------------------|-------------------|----------|
| Long-term memory formation | Population genetics persist across generations | Forgetting rate 0.010 (near-zero) |
| Spatial mapping (place cells) | Particle spatial distribution encodes resource locations | Tibet video: north_frac=0.76 tracking bright regions |
| Learning beyond defaults | Selection pressure accumulates useful adaptations | +16.3% improvement over physics baseline |
| Catastrophe recovery | Survivors' genes rebuild population from memory | Recovery rate 1.0 |
| Temporal pattern memory | Population adapts to cyclical environments | Day/night: 186 generations, largest population |
| Novelty detection | New perturbation → measurable population response | Multimodal experiments show distinct responses |
| Lossy/abstract storage | Genes encode statistical summaries, not exact events | By design — no replay buffer needed |
| "False memory" | Mutations drift genetic memory over time | Observed: dominant sequences change across runs |
| Emotional weighting | Strong selection pressure = deeper genetic imprint | Catastrophe survivors show larger genetic shifts |

---

## Signal Flow: How the Hippocampus Connects

### Real brain pathway
```
Sensory organs → Thalamus (routing)
    → Primary cortex (V1, A1, S1)  — raw features
    → Association cortex            — abstract features  
    → Entorhinal cortex            — spatial + context encoding
    → HIPPOCAMPUS                  — memory formation
    → Back to entorhinal → cortex  — memory recall influences perception
```

### DAERWEN AI system pathway
```
Sensors (camera, mic) → CNN / audio model (cortex equivalent)
    → Feature extraction (10-15 abstract values)
    → Thalamus module (routing + attention gating)
    → DAERWEN (hippocampus)
        Input:  sparse, theta-rhythmic feature spikes
        Process: population evolves under feature-driven selection
        Output: memory state vector (5-10 floats)
    → Decision module (basal ganglia equivalent)
    → Action output
    → Results feed back as new sensory input
```

---

## Input Specification (What DAERWEN Should Receive)

### NOT raw sensory data

WRONG: video frame (80×80 pixels = 6400 values)
RIGHT: pre-processed features (10-15 abstract floats)

### Input characteristics (matching real hippocampal input)

| Property | Real Hippocampus | DAERWEN Implementation |
|----------|-----------------|----------------------|
| **Format** | Spike trains (binary events) | Spike-encoded chemical pulses |
| **Rhythm** | Theta rhythm (6-10 Hz bursts) | Every N steps, not every step |
| **Sparsity** | 1-5% of neurons active | Only above-threshold features trigger spikes |
| **Content** | Spatial + object + context features | CNN-extracted feature vector |
| **Spatial mapping** | Different features → different brain regions | Different features → different world regions |

### Example input protocol
```python
THETA_PERIOD = 15  # steps between input bursts

if step % THETA_PERIOD == 0:
    features = cnn_extract(current_frame)  # [brightness, motion, edges, ...]
    for i, value in enumerate(features):
        if value > SPARSITY_THRESHOLD:
            region_x, region_y = FEATURE_REGIONS[i]
            spike_inject(system, region_x, region_y, intensity=value * SCALE)
```

---

## Output Specification (What DAERWEN Produces)

### Memory state vector (derived from population statistics)

These are NOT designed outputs — they are OBSERVATIONS of population state
that happen to carry memory information:

| Signal | Derivation | Meaning |
|--------|-----------|---------|
| `familiarity` | Cosine similarity of current gene distribution vs historical average | "I've seen this before" |
| `threat_memory` | Recent death rate relative to baseline | "Last time this happened, many died" |
| `spatial_memory` | Population center of mass direction | "Resources were THAT way" |
| `population_health` | Alive count / carrying capacity | "System is stressed / healthy" |
| `novelty` | Divergence of current gene distribution from recent history | "This is new / familiar" |
| `adaptation_rate` | Recent generation velocity | "System is actively evolving / stable" |

### Key principle
These outputs are MEASURED, not PROGRAMMED. The population dynamics
naturally produce these signals. We just read them.

---

## Memory Mechanism: Why Population Genetics = Hippocampal Memory

### Encoding (experiencing → storing)
```
Environmental pressure (e.g., toxin in north)
  → Particles in north die (selection)
  → Survivors have "avoid-north" genetic bias
  → Population gene distribution SHIFTS
  = Memory encoded in gene frequencies
```

### Storage (maintaining)
```
Gene frequencies persist across generations
  → As long as selection pressure is gone, memory slowly drifts (mutation)
  → But strong memories (from strong selection) persist longer
  = Emotional memories are stronger (matches real hippocampus)
```

### Recall (remembering)
```
Similar environmental pattern appears again
  → Population already has "prepared" gene distribution
  → Response is FASTER than naive population
  = "+16.3% over physics baseline" IS successful recall
```

### Forgetting
```
Mutation gradually drifts gene frequencies
  → Old memories fade unless reinforced
  → Forgetting rate 0.010 = very slow drift
  = Matches hippocampal memory decay
```

### False memory
```
Mutation changes stored gene patterns
  → Population "remembers" slightly wrong version of past
  → Response to familiar situation is close but not exact
  = Creative/reconstructive recall (matches human memory)
```

---

## What DAERWEN Does NOT Do (And Shouldn't)

- ❌ Process raw sensory data (that's cortex's job)
- ❌ Make decisions (that's basal ganglia's job)
- ❌ Regulate hormones (that's hypothalamus's job)
- ❌ Detect threats fast (that's amygdala's job)
- ❌ Plan or reason abstractly (that's prefrontal cortex's job)
- ❌ Understand language (that's language cortex's job)

DAERWEN does ONE thing: **form, maintain, and output memories based on
abstract sensory features, through population-level evolution.**

---

## Roadmap

### Phase 1: Define hippocampal I/O interface (current)
- Input: feature vector specification
- Output: memory state vector specification
- Document the architecture (this file)

### Phase 2: Implement correct input pipeline
- CNN feature extraction → sparse theta-rhythmic spikes
- Replace current "raw pixel dump" with proper abstracted input

### Phase 3: Implement output reading
- Extract memory state vector from population statistics
- Expose as API for downstream modules

### Phase 4: Integration with external modules
- Connect CNN (visual cortex) → DAERWEN
- Connect DAERWEN → decision module
- Full loop: sense → remember → decide → act → sense

---

## References

- O'Keefe & Moser (2014 Nobel Prize) — place cells and grid cells
- Hubel & Wiesel (1962) — visual cortex hierarchy (inspired CNN)
- Buzsáki (2006) — theta rhythm and hippocampal memory
- Marr (1971) — computational theory of hippocampus
  (pattern separation in DG, pattern completion in CA3)
- Squire (1992) — declarative memory and the hippocampus
