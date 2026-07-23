# Input Abstraction Research: How the Real Hippocampus Receives Data

> Compiled: 2026-04-18
> Sources: Wikipedia (Place cells), NCBI, Journal of Neuroscience, Springer

---

## The Real Signal Path

```
Sensory organs
  ↓
Thalamus (routing)
  ↓
Primary cortex (V1/A1/S1) — raw features
  ↓
Association cortex — abstract features
  ↓
Entorhinal cortex (EC) — spatial + context encoding
  │
  ├── Grid cells (hexagonal spatial pattern)
  ├── Head direction cells (which way facing)
  ├── Speed cells (how fast moving)
  └── Border cells (where are walls)
  ↓
HIPPOCAMPUS
  │
  ├── Dentate Gyrus — pattern SEPARATION (make similar inputs distinct)
  ├── CA3 — pattern COMPLETION (recall full from partial)
  └── CA1 — output back to EC and cortex
```

---

## What are Grid Cells?

Grid cells fire in a **hexagonal pattern** across space. Each grid cell has:
- A specific spatial frequency (fine grid vs coarse grid)
- A specific orientation (rotated slightly from others)
- A specific phase (offset)

Different grid cells cover the same space at different scales — like Fourier
basis functions at different frequencies. The hippocampus COMBINES these
to create place cells.

**Analogy for DAERWEN**: grid cells are like 2D Fourier components.
The hippocampus does a kind of "inverse Fourier transform" to create
a sparse spatial map from these periodic inputs.

---

## What are Place Cells?

A place cell fires when the animal is at ONE specific location.
- Firing rate: >100 Hz when in the "place field"
- Firing rate: <1 Hz everywhere else
- This is EXTREMELY sparse: at any location, only ~1-5% of place cells are active

"All hippocampal pyramidal cells appear to be place cells" (in rodents),
but at any moment, the vast majority are silent.

**Analogy for DAERWEN**: each particle (or group of particles) in a
specific 2D region could function as a "place cell" — active when
the sensory input pattern matches its genetic "preference."

---

## Encoding Format

The hippocampus uses **population sparse coding**:
- N total neurons (~1 million in rodent hippocampus)
- At any moment: ~10,000-50,000 active (1-5%)
- Active neurons COLLECTIVELY represent current location + context
- Different locations → different subsets active
- Same location in different context → DIFFERENT subsets (remapping)

**For DAERWEN input**: don't activate the entire chemical field.
Activate SPECIFIC REGIONS sparsely, based on the current feature vector.

---

## Remapping: Context Changes Everything

When the environment changes significantly (different room, different
color walls, different task), place cells REMAP:
- Global remapping: most cells change their preferred location
- Rate remapping: cells keep location but change firing rate

This means the SAME neurons can encode COMPLETELY DIFFERENT spaces
depending on context.

**For DAERWEN**: different "contexts" (different types of sensory input)
should activate DIFFERENT spatial regions of the 2D world, even if the
abstract features are similar. This prevents memory interference.

---

## Replay: Memory Consolidation

During rest/sleep, hippocampal place cells REPLAY experienced
sequences at 5-20x real speed. This:
- Strengthens the synaptic connections encoding the memory
- Transfers information from hippocampus → cortex (long-term storage)
- Compresses temporal experiences into rapid neural patterns

**For DAERWEN**: could be implemented as periodic "fast-forward"
runs where the system replays recent environmental patterns at
accelerated speed. This would strengthen genetic memory of
recent experiences.

---

## Implications for DAERWEN's Input Interface

### What we should NOT do
- ❌ Overwrite the entire chemical field with pixel values
- ❌ Inject continuous values at every grid point
- ❌ Feed raw sensory data (pixels, waveforms)
- ❌ Use the same input format for all contexts

### What we SHOULD do

1. **Pre-process through "cortex" (CNN)**
   - Raw video → ResNet → 512-dim feature vector → PCA → 10-15 dim
   - Each dimension = one abstract feature (brightness, edges, motion, etc.)

2. **Map features to sparse spatial activation**
   ```
   For each feature i with value v_i:
     - Map to a specific REGION of the 2D world (feature_i → region_i)
     - Activation probability = v_i * max_rate
     - If active: inject small chemical spike in region_i
     - If not: nothing
   
   Result: 10-15 features → 3-5 active regions per cycle
   ```

3. **Theta-rhythmic input (6-10 Hz equivalent)**
   - Don't input every step
   - Input in bursts every THETA_PERIOD steps
   - Between bursts: system processes internally (evolution, diffusion)

4. **Context-dependent mapping**
   - Different "contexts" (day vs night, visual vs audio) use DIFFERENT
     spatial mappings
   - This prevents cross-context interference
   - Context signal could be an additional input dimension

5. **Sparsity control**
   - At most 30% of the 2D world should receive input at any moment
   - The rest is "dark" — no input → population there is free to evolve
     based on internal dynamics and diffusion from active regions

### Concrete parameter suggestions

```
THETA_PERIOD = 15 steps        (every 15 steps = one input burst)
N_FEATURES = 12                (from CNN dimensionality reduction)
SPARSITY = 0.3                 (max 30% of world area activated per burst)
SPIKE_STRENGTH = 0.02          (small, needs accumulation for effect)
REGION_SIZE = 8x8 grid cells   (each feature activates an 8x8 patch)
N_ACTIVE_FEATURES = 3-5        (out of 12, only top-K activate)
```

---

## Open Questions

1. Should features always map to the SAME spatial regions, or should
   the mapping itself evolve? (Real brain: mostly fixed after critical period)

2. Should DAERWEN have a "sleep" phase where it replays recent
   experiences? (Matches biology but adds complexity)

3. How to handle TEMPORAL sequences? Grid cells encode position, but
   time cells (recently discovered) encode time. Should DAERWEN have
   time-encoding regions?

4. What is the "context signal"? How does DAERWEN know it's in a
   different situation vs the same situation with different details?
