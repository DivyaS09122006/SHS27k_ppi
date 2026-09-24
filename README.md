# Multi-Type PPI Prediction on SHS27k: GIN vs. dpdGAT

**A brief replication and comparison study against TRGH-PPI, benchmarked on CPU**

---

## 1. Objective

Predict the *type* of interaction between pairs of proteins — not simply whether they
interact, but which of 7 biologically distinct mechanisms is involved: reaction, binding,
catalysis, activation, inhibition, post-translational modification (ptmod), and expression.
A given pair can exhibit more than one mechanism simultaneously, making this a multi-label
classification problem over graph-structured data.

The benchmark dataset is **SHS27k**, used by three prior papers in this space — E²PPI,
TRGH-PPI, and MEGAE — which allows direct comparison against published results. All work was
done on a CPU-only machine (no usable GPU).

---

## 2. Data Preparation

Two raw files formed the starting point:

- **1,553 proteins**, each with an amino acid sequence.
- **54,811 interaction records**, each listing a protein pair and one interaction mechanism.

**Sequence embeddings.** Every protein sequence was passed through ESM-2
(`esm2_t12_35M_UR50D`), a pretrained protein language model, producing a 480-dimensional
fingerprint per protein (1,553 × 480 tensor). This was cached once and reused throughout.

**Collapsing to labeled pairs.** The raw interaction file lists each unordered pair twice (once
per direction), and a pair can appear multiple times if it exhibits multiple interaction types.
Sorting each pair's protein IDs and using that as a dictionary key collapsed the 54,811 rows
into **6,660 unique unordered pairs**, each carrying a 7-dimensional multi-label vector.
**58.4%** of pairs carry more than one label — consistent with the multi-label proportion
TRGH-PPI itself reports for this dataset, which gave confidence the preprocessing was correct.

---

## 3. Reference Implementation Study

Before building anything, HIGH-PPI's own code (`gnn_data.py`, `model_train.py`, `utils.py`) was
read to understand established practice in this line of work. Notable findings:

- Its train/valid split is only 80/20, not the 60/20/20 convention used by the papers being
  compared against.
- The split is **unseeded** and therefore not reproducible.
- Critically, **message passing at training time uses only training-set edges**, never the full
  graph — the setup is non-transductive, meaning the model cannot see validation/test structure
  during training.

These findings directly shaped the design of the pipeline built here.

---

## 4. Splits: Random, BFS, DFS

Following the convention established by Lv et al. (used by GNN-PPI, HIGH-PPI, and TRGH-PPI
alike), the 6,660 pairs were split three different ways, each 60/20/20, seeded and reproducible
(unlike the HIGH-PPI reference code):

- **Random** — a plain shuffle. Easiest; allows near-duplicate or closely related proteins to
  appear on both sides of the split.
- **BFS / DFS** — a connected region of the graph is grown from a randomly chosen low-degree
  starting node until it covers 60% of edges (train), then the next 20% (valid); the remainder
  is test. This forces the model to generalize to a topologically distinct, previously unseen
  region of the interaction graph — a substantially harder and more realistic test.

Five seeds (0–4) were used per split type throughout all experiments.

---

## 5. Baseline Model: GIN

A 3-layer Graph Isomorphism Network (GIN), 256-dim hidden size, operating on the ESM-2 node
features. Message passing was restricted to training-set edges only, matching the
non-transductive protocol identified above. Each pair's 7 labels were predicted via
`FC(x_i * x_j)` — a two-layer head on the elementwise product of the two proteins' learned
representations. Trained with multi-label binary cross-entropy.

**Baseline results (5 seeds, micro-F1):**

| Split  | Mean  | Std   |
|--------|-------|-------|
| Random | 66.4% | ±1.5% |
| BFS    | 55.2% | ±8.5% |
| DFS    | 53.4% | ±8.7% |

The ordering (Random > BFS > DFS) and the growing variance moving from Random to DFS both match
the pattern reported across this line of literature — random splits are easier and more stable;
BFS/DFS are harder and more sensitive to exactly where the graph walk begins.

---

## 6. Reading TRGH-PPI

The actual TRGH-PPI paper (Wang et al., *IEEE TCBB*, 2025) was obtained and read in full. Its
architecture is considerably larger in scope than initially assumed:

- **Structural input, not just sequence.** Each protein is modeled as a residue-level graph,
  with edges from a real 3D contact map (residues within 10 Å, derived from PDB structures,
  supplemented by AlphaFold predictions where no crystal structure exists) and 7 RDKit-derived
  physicochemical features per residue.
- **Feature extraction ("TransGCN" block, ×3):** alternates a Transformer layer (global,
  sequence-order attention) and a GCN layer (local, structure-based), joined by a gated residual
  connection, then pooled via self-attention graph pooling into one protein-level vector.
- **Prediction module ("dpdGAT" block, ×3):** a graph attention variant using dot-product-based
  dynamic attention (closer to GATv2 than to standard GAT), operating on the protein-level PPI
  graph. Final prediction: `FC(x_i ⊙ x_j)` — a single linear layer on the elementwise product,
  no hidden layer.
- **Hyperparameters (from the paper):** 256-dim TransGCN output, 1024-dim dpdGAT output, 2
  attention heads, Adam (lr=0.001, weight_decay=1e-4), 600 epochs, 3 seeds averaged, same
  Random/BFS/DFS 60/20/20 protocol.

### Scope decision

The structural half of this pipeline — sourcing PDB/AlphaFold structures for 1,553 proteins,
building per-residue contact graphs, extracting RDKit features, and training a residue-level
Transformer+GCN (attention over hundreds of residues per protein, for every protein, every
epoch) — constitutes a second full data-engineering project and was judged out of scope for a
CPU-only weekend.

What **was** feasible: implementing TRGH-PPI's prediction-module architecture (dpdGAT) on top of
the existing ESM-2 protein-level embeddings, as a controlled comparison against the GIN baseline
already built. This isolates one specific question — *does the attention mechanism itself add
value, independent of the structural data pipeline?* — that can be answered without the
structural pipeline.

---

## 7. dpdGAT Implementation

dpdGAT was implemented directly from the paper's equations (12–14), with one interpretive
decision flagged explicitly:

> **Note on Eq. 14.** As printed, the paper's node-update equation is an *unweighted* GIN-style
> sum — it does not reference the attention weights (γ_ij) defined immediately prior in Eq. 13.
> Since the paper explicitly reports dpdGAT outperforming its own GIN ablation, an unweighted
> aggregation would make the two mathematically indistinguishable, which would contradict that
> result. This was treated as a likely typographical omission, and the aggregation was
> implemented as attention-weighted — the interpretation consistent with the rest of the paper's
> narrative, though not something that could be independently verified against the paper's own
> (unavailable) source code at the equation level alone.

The prediction head follows Eq. 15 exactly: a single linear layer on `x_i * x_j`, with no hidden
layer — a real architectural asymmetry against the GIN baseline's two-layer head, since the
comparison holds each model's head as its own paper/implementation specifies rather than
artificially equalizing them.

Hyperparameters followed the paper directly: 1024-dim hidden size, 3 layers, 2 attention heads,
Adam (lr=0.001, weight_decay=1e-4). Due to CPU time constraints, **epoch count was reduced from
the paper's 600 to 200**, and **hyperparameter grid search was not performed** — the paper's
reported optimal values were used as-is rather than re-derived.

---

## 8. Experimental Setup

All 30 combinations (2 models × 3 splits × 5 seeds) were run under an identical protocol: same
data, same splits, same training/evaluation procedure, 200 epochs, best-validation-checkpoint
selection, micro-F1 on the held-out test set.

**Infrastructure note:** this comparison was run three times across three different
environments over the course of the weekend — WSL2 Ubuntu with parallel multiprocessing (halted
repeatedly by WSL connection instability under sustained multi-core load), a Google Colab
sequential fallback (prepared but not ultimately used), and finally a native Windows Python
environment with no threading at all, which completed the full run without interruption in
approximately 1 hour 48 minutes.

---

## 9. Results

### GIN vs. dpdGAT (this work, ESM-2 sequence features only)

| Model  | Split  | Mean micro-F1 | Std    |
|--------|--------|---------------|--------|
| GIN    | Random | 66.4%         | ±1.5%  |
| GIN    | BFS    | 55.2%         | ±8.5%  |
| GIN    | DFS    | 53.4%         | ±8.7%  |
| dpdGAT | Random | **74.1%**     | ±1.5%  |
| dpdGAT | BFS    | 46.0%         | ±14.5% |
| dpdGAT | DFS    | 55.7%         | ±10.0% |

### TRGH-PPI's own ablation table (Table IV, structural features, SHS27k)

| Method                          | DFS   | BFS   |
|----------------------------------|-------|-------|
| TRGH-PPI (full)                  | 72.83 | 71.05 |
| w/o Transformer                  | 70.44 | 70.53 |
| w/o dpdGAT (plain GAT)            | 71.42 | 70.62 |
| w/ GIN (their structural feats)   | 71.92 | 70.45 |
| w/o Transformer or dpdGAT         | 68.94 | 67.43 |

---

## 10. Discussion

**dpdGAT vs. GIN, on identical inputs, is not a clean win for dpdGAT.** On Random split,
dpdGAT's higher capacity clearly helps (74.1% vs. 66.4%). On BFS, it is *worse* than the simpler
GIN baseline (46.0% vs. 55.2%) and nearly three times as variable across seeds (±14.5 vs. ±8.5).
On DFS, the two are close, with dpdGAT slightly ahead but both noisy. A plausible explanation:
dpdGAT's larger hidden size (1024 vs. 256) and lack of dropout — both specified by the paper —
give it more capacity to overfit specifically on the harder, generalization-testing splits,
where the training region is topologically distinct from the test region.

**Feature richness appears to matter more than architecture.** Every variant in TRGH-PPI's own
ablation table — including its weakest configuration, plain GAT with no Transformer feature
extraction at all — scores at or above 67% on both BFS and DFS. The best result obtained here on
those same harder splits (55.7%, dpdGAT/DFS) sits 12–17 points below that floor, despite using
TRGH-PPI's own prediction-module architecture verbatim. Given that this implementation's ESM-2
embeddings and TRGH-PPI's residue-level structural graphs are the primary point of divergence,
this gap is best read as evidence that a substantial share of TRGH-PPI's reported performance
comes from the richness of its structural input, not from the dpdGAT mechanism alone.

**Caveats on the comparison's fairness:**
- Not apples-to-apples with the paper's own numbers: those use structural features; this work
  uses sequence-only ESM-2 embeddings throughout.
- 200 epochs vs. the paper's 600, and no hyperparameter search vs. their 5-fold grid search.
- 5 seeds here vs. 3 in the paper (a minor point in the opposite direction — arguably a slightly
  more robust seed average).
- The Eq. 14 interpretation (attention-weighted aggregation) is a best-effort reading, not a
  verified match to the authors' original implementation.

None of these caveats change the qualitative story — they bound its precision, not its
direction.

---

## 11. Conclusion

Over the course of the weekend: SHS27k was cleaned and reshaped from 54,811 raw interaction
records into 6,660 labeled pairs; three split protocols (Random, BFS, DFS) were built, seeded,
and validated against known dataset statistics; a GIN baseline was implemented and benchmarked;
TRGH-PPI's actual published methodology was read and its dpdGAT prediction module
reconstructed and benchmarked under an identical protocol; and the resulting comparison points
toward structural protein features, rather than prediction-head architecture, as the more likely
source of TRGH-PPI's reported advantage over simpler baselines — a testable hypothesis for
future work with access to GPU compute and a protein-structure data pipeline.
