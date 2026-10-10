# Gated Register Transformer with Global Attention Pooling and Overwrite-driven Decay

**Architecture Proposal — Revision 1**

---

## 1. Introduction and Novelty

Standard Transformer-based language models [Vaswani et al., 2017] suffer from $O(N^2)$ KV-cache growth as context length increases. Segment-level recurrence approaches such as Transformer-XL [Dai et al., 2019] and the Recurrent Memory Transformer (RMT) [Bulatov et al., 2022] mitigate this across segment boundaries, yet still incur token-accumulation costs _within_ each segment.

This proposal introduces the **Gated Register Transformer (GRT)**, a memory-augmented architecture drawing on two lines of prior work: the external differentiable memory introduced by Neural Turing Machines [Graves et al., 2014], and the hardware-inspired **Control Unit / Register File** abstraction. The key idea is to discard the KV-cache of past tokens entirely and instead maintain a fixed-size external register bank $S \in \mathbb{R}^{M \times D}$ that is centrally managed by a global routing controller. This design fixes the inference compute at $O((N+M)^2)$ per segment—a constant with respect to sequence length—rather than growing quadratically with the full context.

A differentiable soft-gating mechanism yields the update Jacobian:

$$\frac{\partial S_{t+1}}{\partial S_t} = 1 - W_t$$

which simultaneously governs **information retention lifetime** and **gradient propagation depth**, providing a theoretically grounded and interpretable memory management system.

---

## 2. Architecture Hyperparameters

| Parameter                       | Recommended Value     | Description                                                          |
| ------------------------------- | --------------------- | -------------------------------------------------------------------- |
| $N$ (Segment Size)              | `128`                 | Number of input tokens processed per step                            |
| $M$ (Register Count)            | `32`                  | Number of fixed-size external register slots                         |
| $D$ (Model Dimension)           | `256`                 | Embedding and register vector dimensionality                         |
| ALU                             | `2-layer Transformer` | Central computation unit (`nhead=4`, `d_ff=1024`)                    |
| $M_{\text{drop}}$ (Reg Dropout) | `0.10` (10%)          | Per-step register reset probability to encourage distributed storage |

---

## 3. Module Design and Forward Pass

At each segment timestep $t$, data passes through the following four-stage pipeline.

### Stage 1: Global Router Unit

Assigning read/write addresses independently per register leads to redundancy. The Global Router instead extracts a rich contextual summary of the entire input segment and cross-attends it against all current register states to compute allocation flags jointly.

**Step 1 — Attention Pooling (Rich Context Extraction).**
A learnable query token $q_{\text{pool}} \in \mathbb{R}^{1 \times D}$ attends over the input $X_t \in \mathbb{R}^{N \times D}$, following the Pooling by Multihead Attention (PMA) paradigm of Lee et al. [2019]:

$$\alpha = \text{Softmax}\!\left(\frac{q_{\text{pool}}\,(X_t W_K)^\top}{\sqrt{D}}\right) \in \mathbb{R}^{1 \times N}$$

$$X_{\text{rich}} = \alpha\,(X_t W_V) \in \mathbb{R}^{1 \times D}$$

**Step 2 — Register-Aware Cross-Attention.**
Rather than collapsing register state into a single mean vector, the router cross-attends $X_{\text{rich}}$ against the full register bank $S_t$ to produce a register-aware context vector $C \in \mathbb{R}^{1 \times D}$:

$$C = \text{CrossAttn}(Q = X_{\text{rich}},\; K = S_t W_{K}^{reg},\; V = S_t W_{V}^{reg})$$

This preserves slot-level distinctions (e.g., which register holds subject information vs. tense information) that a mean-pooled summary would destroy.

**Step 3 — Global Allocation.**
The context vector $G = [X_{\text{rich}}\,;\,C] \in \mathbb{R}^{1 \times 2D}$ is passed through two master MLPs to produce per-register read and write flags:

$$R_t = \text{Sigmoid}(\text{MLP}_R(G)) \in \mathbb{R}^{M \times 1}$$

$$W_t = \text{Sigmoid}(\text{MLP}_W(G)) \in \mathbb{R}^{M \times 1}$$

The write-gate bias is initialized to $-2.0$ to ensure $W_t \approx 0.12$ at the start of training, preventing the register bank from being overwritten by noise during early optimization.

### Stage 2: Conditional Read Phase

Registers not addressed by the router ($R_{t,i} \approx 0$) are masked to zero vectors before being presented to the ALU, effectively blocking irrelevant memory from influencing computation:

$$\tilde{S}_t = R_t \odot S_t \in \mathbb{R}^{M \times D}$$

### Stage 3: Execution Phase (ALU)

The masked register bank is appended to the input tokens to form a fixed-length sequence:

$$\text{Input}_t = [X_t\,;\,\tilde{S}_t] \in \mathbb{R}^{(N+M) \times D}$$

$$\text{Output}_t = \text{Transformer}(\text{Input}_t) \in \mathbb{R}^{(N+M) \times D}$$

The output is split: the first $N$ vectors become the token output $Y_t$, and the last $M$ vectors become the write candidate $\Delta S_t \in \mathbb{R}^{M \times D}$.

The transformer ALU always sees exactly $N + M = 160$ tokens, regardless of how many segments have been processed.

### Stage 4: Overwrite-driven Write-back and Register Dropout

Forgetting arises naturally from the write operation itself—no explicit decay term is required. A 10% register dropout mask $M_{\text{drop}}$ is applied to the _write gate logits_ (before sigmoid) to stochastically block individual write paths during training. This prevents the model from concentrating information in a small subset of registers and encourages robust distributed storage across the full register bank.

$$S_{t+1} = \bigl((1 - W_t) \odot S_t + W_t \odot \Delta S_t\bigr)$$

where $W_t = \text{Sigmoid}(\text{MLP}_W(G) + \epsilon_{\text{drop}})$ and $\epsilon_{\text{drop},i} \sim -\infty$ with probability $p=0.10$ during training (logit masking), else $0$.

The register state $S_0$ is a **learnable parameter** initialized to zero, ensuring the ALU receives a clean and well-defined memory state at the start of each sequence.

---

## 4. Backward Pass Dynamics

### 4.1 Information Control Jacobian

The Jacobian of $S_{t+1}$ with respect to $S_t$ is:

$$\frac{\partial S_{t+1}}{\partial S_t} = 1 - W_t$$

- **Keep regime** ($W_t \to 0$): gradient flows back through time without attenuation, acting as a long-range memory highway.
- **Overwrite regime** ($W_t \to 1$): gradient is suppressed proportionally. At the hard limit ($W_t = 1$), the path is severed entirely—causally consistent with the fact that no information from $S_t$ survives in $S_{t+1}$.
- **Dropout escape routes**: registers forced to $W_t = 0$ by the dropout mask maintain full gradient connectivity, training the model to distribute information across multiple slots as a backup strategy.

This mechanism provides a structural brake against exploding gradients [Hochreiter & Schmidhuber, 1997; Cho et al., 2014] without requiring gradient clipping.

### 4.2 Write Gate Learning Signal (Predictive Coding)

The gradient flowing into the write gate weights carries the following signal:

$$\frac{\partial L}{\partial w_t} = \frac{\partial L}{\partial s_{t+1}} \cdot (\Delta s_t - s_t)$$

The magnitude $|\Delta s_t - s_t|$ represents the **prediction error** between the existing memory and the ALU's proposed update. The write gate weight is updated most aggressively when the incoming information differs substantially from what is already stored—an inductive bias analogous to surprise-driven memory consolidation as described in predictive coding theory [Rao & Ballard, 1999].

---

## 5. Expected Register Specialization

Once trained end-to-end with global routing, the 32-slot register bank is expected to exhibit functional differentiation into three ecological niches:

**Long-term Memory Registers** ($W \approx 0.01$): Gradient pathways remain intact over many timesteps; these slots stably encode global context such as the document topic or discourse structure.

**Working Memory Registers** ($W \approx 0.5$): Content evolves as a soft moving average, tracking local context and recent referents.

**Scratch-pad Registers** ($W \approx 0.9$): Updated aggressively each step; serve as temporary buffers for intra-segment token interactions.

---

## 6. Interpretability: Register Trace Log Analyzer

A primary advantage of GRT over prior memory-augmented architectures is that its gating tensors have shape $[B, M, 1]$—a dimensionality low enough to be fully logged, visualized, and analyzed without compression or approximation. The **Register Trace Log Analyzer (RTLA)** is a first-class component of the GRT research pipeline, built and validated _before_ main experiments begin. The analyzer serves both as a debugging instrument during early training and as an interpretability tool for the final model.

### 6.1 Logged Quantities

At every segment timestep $t$, the following tensors are written to a structured trace buffer:

| Signal                   | Shape    | Description                                     |
| ------------------------ | -------- | ----------------------------------------------- |
| $R_t$                    | $[T, M]$ | Read gate activations over all timesteps        |
| $W_t$                    | $[T, M]$ | Write gate activations over all timesteps       |
| $\|S_t\|_2$              | $[T, M]$ | Per-register L2 norm (memory load indicator)    |
| $\|\Delta S_t - S_t\|_2$ | $[T, M]$ | Per-register prediction error (surprise signal) |
| $\alpha_t$               | $[T, N]$ | Attention pooling weights over input tokens     |

Here $T$ denotes the total number of segments in the sequence. All quantities are logged in float32 during inference and in a detached no-grad pass during training checkpoints.

### 6.2 Visualization Panels

The RTLA renders four primary panels:

**Panel A — R/W Gate Heatmap (Logic Analyzer View).** A $2 \times T \times M$ grid displaying $R_t$ (top) and $W_t$ (bottom) as color-coded heatmaps over time. This is the oscilloscope view of the register file: each column is a register slot, each row is a timestep, and color intensity encodes gate activation. Stable columns with low $W$ and high $R$ identify long-term memory registers; rapidly oscillating columns identify scratch-pad registers.

**Panel B — Register Lifecycle Plot.** For each slot $i$, plots $W_{t,i}$ and $\|S_{t,i}\|_2$ on a shared time axis. The write gate trace shows _when_ the register was written; the norm trace shows _how much information_ it is carrying. Together these reveal write frequency, saturation, and decay behavior per slot.

**Panel C — Attention Pooling Token Map.** Plots $\alpha_t$ as a heatmap over input token positions across timesteps, revealing which tokens the router consistently attends to when constructing $X_{\text{rich}}$. Cross-referencing this with high-$W$ events in Panel A identifies which input tokens trigger memory updates.

**Panel D — Prediction Error Heatmap.** Plots $\|\Delta S_{t,i} - S_{t,i}\|_2$ across slots and time. High-error events correspond to timesteps where the ALU proposes a large memory revision. Correlating these with the input text reveals the _surprise threshold_ at which the model decides to overwrite a register.

### 6.3 Diagnostic Use Cases

**Dead Register Detection.** Slots where $\sum_t W_{t,i} < \epsilon$ across all training sequences are structurally unused. Panel A makes these immediately visible as uniformly dark columns, enabling $M$ to be reduced without ablation.

**Blackhole Collapse Detection.** If $W_t \to 1$ uniformly across all slots early in training, all gradient paths are severed and the register bank becomes informationally opaque. Panel A will show a fully saturated red field. The write-gate bias should be decreased below $-2.0$ until the heatmap shows a mixed pattern.

**Specialization Confirmation.** The primary empirical hypothesis of GRT—that registers self-organize into long-term, working, and scratch-pad tiers—is directly testable by clustering the $W$ column distributions from Panel A. Three well-separated clusters in the $\bar{W}_i$ distribution confirm the hypothesis without requiring probing classifiers.

**Routing Consistency Analysis.** Across multiple documents in the same domain, the RTLA can measure whether the same register slots are assigned similar functional roles. High cross-document consistency in $R_t$ and $W_t$ patterns indicates that the router has learned a stable addressing schema, not a sample-specific heuristic.

### 6.4 Implementation Notes

The RTLA is implemented as a standalone `RegisterAnalyzer` class that wraps the GRT model. During a trace pass, forward hooks on the router MLP output capture $R_t$ and $W_t$ without modifying the computation graph. The trace buffer is written to disk as a compressed `.npz` file (one per sequence), and the visualization panels are generated via `matplotlib` with a single `analyzer.plot(trace_path)` call. The analyzer is designed to be run on a single CPU after inference completes, adding no overhead to GPU training.

```
RegisterAnalyzer
├── attach_hooks(model)       # registers forward hooks on MLP_R, MLP_W, attn_pool
├── run_trace(input_ids)      # runs inference and collects trace buffer
├── save_trace(path)          # writes [T, M] arrays to .npz
└── plot(trace_path)
    ├── panel_a_rw_heatmap()
    ├── panel_b_lifecycle()
    ├── panel_c_token_map()
    └── panel_d_prediction_error()
```

---

## 7. Experiment Pipeline

The RTLA is built and validated as a prerequisite to all main experiments. The intended sequence is as follows:

**Phase 0 — Analyzer Validation (pre-experiment).** Train a minimal GRT instance (1 layer, $M=8$, $N=32$) on a synthetic long-range copy task for 1000 steps. Run the RTLA and confirm: (a) Panel A shows non-uniform gate distributions, (b) at least one slot shows consistently low $W$ (long-term candidate), (c) Panel D shows high prediction error on token positions where the copy target changes. If diagnostics pass, the analyzer is confirmed functional and the full experiment proceeds.

**Phase 1 — Architecture Sanity (toy tasks).** Evaluate on synthetic benchmarks designed to isolate specific memory behaviors: passkey retrieval (tests long-term register formation), sliding window entity tracking (tests working memory), and in-context arithmetic (tests scratch-pad usage). RTLA output for each task is examined to confirm that the hypothesized register tier activates for the corresponding task type.

**Phase 2 — Baseline Comparison.** Compare against RMT [Bulatov et al., 2022] and a vanilla Transformer [Vaswani et al., 2017] with fixed context window on long-document language modeling (e.g., PG-19 [Rae et al., 2020] or SCROLLS). Primary metrics: perplexity, throughput (tokens/sec), and peak memory. RTLA traces from Phase 1 are included in the paper as qualitative evidence of learned specialization.

**Phase 3 — Ablation Studies.** Ablate: (a) cross-attention router vs. mean-pooled router, (b) learnable $S_0$ vs. zero-initialized $S_0$, (c) logit-masking dropout vs. no dropout, (d) write-gate bias initialization value. Each ablation is accompanied by RTLA traces to attribute performance differences to specific routing or gating behaviors.

---

## 8. Computational Complexity

Per-segment inference cost is $O((N+M)^2 \cdot D)$, where $M$ is a fixed constant. This means inference cost scales as $O(N^2)$ in the segment size but is **independent of total sequence length**, unlike full-context attention which scales as $O(L^2)$ in the cumulative sequence length $L$. For long documents processed in chunks ($L \gg N$), GRT provides a constant per-segment budget regardless of how many prior segments have been seen.

---

## 9. Conclusion

The Gated Register Transformer integrates three complementary contributions:

1. **Register-aware cross-attention routing** for precise global address allocation that respects slot-level memory distinctions.
2. **Overwrite-driven forgetting** with a $(1-W)$ Jacobian that unifies gradient control and memory lifecycle management.
3. **Write-gate logit dropout** (10%) that enforces distributed storage and provides gradient escape routes without corrupting register state.

Together, these components yield a theoretically grounded, interpretable, and computationally efficient architecture for long-context language modeling. The fixed per-segment compute budget, learnable initial register state, and fully observable gating dynamics make GRT immediately amenable to prototype implementation and systematic empirical evaluation.

---

## References

**[Bulatov et al., 2022]** Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. _Recurrent Memory Transformer._ Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2207.06881.

**[Bulatov et al., 2023]** Aydar Bulatov, Yury Kuratov, and Mikhail S. Burtsev. _Scaling Transformer to 1M tokens and beyond with RMT._ arXiv preprint arXiv:2304.11062, 2023.

**[Cho et al., 2014]** Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. _Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation._ Proceedings of EMNLP, 2014. arXiv:1406.1078.

**[Dai et al., 2019]** Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. _Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context._ Proceedings of ACL, 2019. arXiv:1901.02860.

**[Graves et al., 2014]** Alex Graves, Greg Wayne, and Ivo Danihelka. _Neural Turing Machines._ arXiv preprint arXiv:1410.5401, 2014.

**[Hochreiter & Schmidhuber, 1997]** Sepp Hochreiter and Jürgen Schmidhuber. _Long Short-Term Memory._ Neural Computation, 9(8):1735–1780, 1997.

**[Lee et al., 2019]** Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. _Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks._ Proceedings of ICML, 2019.

**[Rae et al., 2020]** Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. _Compressive Transformers for Long-Range Sequence Modelling._ Proceedings of ICLR, 2020. arXiv:1911.05507. _(Introduces the PG-19 benchmark.)_

**[Rao & Ballard, 1999]** Rajesh P. N. Rao and Dana H. Ballard. _Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects._ Nature Neuroscience, 2(1):79–87, 1999.

**[Vaswani et al., 2017]** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. _Attention Is All You Need._ Advances in Neural Information Processing Systems (NeurIPS), 2017. arXiv:1706.03762.
