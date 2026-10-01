# Upcycling a Dense Language Model into a Mixture of Experts

**ERA V5 · Session 14 assignment**

## Summary

**Objective.** Train a dense ("linear") transformer language model, convert it into a Mixture-of-Experts
(MoE) model part-way through training, and demonstrate that the converted model continues to train and
continues to reduce its loss.

**Approach.** A 22.2M-parameter dense model was trained for 4,000 steps on FineWeb. It was then converted
into a 71.8M-parameter MoE (29.3M active per token) by splitting each feed-forward layer into one shared
expert and eight routed experts, and trained for a further 4,000 steps. A dense control branch, started
from the same checkpoint and trained on identical data with an identical schedule, isolates the effect of
the conversion.

**Outcome.**

| | held-out loss at conversion | held-out loss after 4,000 more steps | throughput |
|---|---:|---:|---:|
| MoE (converted) | 4.064 | **3.744** | 172,684 tokens/s |
| Dense control | 4.055 | **3.796** | 327,984 tokens/s |

* The conversion preserved the model's function: the held-out loss changed by only +0.0085 at the switch.
* The MoE continued to train and reduce its loss for all 4,000 steps, and was still improving at the end.
* The MoE finished 0.052 nats below the dense control at equal tokens and steps.
* At equal wall-clock time the dense control was ahead, because the MoE ran at 53% of its throughput.
* The full session, including data preparation, completed in **31 minutes** on a Colab TPU v5e-1.

---

## Contents

| Path | Description |
|---|---|
| [`Session-14-assignment.ipynb`](Session-14-assignment.ipynb) | The notebook (JAX, Google Colab TPU v5e-1) |
| [`notebook_src.py`](notebook_src.py) | Source of truth for the notebook, one `# %%` block per cell |
| [`build_notebook.py`](build_notebook.py) | Builds the `.ipynb` from the source; `--flat` also writes a script for local testing |
| [`run-output/`](run-output/) | Everything the run produced: console log, ledgers, shard manifests, loss curves, summaries |
| [`figures/`](figures/) | Figures reproduced in this document |

---

## 1. Experimental design

```
             Phase 1: 4,000 steps                  Phase 2: 4,000 steps (identical data and schedule)
  Dense  ─────────────────────────► checkpoint ─┬─► convert to MoE, continue training   (treatment)
  22.2M parameters                               └─► continue as dense                   (control)
```

A continued-training curve on its own cannot show that the conversion helped, because the loss of any
model still in training will keep falling. The control branch addresses this. Both branches:

* start from the same Phase 1 checkpoint,
* read the same sequences in the same order (confirmed step by step by the data ledger), and
* follow the same learning-rate schedule.

The difference between them is therefore attributable to the conversion.

| | Dense model | MoE model |
|---|---|---|
| Architecture | 8 layers, width 384, 6 heads, pre-norm RMSNorm, learned positions, tied embeddings | Identical except for the feed-forward layers |
| Feed-forward layer | SwiGLU, 384 → 1,536 → 384 | 1 shared expert (768 wide) + 8 routed experts (768 wide), top-2 routing |
| Parameters | 22.2M | 71.8M total, 29.3M active per token |
| Sequence length / batch | 512 tokens / 64 sequences (32,768 tokens per step) | Same |
| Data | FineWeb `sample-10BT`, 131M tokens per phase | Same stream, continued |
| Hardware / precision | Colab TPU v5e-1; bf16 matrix multiplies, fp32 weights, optimizer state and residual stream | Same; router kept entirely in fp32 |

---

## 2. Method, step by step

The notebook runs top to bottom with *Run all*. Every stage caches its output in Google Drive, so a
re-run skips completed work and resumes interrupted work.

### Step 1 — Settings
All tunable values are collected in one cell: time budget, minimum and maximum steps per phase, model
dimensions, MoE configuration, optimizer and schedule, data parameters, checkpoint interval, and output
location. A reduced configuration (`S14_SMOKE=1`) runs the entire notebook on a laptop CPU in about a
minute. It was used for all pre-run testing.

### Step 2 — Environment checks
* **Hardware.** The notebook confirms that a TPU is attached and stops otherwise, rather than silently
  training on a CPU for hours.
* **Storage.** It mounts Google Drive. **If Drive does not mount, the notebook stops** with remediation
  steps, because without persistent storage neither caching nor resumption can work.
* **Setup.** It creates the device mesh and starts a timestamped console log, which is mirrored to
  `console.log` in Drive.

### Step 3 — Model definition
A single set of pure-JAX functions serves both models. A layer containing an `ffn` entry is dense; a
layer containing a `moe` entry is a Mixture-of-Experts layer. Attention, normalization and embeddings
are shared code, so the only architectural difference between the two models is the feed-forward
block. The optimizer (AdamW) is written out explicitly rather than taken from a library, and the
learning-rate schedule is computed from the step number inside the compiled training step.

### Step 4 — Verification gates
Five checks run in full fp32 precision before any significant TPU time is used, and the notebook
stops if any fails:

| Gate | What it verifies | Result on the TPU |
|---|---|---|
| 1 | The vectorised expert dispatch matches a token-by-token reference implementation | max difference 6.1e-5 |
| 2 | When an expert is full, exactly the right tokens are dropped | max difference 6.1e-5 |
| 3 | A dense model converted to an MoE produces the same logits as the original | max difference 8.9e-8 |
| 4 | After conversion, gradient reaches the router and every expert | passed |
| 5 | The dense and MoE training steps each compile once and are reused | passed |

### Step 5 — Throughput measurement and planning
Real training steps are timed on the attached TPU for both models before any long run begins. The
measured step times are used to choose the number of steps that fits the time budget:

> steps per phase = 0.85 × budget ÷ (2 × dense step time + MoE step time), capped at 4,000

A plan that falls below the minimum of 1,000 steps per phase is re-attempted at a smaller batch. If no
batch size fits, the notebook stops. The chosen plan is saved and reused by later sessions, so a resumed
session trains exactly what the first session started.

*Measured on the TPU v5e-1:* 100 ms per dense step, 190 ms per MoE step. bf16 was 1.18× faster than fp32.
The plan was 4,000 steps per phase at batch 64, which fitted within the 40-minute budget.

### Step 6 — Data preparation
* **Corpus.** FineWeb `sample-10BT` (English web text), streamed from Hugging Face. Documents shorter
  than 200 characters are dropped.
* **Tokenizer.** A byte-level BPE with 8,192 tokens, trained on the first 80 MB of text.
* **Packing.** Documents are concatenated in stream order with an end-of-text token between them,
  without padding.
* **Shards.** The token stream is cut into immutable **shards** of 32,768 sequences (16.8M tokens),
  giving 16 training shards and 1 validation shard.
* **Held-out data.** The validation shard is built from documents that appear after every training
  document, so no validation text is ever trained on.
* **Manifests.** Each shard receives a manifest recording:
  * SHA-256 hashes of its tokens and its text
  * provenance: dataset, stream rows, Common Crawl dumps
  * cleaning steps, deduplication and contamination status, language score
  * tokenizer hash, packing policy, parent shard, and a resume pointer
* **Document index.** A gzipped index records every document's FineWeb identifier, URL and token range.
* **Integrity check.** All shards are verified against their manifests before training.

Downloading runs in a background thread while tokenization proceeds. The build took 3 minutes and
resumes at the first incomplete shard if interrupted.

### Step 7 — Data loading and device placement
* **Order.** Training shards are read in a fixed, seeded order. Within a shard, sequences are read in a
  fixed permutation seeded by the shard number.
* **No boundary-spanning steps.** A step never spans two shards: 512 steps per shard at batch 64.
* **Placement.** The actual placement of batch rows on devices is read from the sharded arrays rather
  than assumed. On a v5e-1 all 64 rows are placed on `TPU_0`.
* **Schedule table.** Before training, the notebook prints a table of which steps read which shard on
  which device, and records it in `ledger/run_manifest.json`.

### Step 8 — Training loop
One function runs every phase. It:
* **Trains and records.** Trains one step per batch, and writes one ledger line per step every 25 steps.
* **Evaluates.** Measures held-out loss 16 times per run.
* **Reports progress.** Prints a progress line at least once a minute.
* **Checkpoints.** Writes a checkpoint every 5 minutes, and keeps the previous one as a backup.
* **Saves on interruption.** Writes an emergency checkpoint at the exact step if the run is stopped or
  an error is raised.
* **Pauses on overrun.** If a run exceeds 1.5× its planned time, it saves and pauses, so it cannot
  overrun silently.

On restart, the loop loads the newest readable checkpoint, rewrites the step ledger to match it exactly,
and continues. Tests confirmed that a resumed run reproduces an uninterrupted run exactly (Section 5).

### Step 9 — Phase 1: dense training
The dense model trains for 4,000 steps on 131M tokens:
* **Learning rate.** A 100-step linear warm-up, then held constant at 1e-3.
* **Optimizer.** AdamW with β = (0.9, 0.95), weight decay 0.1 on matrices, gradient clipping at 1.0.

Held-out loss fell from 9.10 to 4.055.

### Step 10 — Conversion to MoE
Each layer's SwiGLU feed-forward block (1,536 hidden neurons) is converted as follows.

1. **Shared expert.** The first 768 neurons are copied unchanged into an always-active shared expert.
2. **Routed experts.** The remaining 768 neurons are copied into each of 8 routed experts. Gaussian noise
   at 1% of each weight matrix's standard deviation is added so that the copies are not exact clones.
3. **Router.** A new router is created (fp32, small random initialisation). It scores the experts with a
   sigmoid, selects the top 2, and renormalises their weights to sum to 1.
4. **Function preservation.** Because the routed copies are identical and the selected weights sum to 1,
   *shared + weighted routed* reproduces the original feed-forward output exactly (Gate 3).
5. **Optimizer continuity.** AdamW's first and second moments are split and copied in the same way as the
   weights, and the step counter continues. The optimizer therefore does not restart cold, and the
   warm-up is not re-triggered.
6. **Router pre-balancing.** Before the first MoE step, the load-balancing rule is run offline on a real
   batch, layer by layer, so that no expert's capacity overflows at the switch.
7. **Continuity check.** The converted model is evaluated on the held-out shard before any training. The
   notebook stops if its loss differs from the dense model's by more than 0.1.
   *Observed: 4.0554 (dense) vs 4.0639 (MoE), a difference of +0.0085.*

### Step 11 — Phase 2: MoE and dense control
Both branches train for 4,000 steps on the same 131M tokens. The learning rate stays at 1e-3 until the
last 1,200 steps, then follows a cosine decay to 1e-4. The MoE additionally uses:

* **Probabilistic top-k for the first 100 steps.** Experts are sampled in proportion to their router
  scores, so every copy receives gradient before the router settles.
* **Auxiliary-loss-free load balancing** (DeepSeek-V3):
  * Each expert has a bias that only affects which experts are selected, not their weights.
  * After each step, overloaded experts' biases are lowered and underloaded experts' raised: by 0.01 for
    the first 1,000 steps, then 0.001.
  * No balancing term is added to the loss.
* **Fixed expert capacity.** Each expert can take at most 320 token-choices per group of 1,024 tokens
  (1.25× its fair share). A token whose chosen expert is full keeps the shared expert's output.

### Step 12 — Results and ledger read-back
The final cell does the following:
* **Reporting.** Compiles the results table, generates the figures, and writes `results.md`.
* **Ledger summary.** Prints a shard-by-shard summary for each run.
* **Step lookup.** Performs a lookup that traces a chosen step to its shard, device, individual
  sequences and source web pages, and replays the batch from the shard to verify its hash.
* **Samples.** Produces text samples from each model.
* **Packaging.** Writes all outputs except checkpoints and token shards to `session14_outputs.zip`.

---

## 3. Design decisions

| Area | Decision | Rationale | Alternatives considered |
|---|---|---|---|
| Framework | Pure JAX (no Flax or Optax) | JAX is the TPU's native framework and is pre-installed on Colab TPU runtimes. Avoiding other libraries removes version-compatibility risk. | PyTorch/XLA: requires a separate install matched to the PyTorch version, and is prone to recompilation with dynamic shapes. |
| Hardware and precision | TPU v5e-1, bf16 matrix multiplies, fp32 master weights, router in fp32 | bf16 is native on TPUs (measured 1.18× faster than fp32). The router is kept in fp32 because routing decisions are sensitive to precision; the lecture noted that Switch Transformer diverged with a bf16 router. | fp16, which the TPU does not run natively. |
| Dataset | FineWeb `sample-10BT` | The same corpus as Session 13, which makes the two assignments roughly comparable and reflects realistic web-scale text. | TinyStories: used in the first version of the notebook and replaced on request. |
| Tokenizer | Byte-level BPE, 8,192 tokens, trained on the first 80 MB | Matches the Session 13 recipe. A small vocabulary keeps the embedding and output layers from dominating a 22M-parameter model. | A pretrained 50K-token GPT-2 tokenizer. |
| Model size | 22.2M dense / 29.3M active MoE | Small enough to complete three runs within the time budget on one chip, and large enough to learn non-trivial language from 262M tokens. | — |
| Step count and batch | 4,000 steps per phase at 32,768 tokens per step; at least 1,000 steps per phase guaranteed | Session 13 showed that step count dominates at a fixed token budget: its large-batch run had only 102 optimizer steps. The planner reduces the batch rather than the step count if time is short. | Maximum batch that fits in memory. |
| Learning-rate schedule | Warm-up (100 steps) → constant → cosine decay over the last 30% of Phase 2 | A constant rate at the conversion point avoids the confound of a decaying schedule. Warm-up is counted in steps because warm-up counted in tokens collapsed to 3 steps in Session 13. Both Phase 2 branches share the decay, so the comparison stays fair. | Cosine decay across the full run. |
| MoE shape | 1 shared + 8 routed experts, each 768 wide, top-2 routing | This shape allows an exact, function-preserving conversion (below). The shared expert follows the course's Lightning LM design. | Finer-grained experts (for example 32 × 192): lower active share, but the conversion cannot preserve the function. |
| Conversion method | Shared expert = first half of the neurons; routed experts = copies of the second half + 1% noise | Preserves the model's output exactly at the switch, so any loss movement afterwards reflects training rather than a disruption. | Partition (non-preserving); drop-upcycling with re-initialised neurons (non-preserving; reported to help over longer runs). |
| Router scoring | Sigmoid, top-2, weights renormalised to sum to 1 | Sigmoid is the current choice in DeepSeek-V3 and similar models, and works well with bias-based balancing. Renormalisation is required for exact function preservation. | Softmax (as in Qwen3). |
| Load balancing | Auxiliary-loss-free per-expert bias | The method the course has adopted. It keeps the language-modelling loss untouched, avoiding the gradient conflict that made auxiliary losses unstable. | Auxiliary balancing loss; token dropping with capacity limits alone. |
| Expert dispatch | Fixed capacity (1.25× fair share, per 1,024-token group), with gather/scatter into per-expert buffers | TPU programs require static shapes. Gather/scatter scales with the number of tokens, whereas one-hot dispatch matrices scale with tokens × capacity. Tokens are queued in order, so a token's chance of being dropped depends only on earlier tokens. | Computing every expert for every token (8× the compute); ragged grouped matrix multiplies (uncertain TPU support at the time). |
| Router initialisation | Small random weights plus pre-calibrated biases | Random rows give distinct routing from the first step. Pre-calibrated biases prevent capacity overflow at the switch. | Zero-initialised router: all experts tie, and the tie-break sends every token to the first experts. |
| Early routing | Probabilistic top-k for the first 100 steps | The lecture's fix for identical clones that never receive gradient. | Hard top-k from the first step. |
| Evaluation | Held-out shard of 65K tokens; 16 evaluations per run | Cheap enough to run frequently. Built from documents after all training text, so it is guaranteed unseen. | Larger validation set: lower noise, at more cost per evaluation. |
| Persistence | Google Drive required; checkpoints every 5 minutes with backup; emergency save; stage caching | Session 13 retrained completed runs after its Drive folder was lost. A 5-minute interval bounds the work a hard disconnect can lose. | Checkpoint every 25% of a run (the first version). |

---

## 4. Changes introduced

### 4.1 Lessons carried over from Session 13

| Problem in Session 13 | Cause | Change in this notebook |
|---|---|---|
| The run took over 3 hours and exhausted the compute allocation | Emulated bf16 on a T4 ran 4.8× slower; no throughput check preceded the long runs | Throughput is measured on the target device and the runs are sized to a budget before training. bf16 vs fp32 is measured directly. Each run pauses at 1.5× its planned time. |
| Completed runs were retrained | The Drive folder was missing in a new session | Every stage is cached. The plan is saved and reused. Checkpoints are frequent, with a backup. A missing Drive mount now stops the notebook. |
| The large-batch run collapsed | Warm-up was defined in tokens and shrank to 3 steps | Warm-up is defined in steps, with a floor of 100. The conversion preserves optimizer state. |
| One run had only 102 optimizer steps | Large batch at a fixed token budget | Minimum step count enforced; the batch is reduced before the step count. |
| (New risk on TPU) | JAX recompiles whenever a shape changes, and MoE routing naturally produces variable shapes | All shapes are static. The step number is passed as data. Gate 5 verifies single compilation. |

### 4.2 Iterations during development

| Version | Change | Reason |
|---|---|---|
| 1 | JAX/TPU notebook on TinyStories with a dense control, planner, gates and stage caching | Initial implementation |
| 2 | Switched to FineWeb `sample-10BT` with Session 13's tokenizer recipe; sequence length 512; new Drive folder; settings fingerprint so folders from other configurations are refused | Requested: train on FineWeb |
| 3 | Exact resumption (5-minute checkpoints, backup checkpoint, emergency save, pause instead of truncation, shard-level data resume); timestamped console log mirrored to Drive; Session 6-style data ledgers | Requested: resume from the point of interruption, console messages, ledgers |
| 4 | The notebook stops if Google Drive fails to mount (opt-out: `ALLOW_LOCAL_OUTPUT = True`) | During the actual run, Drive failed to mount and the notebook only warned before falling back to local disk. This is now a standing rule for future notebooks. |

### 4.3 Defects found and fixed during testing

| Defect | Effect had it shipped | Fix |
|---|---|---|
| The converted MoE shared some weight buffers with the dense model, and the training step frees its input buffers | Starting the MoE run would have deleted the dense model's embeddings and attention weights, which the control branch still needed | Each run takes its own copy of the state before training |
| The background download thread did not stop after data preparation | A thread would continue downloading into memory for the rest of the session | The thread is signalled to stop when enough tokens have been collected |
| When resuming a partial data build, the download thread could block process exit mid-read | Local scripts hung on exit (Colab kernels were unaffected) | The notebook waits for the thread to finish its current read |

---

## 5. Verification

Before the TPU run, the notebook was tested end to end on a CPU with a reduced configuration, and at an
intermediate scale with the full MoE settings (8 experts, 1,024-token groups):

* **Interruption and resumption.** Each scenario was compared against an uninterrupted reference run,
  and in every case the per-step ledgers were **byte-identical**:
  * the run stopped manually (emergency checkpoint);
  * the process killed outright (periodic checkpoint);
  * the newest checkpoint corrupted (backup checkpoint);
  * data shards deleted and rebuilt. Rebuilt shards had SHA-256 hashes identical to the originals.
* **Intermediate-scale behaviour on FineWeb.**
  * Conversion changed the loss by +0.0003 to +0.0017.
  * At most 0.9% of expert choices overflowed capacity.
  * All experts stayed within 0.92–1.07× of their fair share.
  * The ledger added no measurable throughput cost.
* **Multi-device.** A four-device simulated mesh ran successfully.

The TPU run itself required no resumption. All five gates passed, the shard integrity checks passed,
and the ledger replay check reproduced the recorded batch hash.

---

## 6. Results

| Run | Parameters | Active | Steps | Tokens | Held-out loss (start) | Held-out loss (end) | Tokens/s | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 1: dense | 22.2M | 22.2M | 4,000 | 131M | 9.101 | **4.055** | 327,978 | 6.7 min |
| Phase 2: MoE | 71.8M | 29.3M | 4,000 | 131M | 4.064 | **3.744** | 172,684 | 12.6 min |
| Phase 2: dense control | 22.2M | 22.2M | 4,000 | 131M | 4.055 | **3.796** | 327,984 | 6.7 min |

Loss is cross-entropy in nats per token on the held-out shard.

![Loss curves](figures/loss_curves.png)

### 6.1 The conversion preserved the model
Held-out loss was 4.0554 before conversion and 4.0639 immediately after, before any MoE training: a
difference of +0.0085. The residual difference is most plausibly due to the 2.0% of expert choices that
exceeded capacity at the switch. Those tokens lost that expert's share of their feed-forward output.
The 1% noise on the copies is too small to matter.

### 6.2 The MoE continued to train and to reduce its loss
Held-out loss fell from 4.064 to 3.744 over 4,000 steps, including a further 0.206 between steps 6,000
and 8,000. Mean training loss fell on every shard read after step 4,096. The MoE was still improving
when the run ended.

### 6.3 The MoE outperformed the dense control at equal tokens

| Global step | 4,000 | 4,500 | 5,000 | 6,000 | 7,000 | 8,000 |
|---|---:|---:|---:|---:|---:|---:|
| MoE | 4.064 | 4.041 | 4.006 | 3.950 | 3.898 | 3.744 |
| Dense control | 4.055 | 4.034 | 4.012 | 3.976 | 3.939 | 3.796 |
| Control − MoE | −0.009 | −0.007 | +0.006 | +0.027 | +0.041 | **+0.052** |

The MoE trailed slightly for roughly the first 800 steps (by at most 0.011), while the copied experts
diverged from one another. It overtook the control near step 4,900. The gap then widened at every
evaluation, reaching 0.052 at the end (perplexity 42.3 vs 44.5). The per-shard training losses show the
same pattern, moving from −0.008 on the first full Phase 2 shard to +0.051 on the last.

### 6.4 At equal wall-clock time the dense control was ahead
The MoE ran at 53% of the dense model's throughput (190 ms vs 100 ms per step). It required 11.9 minutes
of training to reach the loss the control reached in 6.7 minutes. Two factors account for the difference:

* **More computation per token.** The shared expert, two routed experts and the 25% spare capacity amount
  to about 1.45× the dense model's arithmetic per token.
* **Routing overhead.** Moving tokens into and out of expert buffers accounts for approximately a further
  30%.

At this scale (8 experts, 41% of parameters active, a single chip) the MoE delivers a better model per
token but not per second. The efficiency gains described in the lecture arise with many more, narrower
experts and active shares of 2–10%.

![Loss against training time](figures/loss_vs_time.png)

### 6.5 Expert load remained balanced; overflow depended on balancing scope
* **Load.** Over the final 20% of training, every expert carried between 1.00× and 1.01× its fair share
  of the batch. No expert fell idle (0 of 64). The busiest expert in any layer rarely exceeded 1.2×;
  layer 3 showed the largest fluctuations.
* **Overflow.** The proportion of expert choices exceeding capacity rose from 0.8% to 1.4% during the
  second half of training, although batch-level balance remained near-perfect.

The bias balances load across the full 32,768-token batch, while capacity is enforced per 1,024-token
group. As experts specialise, tokens from the same document increasingly select the same experts, so an
individual group can overflow while the batch as a whole is balanced. This is a small-scale instance of
the *balancing scope* issue discussed in the lecture.

![Expert load](figures/experts.png)

### 6.6 Text samples
Both phase-2 models produce fluent-looking but loosely connected text, and both repeat phrases. This is
typical of a 22M-parameter model trained on 262M tokens. The samples confirm that the MoE is a
functioning language model; they do not distinguish the two models in quality. See
[`run-output/samples.json`](run-output/samples.json).

---

## 7. Operational record

| Stage | Time |
|---|---:|
| Verification gates | 3 s |
| Throughput measurement and planning | 26 s |
| Data download, tokenization and sharding (262M tokens) | 3.0 min |
| Phase 1: dense | 6.7 min (planned 6.7) |
| Conversion and continuity check | 10 s |
| Phase 2: MoE | 12.6 min (planned 12.7) |
| Phase 2: dense control | 6.7 min (planned 6.7) |
| **Total session** | **31 min** |

Periodic checkpoints were written as scheduled (270 MB for the dense model; 865 MB for the MoE, in 2–8
seconds each). No interruption occurred. At the start of the session Google Drive failed to mount and
was mounted manually; the notebook now stops in that situation instead of continuing without persistent
storage.

### Data ledger
The run records what was trained on, on which device, what the model returned, and whether each step
can be replayed. This follows the data-ledger design from Session 6.

* **Shards.** 16 training shards (327,370 documents) and 1 validation shard, each with a manifest and a
  document index. All were verified by SHA-256 before training.
* **Per-step ledgers.** 12,000 lines in total. Each line records:
  * the shard read, the device (`TPU_0`, rows 0–63) and the 64 sequence identifiers;
  * the loss of each sequence, the learning rate and the gradient norm;
  * a hash of the batch;
  * for the MoE, the overflow rate and the busiest expert.
* **Checkpoint and event ledgers.** The step, SHA-256 and data position of every checkpoint, and every
  run start and finish.

*Example.* At global step 6,000 both phase-2 runs read shard `train_00009`, the same 64 sequences on
`TPU_0`. Batch loss was 3.711 for the MoE and 3.736 for the control. In both runs the hardest sequence
came from a business-directory profile page (loss 4.36 and 4.39) and the easiest from a physics textbook
page on the Rayleigh criterion (2.89 and 2.88). Re-reading step 6,000 from the shard reproduced the
recorded batch hash.

---

## 8. Limitations

* **Equal tokens, not equal compute.** The MoE uses 1.32× the active parameters of the control. Part of
  its advantage may therefore come from additional computation rather than from sparsity. A dense model
  of about 29M parameters trained on the same data would separate the two effects.
* **Single run.** Each configuration was trained once, with one seed. In Session 13 the dense baseline
  reproduced to within 0.005 across sessions, so a 0.052 gap is well outside that range. The consistent
  widening of the gap over successive evaluations is nonetheless stronger evidence than the final value
  alone.
* **Small validation set.** At 65K tokens, individual evaluations carry noise; conclusions rest on trends
  across evaluations.
* **Loss comparability with Session 13 is approximate.** The corpus and tokenizer recipe match, but the
  tokenizer instance, validation data and model differ.

## 9. Future work

1. Add a compute-matched dense baseline (about 29M parameters) to separate parameter count from sparsity.
2. Balance load per group, or raise the capacity factor to 1.5, to address the late rise in overflow.
3. Test more, narrower experts (for example 32 experts of width 192) to move towards the active-parameter
   ratios where MoE improves throughput.
4. Compare drop-upcycling, which re-initialises half of each copy's neurons, to test whether more
   diverse starting experts shorten the initial catch-up period.

---

## 10. Reproduction

1. Open [`Session-14-assignment.ipynb`](Session-14-assignment.ipynb) in Google Colab (*File → Upload notebook*).
2. Select a TPU runtime: *Runtime → Change runtime type → v5e-1 TPU*. TPU runtimes require a paid Colab plan.
3. Choose *Runtime → Run all* and approve the Google Drive authorisation prompt. About 4 GB of free Drive
   space is required.
4. Outputs are written to `MyDrive/session14_moe_fineweb/`. If the session is interrupted, choose
   *Run all* again; the notebook resumes from the point of interruption.

To rebuild the notebook from source, run `python3 build_notebook.py`. To test the complete notebook
locally on a CPU (requires `jax`, `datasets`, `tokenizers`, `nbformat` and `matplotlib`), run:

```bash
python3 build_notebook.py --flat flat.py && S14_SMOKE=1 python3 flat.py
```
