# %% [markdown]
# # Dense → Mixture of Experts: grow a model mid-training
#
# **ERA V5 — Session 14 assignment.** Train a dense ("linear") language model, convert it into a
# Mixture-of-Experts model part-way through training, and show that the MoE keeps training and keeps
# reducing the loss.
#
# Built for a **Colab TPU** (v5e-1 or v6e-1), written in **JAX** (the TPU's native framework, already
# installed on Colab's TPU runtimes). Run it top to bottom with *Runtime → Run all*.
#
# ### The experiment
#
# ```
#              phase 1                          phase 2  (same data, same learning-rate schedule)
#   dense  ────────────────────────► checkpoint ─┬─► convert to MoE, keep training   (the assignment)
#   ~22M params, S steps                          └─► dense, keep training           (the control)
# ```
#
# The **control** answers the obvious objection: "the loss would have gone down anyway". Both branches
# start from the same checkpoint, see the same tokens in the same order, and follow the same learning-rate
# schedule, so the gap between them is what the conversion bought.
#
# | | dense | MoE (after conversion) |
# |---|---|---|
# | layers × width | 8 × 384, 6 heads, sequence 512 | same |
# | feed-forward per layer | one SwiGLU, 384 → 1536 → 384 | 1 shared expert (768 wide) + 8 routed experts (768 wide each), top-2 |
# | parameters | ~22M | ~72M total, ~29M active per token |
# | data | FineWeb `sample-10BT` (the Session 13 corpus), 8,192-token BPE trained here | same stream, continued |
#
# ### How the dense FFN becomes experts (the course's recipe, section 8 of the study guide)
#
# * **Shared expert = the first half of the dense FFN's neurons** (the Lightning LM move: "take the first
#   1024 of 2048 as the shared expert").
# * **Each routed expert = a copy of the other half**, with 1% noise so the copies are not exact clones
#   ("copy" upcycling). The router picks 2 of the 8 and its weights are renormalized to sum to 1, so at
#   the moment of conversion *shared + weighted routed* adds up to exactly the dense FFN. The model's
#   output does not jump; section 9 checks this on held-out data.
# * **Router:** sigmoid scores, top-2, kept in fp32. **Balancing:** auxiliary-loss-free per-expert bias
#   (DeepSeek-V3, the method the course will use), no extra loss term.
# * **Probabilistic top-k for the first 100 MoE steps**, so every clone receives gradient before the router
#   settles (the fix for the dead clones in Lightning LM's 20 → 460 growth).

# %% [markdown]
# ## What went wrong last time, and what this notebook does about it
#
# | Session 13 | cause | what this notebook does |
# |---|---|---|
# | **Took over 3 hours and used up the compute** | The T4 has no bf16 hardware, `torch.cuda.is_bf16_supported()` said yes anyway, and every matmul ran emulated, 4.8× slower. Nothing measured the speed before four long runs were started. | **Section 5 times real training steps on your TPU before anything long starts**, projects the total, and sizes the runs to fit `TIME_BUDGET_MIN`. If the plan cannot fit, it stops with a message instead of starting. On a TPU **bf16 is the native type**; section 5 shows bf16 vs fp32 steps side by side. Every run also **pauses at 1.5× its planned time** (progress saved; run again to continue), so a surprise cannot turn into hours. |
# | **Finished runs trained again** | The Drive folder was gone in the new session, so nothing was recognized as done. | Every stage caches to `MyDrive/session14_moe_fineweb/` and is skipped when its file is there. The step plan is saved and reused, so a resumed session trains exactly the same thing. **A stopped run resumes at the step where it stopped** (details below). |
# | **Run D collapsed** | Warm-up was defined in tokens and shrank to 3 steps at the big batch. | Warm-up is counted in **steps**, never fewer than 100. The conversion does not reset the optimizer and pre-balances the router, so phase 2 does not start with a shock. |
# | **Run D had only 102 optimizer steps** | Big batch at a fixed token budget. | The planner guarantees at least `MIN_STEPS` (1,000) steps per phase. If the budget cannot afford that, it **lowers the batch, not the step count**. |
# | *(new, TPU-specific)* | JAX recompiles whenever a shape changes; MoE routing naturally produces changing shapes. | All shapes are static (fixed expert capacity), the step number is passed as data, and section 5 checks that each training function compiled exactly once. |

# %% [markdown]
# ## Stopping, resuming, and where everything is written
#
# **If anything stops a run, run the notebook again (*Run all*) and it continues from where it stopped.**
#
# | what stopped it | what is saved | where the next session starts |
# |---|---|---|
# | you press *Stop*, or an error is raised inside training | an emergency checkpoint at the exact step | that step |
# | the runtime disconnects or the machine is reclaimed | the last periodic checkpoint (every 5 minutes) | at most 5 minutes of training earlier |
# | a run passes 1.5× its planned time | a checkpoint, then the notebook pauses with a message | that step (you decide whether to continue) |
# | the data download is cut off | every finished 16.8M-token shard | the first unfinished shard |
#
# The previous checkpoint is kept as a backup, in case Drive had not finished uploading the latest one.
# Resuming is exact: the data position, learning rate, random numbers and router biases all come back, so
# a resumed run produces the same numbers as one that never stopped.
#
# **Console.** Every message below carries a timestamp and is also appended to `console.log` in the Drive
# folder. If the browser tab disconnects while training, the TPU keeps going; open `console.log` in Drive
# (or run `!tail -n 30 "/content/drive/MyDrive/session14_moe_fineweb/console.log"` in a new cell once
# training stops) to see what happened.
#
# **Ledgers** (the data-ledger design from Session 6: *what did we feed, on which device, what came back,
# and can we go back?*):
#
# ```
# MyDrive/session14_moe_fineweb/
# ├── console.log                         every message, timestamped
# ├── plan.json                           the step plan, fixed on the first run
# ├── data/
# │   ├── tokenizer.json
# │   ├── manifest.json                   all shards: ids, token counts, SHA-256
# │   └── shards/
# │       ├── train_00000.bin             immutable token shard (32,768 sequences of 512 tokens)
# │       ├── train_00000.manifest.json   provenance, cleaning, hashes, tokenizer, packing, resume pointer
# │       └── train_00000.docs.jsonl.gz   every document in the shard: FineWeb id, URL, token range
# ├── ledger/
# │   ├── run_manifest.json               chips, mesh, shard order, which steps read which shard on which chip
# │   ├── events.jsonl                    sessions, shards written, runs started / resumed / paused / finished
# │   ├── <run>.steps.jsonl               one line per optimizer step: shard, chip, sample ids, per-sample loss,
# │   │                                   learning rate, gradient norm, batch hash (and expert load for the MoE)
# │   ├── <run>.checkpoints.jsonl         every checkpoint: step, file SHA-256, loader position, RNG, reason
# │   └── <run>.shards.json               per shard: which steps, which chip, tokens, mean loss
# └── runs/<run>/                         checkpoints (ckpt.pkl + ckpt_prev.pkl while running), final weights,
#                                         loss curves, summary
# ```
#
# Section 11 reads the ledger back: a shard-by-shard table for each run, and a lookup that takes any
# step, shows what it trained on (down to the web pages), and replays the batch from the shard to prove
# it is the same.

# %% [markdown]
# ## 1. Settings
#
# **How long it takes.** The plan keeps the three training runs inside `TIME_BUDGET_MIN` = 40 minutes
# (less on a fast chip, because each phase is capped at 4,000 steps). The first session adds 5–15 minutes
# to download and tokenize about 1 GB of FineWeb text (it prints progress), and a few minutes of compiling.
# Expect roughly **45–60 minutes on a v5e-1** the first time; a resumed session skips the data. Because every
# run pauses at 1.5× its planned time, even a badly mis-measured session cannot go past about 80 minutes.
#
# **Before you start:** choose *Runtime → Change runtime type → v5e-1 TPU* (or v6e-1). When *Run all*
# reaches section 2, approve the Google Drive prompt; the notebook waits for it. It needs about 4 GB
# free in Drive while it runs (two checkpoints are kept per run and deleted when the run finishes; Drive
# may keep replaced checkpoints in its Trash, so empty the Trash afterwards if space is tight).
#
# **If anything stops:** reconnect a TPU and *Run all* again (see the table above). Keep
# `MyDrive/session14_moe_fineweb/` until you have submitted.
#
# You should not need to change any setting below.

# %%
import os
SMOKE = os.environ.get("S14_SMOKE") == "1"   # a tiny CPU run, used to test this notebook end to end

TIME_BUDGET_MIN = 40        # training minutes for all three runs together (dense + MoE + dense control)
MIN_STEPS = 1000            # per phase: never plan fewer optimizer steps than this (Session 13's run D had 102)
MAX_STEPS = 4000            # per phase: a fast TPU finishes early instead of training for longer
BATCH_CHOICES = (64, 32, 16)    # sequences per step, tried largest first; the planner takes the first that fits
GUARD = 1.5                 # a run that takes more than GUARD x its planned time pauses (saved; run again to continue)

SEQ_LEN, VOCAB = 512, 8192  # tokens per sequence, tokenizer size (both as in Session 13)
VAL_SEQS = 128              # held-out sequences for validation loss (65K tokens)
TOKENIZER_MB = 80           # the tokenizer is trained on the first 80 MB of text (as in Session 13)
MIN_DOC_CHARS = 200         # skip very short documents (as in Session 13)

D_MODEL, N_LAYER, N_HEAD, FFN_HIDDEN = 384, 8, 6, 1536    # the dense model

N_EXPERTS, TOP_K = 8, 2             # routed experts per layer, experts per token
CAPACITY_FACTOR, GROUP_SIZE = 1.25, 1024   # expert slots = 1.25 x fair share, per group of 1024 tokens
EXPERT_NOISE = 0.01                 # noise on the expert copies, as a fraction of each weight matrix's std
SAMPLED_STEPS = 100                 # probabilistic top-k for the first MoE steps
GAMMA_HI, GAMMA_LO, GAMMA_SWITCH_FRAC = 0.01, 0.001, 0.25  # bias step: 0.01 for the first 25% of phase 2, then 0.001

PEAK_LR, WARMUP_MIN_STEPS, WARMUP_FRAC = 1e-3, 100, 0.02  # warm-up = max(100 steps, 2% of phase 1)
DECAY_FRAC, FINAL_LR_FRAC = 0.3, 0.1   # cosine decay to 10% over the last 30% of phase 2
WEIGHT_DECAY, GRAD_CLIP = 0.1, 1.0

SEED = 0
DRIVE_DIR = "/content/drive/MyDrive/session14_moe_fineweb"
ALLOW_LOCAL_OUTPUT = False  # True = run without Google Drive (nothing survives a disconnect; not recommended)
REPLAN = False              # True = measure the TPU again and make a new plan (changes what gets trained)
COMPARE_DTYPES = True       # time fp32 vs bf16 steps once (about a minute)
LOG_EVERY = 25              # steps between loss read-outs (and ledger writes)
PRINT_EVERY_SEC = 60        # a progress line at least this often while training
CKPT_EVERY_MIN = 5          # periodic checkpoint interval: the most training a hard disconnect can cost
SHARD_SEQS = 32768          # sequences per data shard (16.8M tokens); a step never spans two shards

if SMOKE:
    TIME_BUDGET_MIN, MIN_STEPS, MAX_STEPS, BATCH_CHOICES = 3, 30, 40, (8,)
    SEQ_LEN, VOCAB, VAL_SEQS, TOKENIZER_MB = 64, 512, 16, 2
    D_MODEL, N_LAYER, N_HEAD, FFN_HIDDEN = 64, 2, 2, 256
    N_EXPERTS, GROUP_SIZE, SAMPLED_STEPS, WARMUP_MIN_STEPS, LOG_EVERY = 4, 128, 5, 5, 5
    PRINT_EVERY_SEC, CKPT_EVERY_MIN, SHARD_SEQS = 2, float(os.environ.get("S14_CKPT_MIN", 0.02)), 64
    DRIVE_DIR = os.environ.get("S14_OUT", "s14_smoke_out")

# %% [markdown]
# ## 2. Setup
#
# Checks that a TPU is attached (and refuses to run on anything else, so a wrong runtime cannot quietly
# take hours), mounts Google Drive for the cache, sets up the device mesh, and starts the console log.
#
# **If Google Drive does not mount, the notebook stops here** with the steps to fix it. Without Drive,
# a disconnect would lose everything and nothing could resume, so it refuses to start training.

# %%
import importlib.util, subprocess, sys
for pkg in ("datasets", "tokenizers"):           # present on Colab; installed only if missing
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import dataclasses, functools, hashlib, json, math, pickle, shutil, time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import matplotlib
import matplotlib.pyplot as plt

T_START = time.time()
_early = []                                  # messages from before the Drive folder exists

def log(*parts):
    """Print a timestamped line immediately (flush) and append it to console.log in the Drive folder."""
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    if "CONSOLE_LOG" in globals():
        with open(CONSOLE_LOG, "a") as f:
            f.write(line + "\n")
    else:
        _early.append(line)

BACKEND, DEVICE_KIND = jax.default_backend(), jax.devices()[0].device_kind
log(f"JAX {jax.__version__} | backend: {BACKEND} | {jax.device_count()} x {DEVICE_KIND}")
if BACKEND != "tpu" and not SMOKE:
    raise RuntimeError(
        "No TPU attached. Runtime → Change runtime type → choose a TPU (v5e-1 or v6e-1), then Run all.\n"
        "This notebook refuses to train on a CPU or GPU because it is sized for a TPU and would run for hours.")

DRIVE_HELP = """Google Drive is not mounted, so nothing would survive a disconnect and resuming would not work.
Nothing has been trained. To fix it:
  1. Add a cell at the top and run:
         from google.colab import drive
         drive.mount("/content/drive", force_remount=True)
     In the pop-up, choose the same Google account you use for Colab and tick "Select all" permissions.
  2. If it still fails: use a browser window signed in to only that account (e.g. incognito), allow
     pop-ups and cookies for colab.research.google.com and accounts.google.com, or use the Mount Drive
     button in the Files sidebar.
  3. When it prints "Mounted at /content/drive", choose Runtime -> Run all again.
(To run without Drive anyway, set ALLOW_LOCAL_OUTPUT = True in section 1.)"""

try:
    from google.colab import drive
    ON_COLAB = True
except ImportError:                  # not on Colab (the smoke test): a local folder
    ON_COLAB = False
OUT = DRIVE_DIR
if ON_COLAB:
    try:
        drive.mount("/content/drive")
        mount_error = None
    except Exception as e:
        mount_error = f"{type(e).__name__}: {e}"
    if mount_error or not os.path.isdir("/content/drive/MyDrive"):
        if not ALLOW_LOCAL_OUTPUT:
            raise RuntimeError(f"Google Drive did not mount ({mount_error or 'MyDrive not found'}).\n\n{DRIVE_HELP}")
        OUT = "/content/session14_moe_fineweb"
        log(f"WARNING: running without Google Drive (ALLOW_LOCAL_OUTPUT = True). Results go to {OUT} and are lost "
            "if the runtime disconnects; download session14_outputs.zip at the end.")
os.makedirs(OUT, exist_ok=True)
CONSOLE_LOG = os.path.join(OUT, "console.log")
LEDGER = os.path.join(OUT, "ledger")
os.makedirs(LEDGER, exist_ok=True)
with open(CONSOLE_LOG, "a") as f:
    f.write("\n" + "=" * 100 + f"\nsession started {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "\n".join(_early) + "\n")

def event(kind, **fields):
    """One line in ledger/events.jsonl: sessions, shards, run starts, resumes, pauses, checkpoints."""
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, **fields}
    with open(os.path.join(LEDGER, "events.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")

def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

event("session_start", jax=jax.__version__, backend=BACKEND, devices=[str(d) for d in jax.devices()])
LOCAL = "/content/s14_local" if os.path.isdir("/content") else os.path.join(OUT, "_local")
os.makedirs(LOCAL, exist_ok=True)
log("outputs:", os.path.abspath(OUT))

# Compiled programs are cached on local disk: a kernel restart on the same machine skips recompiling.
jax.config.update("jax_compilation_cache_dir", os.path.join(LOCAL, "jax_cache"))

MESH = Mesh(np.array(jax.devices()), ("data",))
DATA = NamedSharding(MESH, P("data"))      # batches are split across chips (one chip on v5e-1 / v6e-1)
REPL = NamedSharding(MESH, P())            # weights and optimizer state are replicated
put = lambda tree: jax.device_put(tree, REPL)

def save_pickle(obj, path):                # write to a temp file, then rename: a disconnect mid-write
    tmp = path + ".tmp"                    # never leaves a half-written checkpoint behind
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(obj), f, protocol=4)
    os.replace(tmp, path)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)

TIMINGS_PATH = os.path.join(OUT, "timings.json")
TIMINGS = json.load(open(TIMINGS_PATH)) if os.path.exists(TIMINGS_PATH) else {}
def record_time(stage, seconds):
    TIMINGS[stage] = round(TIMINGS.get(stage, 0) + seconds, 1)
    save_json(TIMINGS, TIMINGS_PATH)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

# %% [markdown]
# ## 3. The model
#
# One set of functions serves both models. A layer whose feed-forward entry is `ffn` is dense; a layer
# whose entry is `moe` is a Mixture-of-Experts layer. Everything else (attention, norms, embeddings) is
# shared code, which is what "the router replaces only the FFN" looks like in practice.
#
# **Precision.** Weights and optimizer state are fp32; matmuls run in bf16 (the TPU's native type); the
# residual stream, norms, softmax, loss and the **router** stay in fp32.
#
# **Why expert capacity.** A TPU program has fixed shapes, but "how many tokens chose expert 3" changes
# every step. So each expert gets a fixed number of slots per group of 1,024 tokens:
# `capacity = 1.25 × top_k × 1024 / n_experts`. A token that finds its expert full keeps the shared
# expert's output and skips that routed expert. The bias balancing keeps this rare, and the fraction
# dropped is logged. Tokens are served in order, so whether a token is dropped depends only on tokens
# before it (nothing leaks from the future).

# %%
F32 = jnp.float32

@dataclasses.dataclass(frozen=True)
class Cfg:
    vocab: int; d: int; n_layer: int; n_head: int; ffn: int; seq: int
    n_experts: int = 8; top_k: int = 2; capacity_factor: float = 1.25; group: int = 1024
    sampled_steps: int = 100; gamma_hi: float = 0.01; gamma_lo: float = 0.001
    dtype: str = "bfloat16"; wd: float = 0.1; clip: float = 1.0; seed: int = 0

    @property
    def cdt(self):
        return jnp.dtype(self.dtype)

    @property
    def capacity(self):                       # expert slots per group, rounded up to a multiple of 8
        c = math.ceil(self.capacity_factor * self.top_k * self.group / self.n_experts)
        return -(-c // 8) * 8

CFG = Cfg(VOCAB, D_MODEL, N_LAYER, N_HEAD, FFN_HIDDEN, SEQ_LEN, N_EXPERTS, TOP_K, CAPACITY_FACTOR,
          GROUP_SIZE, SAMPLED_STEPS, GAMMA_HI, GAMMA_LO, "bfloat16", WEIGHT_DECAY, GRAD_CLIP, SEED)


def init_dense(cfg, key):
    keys = iter(jax.random.split(key, 3 + 5 * cfg.n_layer))
    normal = lambda shape, std=0.02: std * jax.random.normal(next(keys), shape, F32)
    out_std = 0.02 / math.sqrt(2 * cfg.n_layer)          # GPT-2 scaling for projections into the residual
    layers = [{"ln1": jnp.ones(cfg.d), "wqkv": normal((cfg.d, 3 * cfg.d)), "wo": normal((cfg.d, cfg.d), out_std),
               "ln2": jnp.ones(cfg.d),
               "ffn": {"w_gate": normal((cfg.d, cfg.ffn)), "w_up": normal((cfg.d, cfg.ffn)),
                       "w_down": normal((cfg.ffn, cfg.d), out_std)}}
              for _ in range(cfg.n_layer)]
    return {"tok_emb": normal((cfg.vocab, cfg.d)), "pos_emb": normal((cfg.seq, cfg.d), 0.01),
            "layers": layers, "ln_f": jnp.ones(cfg.d)}


def rmsnorm(x, w):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w


def attention(h, p, cfg):
    B, T, d = h.shape
    H, Dh = cfg.n_head, d // cfg.n_head
    qkv = (h.astype(cfg.cdt) @ p["wqkv"].astype(cfg.cdt)).reshape(B, T, 3, H, Dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    att = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=F32) / math.sqrt(Dh)
    causal = jnp.tril(jnp.ones((T, T), bool))
    att = jax.nn.softmax(jnp.where(causal, att, -1e30), axis=-1).astype(cfg.cdt)
    o = jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(B, T, d)
    return (o @ p["wo"].astype(cfg.cdt)).astype(F32)


def swiglu(x, p, cdt):                        # a dense FFN, or the shared expert
    x = x.astype(cdt)
    return (jax.nn.silu(x @ p["w_gate"].astype(cdt)) * (x @ p["w_up"].astype(cdt))) @ p["w_down"].astype(cdt)


def route(x, p, bias, cfg, key, sample):
    """Router: which top_k experts each token uses, and with what weights.  x: [groups, tokens, d]."""
    scores = jax.nn.sigmoid(x.astype(F32) @ p["router"].astype(F32))     # fp32, [G, t, E]
    pick = scores + bias                         # the balancing bias steers the choice ...
    gumbel = jnp.log(jnp.maximum(pick, 1e-6)) + jax.random.gumbel(key, pick.shape)
    pick = jnp.where(sample, gumbel, pick)       # probabilistic top-k: sample experts in proportion to score
    _, idx = jax.lax.top_k(jax.lax.stop_gradient(pick), cfg.top_k)
    w = jnp.take_along_axis(scores, idx, axis=-1)                          # ... but the weights ignore it
    return idx, w / jnp.sum(w, axis=-1, keepdims=True)                     # top-k weights sum to 1


def dispatch_group(x, idx, w, experts, cfg):
    """Send one group's tokens to their experts and bring the results back.  x: [t, d], idx/w: [t, k]."""
    t, d = x.shape
    E, C, k = cfg.n_experts, cfg.capacity, cfg.top_k
    flat_e = idx.reshape(-1)                                   # expert of each (token, choice), token order
    onehot = jax.nn.one_hot(flat_e, E, dtype=jnp.int32)
    pos = jnp.sum((jnp.cumsum(onehot, axis=0) - 1) * onehot, axis=-1)   # place in that expert's queue
    keep = pos < C                                             # queue full -> this choice is dropped
    slot = jnp.where(keep, pos, C)                             # C is out of range: writes/reads are skipped
    tok = jnp.repeat(jnp.arange(t), k)
    slot_tok = jnp.full((E, C), t, jnp.int32).at[flat_e, slot].set(tok, mode="drop")  # token in each slot
    xpad = jnp.concatenate([x.astype(cfg.cdt), jnp.zeros((1, d), cfg.cdt)])           # row t = empty slot
    buf = xpad[slot_tok]                                       # [E, C, d]: each expert's tokens
    cdt = cfg.cdt
    g = jnp.einsum("ecd,edh->ech", buf, experts["w_gate"].astype(cdt))
    u = jnp.einsum("ecd,edh->ech", buf, experts["w_up"].astype(cdt))
    out = jnp.einsum("ech,ehd->ecd", jax.nn.silu(g) * u, experts["w_down"].astype(cdt))
    y = out.at[flat_e, slot].get(mode="fill", fill_value=0).astype(F32)       # [t*k, d]
    y = y * (w.reshape(-1) * keep)[:, None]
    return y.reshape(t, k, d).sum(axis=1), keep, onehot.sum(axis=0)


def moe_ffn(h, p, bias, cfg, key, sample):
    B, T, d = h.shape
    assert (B * T) % cfg.group == 0, f"batch x seq ({B * T}) must be a multiple of the group size ({cfg.group})"
    x = h.reshape(B * T // cfg.group, cfg.group, d)
    idx, w = route(x, p, bias, cfg, key, sample)
    routed, keep, counts = jax.vmap(dispatch_group, in_axes=(0, 0, 0, None, None))(x, idx, w, p["experts"], cfg)
    y = swiglu(h, p["shared"], cfg.cdt).astype(F32) + routed.reshape(B, T, d)
    return y, {"counts": counts.sum(axis=0), "kept": keep.mean()}


def forward(params, tokens, cfg, bias=None, key=None, sample=False):
    B, T = tokens.shape
    key = jax.random.PRNGKey(0) if key is None else key
    x = params["tok_emb"][tokens] + params["pos_emb"][:T]            # fp32 residual stream
    stats = []
    for i, layer in enumerate(params["layers"]):
        x = x + attention(rmsnorm(x, layer["ln1"]), layer, cfg)
        h = rmsnorm(x, layer["ln2"])
        if "moe" in layer:
            y, s = moe_ffn(h, layer["moe"], bias[i], cfg, jax.random.fold_in(key, i), sample)
            stats.append(s)
        else:
            y = swiglu(h, layer["ffn"], cfg.cdt).astype(F32)
        x = x + y
    x = rmsnorm(x, params["ln_f"])
    logits = jnp.einsum("btd,vd->btv", x.astype(cfg.cdt), params["tok_emb"].astype(cfg.cdt),
                        preferred_element_type=F32)                    # tied embeddings
    return logits, stats


def cross_entropy(logits, targets):
    lse = jax.nn.logsumexp(logits, axis=-1)
    return jnp.mean(lse - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0])


# ---- optimizer: AdamW written out (the Session 11 assignment), schedule counted in steps ----

def init_opt(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, params), "t": jnp.array(0.0, F32)}


def adamw(params, grads, opt, lr, cfg, b1=0.9, b2=0.95, eps=1e-8):
    t = opt["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    def update(p, m, v):
        u = (m / c1) / (jnp.sqrt(v / c2) + eps)
        return p - lr * (u + cfg.wd * p if p.ndim >= 2 else u)       # decay matrices only
    return jax.tree.map(update, params, m, v), {"m": m, "v": v, "t": t}


def lr_at(step, s):
    """Warm-up (in steps) -> constant -> cosine decay to FINAL_LR_FRAC at the end of phase 2."""
    step = step.astype(F32)
    warm = jnp.minimum(1.0, (step + 1) / s["warmup"])
    prog = jnp.clip((step - s["decay_start"]) / (s["total"] - s["decay_start"]), 0.0, 1.0)
    return s["peak_lr"] * warm * (s["final_frac"] + (1 - s["final_frac"]) * 0.5 * (1 + jnp.cos(jnp.pi * prog)))


def make_state(params, sched, opt=None, bias=None, step=0):
    state = {"params": params, "opt": init_opt(params) if opt is None else opt,
             "step": jnp.array(step, jnp.int32), "sched": {k: jnp.array(v, F32) for k, v in sched.items()}}
    if bias is not None:
        state["bias"] = jnp.asarray(bias, F32)
    return state


def train_step(state, batch, cfg):
    tokens, targets = batch[:, :-1], batch[:, 1:]
    step, sched = state["step"], state["sched"]
    moe = "bias" in state
    moe_step = step - sched["moe_start"].astype(jnp.int32)
    sample = moe_step < cfg.sampled_steps
    key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), step)

    def loss_fn(params):
        logits, stats = forward(params, tokens, cfg, state.get("bias"), key, sample)
        per_token = jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, targets[..., None], -1)[..., 0]
        per_sample = per_token.mean(axis=1)                  # one number per sequence, for the ledger
        return per_sample.mean(), (stats, per_sample)

    (loss, (stats, per_sample)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state["params"])
    gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, cfg.clip / (gnorm + 1e-6)), grads)
    lr = lr_at(step, sched)
    params, opt = adamw(state["params"], grads, state["opt"], lr, cfg)
    new = dict(state, params=params, opt=opt, step=step + 1)
    metrics = {"loss": loss, "gnorm": gnorm, "lr": lr, "sample_loss": per_sample}
    if moe:
        counts = jnp.stack([s["counts"] for s in stats]).astype(F32)          # [layers, experts]
        load = counts / counts.sum(axis=-1, keepdims=True)                     # share of tokens per expert
        gamma = jnp.where(sample, 0.0, jnp.where(moe_step < sched["gamma_switch"], cfg.gamma_hi, cfg.gamma_lo))
        new["bias"] = state["bias"] + gamma * jnp.sign(1.0 / cfg.n_experts - load)   # loss-free balancing
        metrics.update(load=load, kept=jnp.mean(jnp.stack([s["kept"] for s in stats])), bias=new["bias"])
    return new, metrics


def eval_step(params, bias, batch, cfg):
    logits, _ = forward(params, batch[:, :-1], cfg, bias)
    return cross_entropy(logits, batch[:, 1:]) * batch[:, 1:].size


TRAIN_STEP = jax.jit(train_step, static_argnames="cfg", donate_argnums=0)
EVAL_STEP = jax.jit(eval_step, static_argnames="cfg")


# ---- the conversion ----

def to_moe(dense_state, cfg, key, noise=EXPERT_NOISE):
    """Dense state -> MoE state. Shared expert = first half of the FFN's neurons; each routed expert = a copy
    of the second half (+ noise); router new. Adam moments are carried over the same way; the step count
    continues, so the optimizer is not reset."""
    half = cfg.ffn // 2
    E = cfg.n_experts
    def halves(ffn):
        first = {"w_gate": ffn["w_gate"][:, :half], "w_up": ffn["w_up"][:, :half], "w_down": ffn["w_down"][:half]}
        second = {"w_gate": ffn["w_gate"][:, half:], "w_up": ffn["w_up"][:, half:], "w_down": ffn["w_down"][half:]}
        return first, second
    tile = lambda a: jnp.broadcast_to(a[None], (E,) + a.shape)

    def convert(layers, k, weights):
        out = []
        for i, layer in enumerate(layers):
            shared, rest = halves(layer["ffn"])
            experts = {}
            for j, (name, w) in enumerate(rest.items()):
                experts[name] = tile(w)
                if weights and noise:
                    kij = jax.random.fold_in(jax.random.fold_in(k, i), j)
                    experts[name] = experts[name] + noise * jnp.std(w) * jax.random.normal(kij, experts[name].shape)
            router = (0.02 * jax.random.normal(jax.random.fold_in(k, 1000 + i), (cfg.d, E)) if weights
                      else jnp.zeros((cfg.d, E)))
            new = {n: v for n, v in layer.items() if n != "ffn"}
            new["moe"] = {"router": router, "shared": shared, "experts": experts}
            out.append(new)
        return out

    p, o = dense_state["params"], dense_state["opt"]
    params = dict(p, layers=convert(p["layers"], key, True))
    opt = {"m": dict(o["m"], layers=convert(o["m"]["layers"], key, False)),
           "v": dict(o["v"], layers=convert(o["v"]["layers"], key, False)), "t": o["t"]}
    return {"params": params, "opt": opt, "step": dense_state["step"], "sched": dense_state["sched"],
            "bias": jnp.zeros((cfg.n_layer, E), F32)}


def balance_bias(scores, k, iters=300):
    """Run the bias-balancing rule on a fixed batch of router scores until the loads are even."""
    E = scores.shape[1]
    b = np.zeros(E, np.float32)
    for i in range(iters):
        top = np.argpartition(-(scores + b), k - 1, axis=1)[:, :k]
        load = np.bincount(top.ravel(), minlength=E) / top.size
        b += (0.02 * (1 - i / iters) + 1e-4) * np.sign(1 / E - load)
    return b


def calibrate_bias(params, tokens, cfg):
    """Start the balancing before step 1: layer by layer, set each router's bias so that a real batch is
    spread evenly over the experts. Without this, the first MoE steps would overflow some experts' capacity."""
    B, T = tokens.shape
    x = params["tok_emb"][tokens] + params["pos_emb"][:T]
    biases = []
    for layer in params["layers"]:
        x = x + attention(rmsnorm(x, layer["ln1"]), layer, cfg)
        h = rmsnorm(x, layer["ln2"])
        scores = np.asarray(jax.nn.sigmoid(h.reshape(-1, cfg.d) @ layer["moe"]["router"]))
        b = balance_bias(scores, cfg.top_k)
        biases.append(b)
        y, _ = moe_ffn(h, layer["moe"], jnp.asarray(b), cfg, jax.random.PRNGKey(0), False)
        x = x + y
    return jnp.asarray(np.stack(biases))


def count_params(tree):
    return int(sum(np.prod(a.shape) for a in jax.tree.leaves(tree)))


def active_params(params, cfg):
    """Parameters one token actually uses: everything except the routed experts it did not pick."""
    total = count_params(params)
    layer0 = params["layers"][0]
    if "moe" not in layer0:
        return total
    per_expert = count_params(layer0["moe"]["experts"]) // cfg.n_experts
    return total - cfg.n_layer * (cfg.n_experts - cfg.top_k) * per_expert


_d = init_dense(CFG, jax.random.PRNGKey(0))
_m = jax.eval_shape(lambda s: to_moe(s, CFG, jax.random.PRNGKey(1))["params"], make_state(_d, {"x": 0.0}))
log(f"dense: {count_params(_d) / 1e6:.1f}M parameters")
log(f"MoE:   {count_params(_m) / 1e6:.1f}M total, {active_params(_m, CFG) / 1e6:.1f}M active per token "
      f"({N_EXPERTS} routed experts, top-{TOP_K}, 1 shared; {CFG.capacity} slots per expert per group of {GROUP_SIZE})")
del _d, _m

# %% [markdown]
# ## 4. Gates: check the MoE machinery before spending TPU time
#
# Small, fast tests in fp32 at full matmul precision. The notebook stops here if any fails.
#
# 1. The fast dispatch (slots, gathers) gives exactly what a slow token-by-token loop gives.
# 2. When an expert's queue is full, exactly the right tokens are dropped.
# 3. **The conversion preserves the model**: the MoE's output equals the dense model's output.
# 4. After conversion, gradient reaches the router and every expert.
# 5. The training step compiles once and is reused (no recompiling every step).

# %%
t0 = time.time()
with jax.default_matmul_precision("highest"):
    g = Cfg(vocab=64, d=32, n_layer=2, n_head=2, ffn=64, seq=16, n_experts=4, top_k=2, capacity_factor=4.0,
            group=32, dtype="float32")
    k = jax.random.split(jax.random.PRNGKey(42), 8)
    moe_p = {"router": jax.random.normal(k[0], (32, 4)),
             "shared": {"w_gate": jax.random.normal(k[1], (32, 16)), "w_up": jax.random.normal(k[2], (32, 16)),
                        "w_down": jax.random.normal(k[3], (16, 32))},
             "experts": {"w_gate": jax.random.normal(k[4], (4, 32, 16)), "w_up": jax.random.normal(k[5], (4, 32, 16)),
                         "w_down": jax.random.normal(k[6], (4, 16, 32))}}
    h = jax.random.normal(k[7], (4, 16, 32))
    bias0 = jnp.array([0.0, 0.1, -0.1, 0.05])

    def slow_moe(h, p, bias, cfg, capacity):
        x = h.reshape(-1, cfg.group, cfg.d)
        idx, w = route(x, p, bias, cfg, jax.random.PRNGKey(0), False)
        x, idx, w = np.asarray(x), np.asarray(idx), np.asarray(w)
        out = np.zeros_like(x)
        for gi in range(x.shape[0]):
            used = np.zeros(cfg.n_experts, int)
            for ti in range(cfg.group):
                for j in range(cfg.top_k):
                    e = idx[gi, ti, j]
                    if used[e] < capacity:
                        ep = {n: np.asarray(v[e]) for n, v in p["experts"].items()}
                        out[gi, ti] += w[gi, ti, j] * np.asarray(swiglu(jnp.asarray(x[gi, ti]), ep, F32))
                    used[e] += 1
        return out.reshape(h.shape) + np.asarray(swiglu(h, p["shared"], F32))

    fast, _ = moe_ffn(h, moe_p, bias0, g, jax.random.PRNGKey(0), False)
    err1 = float(np.abs(np.asarray(fast) - slow_moe(h, moe_p, bias0, g, g.capacity)).max())
    assert err1 < 1e-4, f"gate 1 failed: fast dispatch differs from the slow loop by {err1}"
    log(f"gate 1  dispatch matches the token-by-token loop         max diff {err1:.1e}")

    tight = dataclasses.replace(g, capacity_factor=0.5)
    fast, st = moe_ffn(h, moe_p, bias0, tight, jax.random.PRNGKey(0), False)
    err2 = float(np.abs(np.asarray(fast) - slow_moe(h, moe_p, bias0, tight, tight.capacity)).max())
    assert err2 < 1e-4 and float(st["kept"]) < 1.0, f"gate 2 failed: {err2}, kept {float(st['kept'])}"
    log(f"gate 2  capacity drops the right tokens ({1 - float(st['kept']):.0%} dropped)   max diff {err2:.1e}")

    dense_s = make_state(init_dense(g, jax.random.PRNGKey(1)), {"x": 0.0})
    toks = jax.random.randint(jax.random.PRNGKey(2), (4, g.seq), 0, g.vocab)
    moe_s = to_moe(dense_s, g, jax.random.PRNGKey(3), noise=0.0)
    moe_s["bias"] = calibrate_bias(moe_s["params"], toks, g)
    ld, _ = forward(dense_s["params"], toks, g)
    lm, _ = forward(moe_s["params"], toks, g, moe_s["bias"])
    err3 = float(jnp.abs(ld - lm).max())
    assert err3 < 1e-4, f"gate 3 failed: the converted model's logits differ by {err3}"
    log(f"gate 3  converted MoE reproduces the dense model         max logit diff {err3:.1e}")

    moe_n = to_moe(dense_s, g, jax.random.PRNGKey(3), noise=0.05)
    moe_n["bias"] = calibrate_bias(moe_n["params"], toks, g)
    toks4 = jax.random.randint(jax.random.PRNGKey(4), (4, g.seq + 1), 0, g.vocab)
    grads = jax.grad(lambda p: cross_entropy(forward(p, toks4[:, :-1], g, moe_n["bias"])[0], toks4[:, 1:]))(moe_n["params"])
    for i, layer in enumerate(grads["layers"]):
        r = float(jnp.abs(layer["moe"]["router"]).sum())
        per_expert = jnp.abs(layer["moe"]["experts"]["w_up"]).sum(axis=(1, 2))
        assert r > 0 and bool(jnp.all(per_expert > 0)), f"gate 4 failed in layer {i}: router {r}, experts {per_expert}"
    log(f"gate 4  gradient reaches the router and all {g.n_experts} experts in every layer")

def compiled_count():
    try:
        return TRAIN_STEP._cache_size()     # private JAX API; if a version lacks it the check is skipped
    except AttributeError:
        return None

tiny_sched = {"peak_lr": 1e-3, "warmup": 2, "decay_start": 8, "total": 10, "final_frac": 0.1, "moe_start": 0,
              "gamma_switch": 2}
tiny_b = jax.device_put(np.random.default_rng(0).integers(0, g.vocab, (8, g.seq + 1)).astype(np.int32), DATA)
tiny_dense = make_state(init_dense(g, jax.random.PRNGKey(5)), tiny_sched)
tiny_moe = to_moe(tiny_dense, g, jax.random.PRNGKey(6))
g5 = dataclasses.replace(g, sampled_steps=2)   # MoE steps 0-1 sample, 2-3 use hard top-k: same program
for label, tiny in (("dense", tiny_dense), ("MoE", tiny_moe)):
    before = compiled_count()
    tiny = put(jax.device_get(tiny))        # own copy: the MoE shares arrays with the dense state
    for _ in range(4):
        tiny, m = TRAIN_STEP(tiny, tiny_b, g5)
    jax.block_until_ready(m["loss"])
    after = compiled_count()
    if before is not None:
        assert after - before == 1, f"gate 5 failed: the {label} training step compiled {after - before} times"
log(f"gate 5  dense and MoE training steps each compiled once and were reused")
del tiny, tiny_dense, tiny_moe, moe_s, moe_n, dense_s, grads
record_time("gates", time.time() - t0)
log(f"all gates passed ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 5. Measure the TPU, then plan the runs
#
# This is the cell that would have caught Session 13's problem before it cost three hours. It times real
# training steps (random tokens, same shapes as the real runs) for the dense model and the MoE, then works
# out how many steps fit in `TIME_BUDGET_MIN`:
#
# `steps per phase = 0.85 × budget ÷ (2 × dense step time + MoE step time)`, capped at `MAX_STEPS`
#
# (phase 1 and the dense control are dense; one branch is MoE; 15% is held back for evaluation and
# checkpoints). If that is below `MIN_STEPS`, it tries the next smaller batch. If nothing fits, it stops
# and tells you what budget would.
#
# The plan is saved to Drive. When you re-run the notebook it is reused, so a resumed session trains
# exactly what the first one started. Set `REPLAN = True` to measure again.

# %%
PLAN_PATH = os.path.join(OUT, "plan.json")

def is_oom(e):
    s = str(e).lower()
    return "resource_exhausted" in s or "out of memory" in s

def time_steps(state, cfg, B, n=10):
    batch = jax.device_put(np.random.default_rng(0).integers(0, cfg.vocab, (B, cfg.seq + 1)).astype(np.int32), DATA)
    t0 = time.perf_counter()
    state, m = TRAIN_STEP(state, batch, cfg)
    float(m["loss"])
    first = time.perf_counter() - t0                 # includes compiling
    compiled = compiled_count()
    for _ in range(2):
        state, m = TRAIN_STEP(state, batch, cfg)
    float(m["loss"])
    t0 = time.perf_counter()
    for _ in range(n):
        state, m = TRAIN_STEP(state, batch, cfg)
    float(m["loss"])
    elapsed = (time.perf_counter() - t0) / n
    if compiled is not None and compiled_count() != compiled:
        raise RuntimeError("The training step recompiled between steps; timings would be wrong and training slow.")
    return elapsed, first

BENCH_SCHED = {"peak_lr": 1e-4, "warmup": 10.0, "decay_start": 100.0, "total": 200.0, "final_frac": 0.1,
               "moe_start": 0.0, "gamma_switch": 10.0}

# What the saved plan and every cached result depend on. A folder made with different settings is refused.
FINGERPRINT = {"data": "fineweb/sample-10BT", "seq": SEQ_LEN, "vocab": VOCAB, "d": D_MODEL, "layers": N_LAYER,
               "heads": N_HEAD, "ffn": FFN_HIDDEN, "experts": N_EXPERTS, "top_k": TOP_K,
               "capacity_factor": CAPACITY_FACTOR, "group": GROUP_SIZE, "lr": PEAK_LR, "seed": SEED,
               "shard_seqs": SHARD_SEQS}

if os.path.exists(PLAN_PATH) and not REPLAN:
    PLAN = json.load(open(PLAN_PATH))
    if PLAN.get("fingerprint") != FINGERPRINT:
        raise RuntimeError(f"{OUT} holds results from different settings:\n  saved: {PLAN.get('fingerprint')}\n"
                           f"  now:   {FINGERPRINT}\nPoint DRIVE_DIR at a new folder, or delete this one to start over.")
    log(f"Using the saved plan from {PLAN['created']} (measured on {PLAN['device']}).")
    if PLAN["device"] != DEVICE_KIND:
        log(f"Note: this session has a {DEVICE_KIND}; step times will differ from the plan, the steps will not.")
else:
    t0 = time.time()
    PLAN, tried = None, []
    for B in BATCH_CHOICES:
        assert SHARD_SEQS % B == 0, f"SHARD_SEQS ({SHARD_SEQS}) must be a multiple of every batch size ({B})"
        try:
            dense = put(make_state(init_dense(CFG, jax.random.PRNGKey(0)), BENCH_SCHED))
            t_dense, c_dense = time_steps(dense, CFG, B)
            moe = put(to_moe(make_state(init_dense(CFG, jax.random.PRNGKey(0)), BENCH_SCHED), CFG, jax.random.PRNGKey(1)))
            t_moe, c_moe = time_steps(moe, CFG, B)
            del dense, moe
        except Exception as e:
            if not is_oom(e):
                raise
            log(f"batch {B}: does not fit in TPU memory")
            continue
        steps = min(MAX_STEPS, int(0.85 * TIME_BUDGET_MIN * 60 / (2 * t_dense + t_moe)))
        tokps = B * SEQ_LEN / t_dense
        log(f"batch {B}: dense {t_dense * 1000:.0f} ms/step ({tokps:,.0f} tok/s), MoE {t_moe * 1000:.0f} ms/step "
              f"-> {steps} steps per phase fit the budget  (first-call compile: {c_dense:.0f}s, {c_moe:.0f}s)")
        tried.append((B, steps))
        if steps >= MIN_STEPS:
            PLAN = {"batch": B, "steps": steps, "t_dense": t_dense, "t_moe": t_moe, "device": DEVICE_KIND,
                    "created": time.strftime("%Y-%m-%d %H:%M"), "budget_min": TIME_BUDGET_MIN,
                    "fingerprint": FINGERPRINT}
            break
    if PLAN is None:
        B, steps = tried[-1] if tried else (BATCH_CHOICES[-1], 0)
        need = MIN_STEPS * TIME_BUDGET_MIN / max(steps, 1)
        raise RuntimeError(f"The plan does not fit: at batch {B} only {steps} steps per phase fit in "
                           f"{TIME_BUDGET_MIN} min (minimum {MIN_STEPS}). Raise TIME_BUDGET_MIN to about "
                           f"{math.ceil(need)} or lower MIN_STEPS. Nothing has been trained.")
    save_json(PLAN, PLAN_PATH)
    record_time("benchmark", time.time() - t0)

    if COMPARE_DTYPES and not SMOKE:
        try:
            f32cfg = dataclasses.replace(CFG, dtype="float32")
            t32, _ = time_steps(put(make_state(init_dense(CFG, jax.random.PRNGKey(0)), BENCH_SCHED)), f32cfg, PLAN["batch"])
            log(f"precision check (dense, batch {PLAN['batch']}): bf16 {PLAN['t_dense'] * 1000:.0f} ms/step, "
                  f"fp32 {t32 * 1000:.0f} ms/step -> bf16 is {t32 / PLAN['t_dense']:.2f}x faster. Training uses bf16.")
            PLAN["t_dense_fp32"] = t32
            save_json(PLAN, PLAN_PATH)
        except Exception as e:
            log(f"(precision check skipped: {str(e)[:120]})")

S, B = PLAN["steps"], PLAN["batch"]
WARMUP = max(WARMUP_MIN_STEPS, int(WARMUP_FRAC * S))
SCHED = {"peak_lr": PEAK_LR, "warmup": WARMUP, "decay_start": 2 * S - int(DECAY_FRAC * S), "total": 2 * S,
         "final_frac": FINAL_LR_FRAC, "moe_start": S, "gamma_switch": int(GAMMA_SWITCH_FRAC * S)}
train_min = S * (2 * PLAN["t_dense"] + PLAN["t_moe"]) / 60
log(f"PLAN  batch {B} x {SEQ_LEN} tokens = {B * SEQ_LEN:,} tokens per step")
log(f"      phase 1 (dense): {S} steps | phase 2: {S} steps each for MoE and the dense control")
log(f"      {S * B * SEQ_LEN / 1e6:.0f}M tokens per phase, {2 * S * B * SEQ_LEN / 1e6:.0f}M unique tokens needed")
log(f"      warm-up {WARMUP} steps; learning rate {PEAK_LR} constant, then cosine to {FINAL_LR_FRAC:.0%} over the "
      f"last {int(DECAY_FRAC * S)} steps")
log(f"      projected training time {train_min:.0f} min (+ data, compiling and evaluation); "
      f"each run pauses at {GUARD}x its planned time")

# %% [markdown]
# ## 6. Data: FineWeb, an 8K tokenizer, and immutable shards
#
# The same corpus and recipe as Session 13: stream FineWeb `sample-10BT` (English web pages), skip documents
# shorter than 200 characters, train a byte-level BPE tokenizer with 8,192 tokens on the first ~80 MB of
# text, then tokenize documents in stream order with `<|endoftext|>` between them.
#
# The tokens are cut into **shards** of 32,768 sequences (16.8M tokens). A shard is never edited once
# written, and each one gets a **manifest** (Session 6's rule: don't train on a shard whose history you
# can't state): SHA-256 of its tokens and of its text, provenance (dataset, stream rows, Common Crawl
# dumps), cleaning steps, dedup and contamination status, language, tokenizer hash, packing policy, parent
# shard, and a resume pointer. A gzipped index lists every document in the shard with its FineWeb id, URL
# and token range, so any training sample can be traced back to the web pages it came from.
#
# The **validation shard is built from documents after every training document** in the stream, so none
# of it is ever trained on. The build resumes at the first unfinished shard if it is cut off. Downloading
# runs in a background thread while the main thread tokenizes; progress prints every 30 seconds. First run:
# 5–15 minutes. Later runs: every shard's SHA-256 is checked against its manifest (a few seconds).

# %%
t0 = time.time()
DATA_DIR = os.path.join(OUT, "data")
SHARD_DIR = os.path.join(DATA_DIR, "shards")
os.makedirs(SHARD_DIR, exist_ok=True)
TOK_PATH = os.path.join(DATA_DIR, "tokenizer.json")
EOS_TOKEN = "<|endoftext|>"
SHARD_TOKENS = SHARD_SEQS * SEQ_LEN + 1       # SHARD_SEQS windows of SEQ_LEN + 1 tokens at stride SEQ_LEN
VAL_TOKENS = VAL_SEQS * SEQ_LEN + 1
N_TRAIN_SHARDS = math.ceil(2 * S * B / SHARD_SEQS)
SHARD_IDS = [f"train_{i:05d}" for i in range(N_TRAIN_SHARDS)] + ["val_00000"]
PIPELINE = {"dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "split": "train",
            "min_doc_chars": MIN_DOC_CHARS, "tokenizer": f"byte-level BPE, {VOCAB} tokens, first {TOKENIZER_MB} MB",
            "eos": EOS_TOKEN, "seq_len": SEQ_LEN, "shard_seqs": SHARD_SEQS, "val_seqs": VAL_SEQS,
            "packing": "documents concatenated with EOS between them, no padding; windows of seq_len + 1 tokens "
                       "at stride seq_len (100% of tokens are real text)"}
PIPELINE_SHA = hashlib.sha256(json.dumps(PIPELINE, sort_keys=True).encode()).hexdigest()


def shard_files(sid):
    base = os.path.join(SHARD_DIR, sid)
    return base + ".bin", base + ".manifest.json", base + ".docs.jsonl.gz"


def shard_ok(sid, tok_sha):
    b, m, d = shard_files(sid)
    if not (os.path.exists(b) and os.path.exists(m) and os.path.exists(d)):
        return False
    man = json.load(open(m))
    return (man["pipeline_sha256"] == PIPELINE_SHA and man["tokenizer"]["sha256"] == tok_sha
            and os.path.getsize(b) == 2 * man["n_tokens"])


from tokenizers import Tokenizer
TOK_META = TOK_PATH.replace(".json", ".meta.json")
tok_valid = os.path.exists(TOK_PATH) and os.path.exists(TOK_META) and \
    json.load(open(TOK_META)).get("pipeline_sha256") == PIPELINE_SHA
TOK_SHA = sha256_file(TOK_PATH) if tok_valid else None
missing = [sid for sid in SHARD_IDS if not tok_valid or not shard_ok(sid, TOK_SHA)]

if not missing:
    TOK = Tokenizer.from_file(TOK_PATH)
    log(f"data cached: {N_TRAIN_SHARDS} training shards + 1 validation shard")
else:
    import gzip, queue, threading
    from datasets import load_dataset
    from tokenizers import models, trainers, pre_tokenizers, decoders

    first = SHARD_IDS.index(missing[0])        # shards are written in order: rebuild from the first missing one
    start_row, start_offset = 0, 0
    if first > 0:
        prev = json.load(open(shard_files(SHARD_IDS[first - 1])[1]))
        start_row, start_offset = prev["next"]["row"], prev["next"]["token_offset"]
        if SHARD_IDS[first].startswith("val") and start_offset:   # validation starts on a whole document
            start_row, start_offset = start_row + 1, 0
        log(f"resuming the data build at {SHARD_IDS[first]} ({first} shards already written, stream row "
            f"{start_row:,}). Earlier rows are skipped: re-read from the network, not re-tokenized.")

    def fineweb_rows(skip):
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        ds = ds.select_columns(["text", "id", "url", "dump", "language_score"])
        if skip:
            ds = ds.skip(skip)
        row = skip
        for batch in ds.iter(batch_size=1000):
            yield row, batch
            row += len(batch["text"])

    def prefetch(gen, threads, depth=32):
        """Run `gen` in a background thread, so downloading continues while the main thread tokenizes.
        Closing the returned generator tells the thread to stop after the read it is in the middle of."""
        q, done, err, stop = queue.Queue(depth), object(), [], threading.Event()
        def put_item(item):                         # wait for room, but give up as soon as we are told to stop
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    pass
            return False
        def work():
            try:
                for item in gen:
                    if not put_item(item):
                        return
            except BaseException as e:             # re-raised below, so a failed download is never silent
                err.append(e)
            finally:
                put_item(done)
        worker = threading.Thread(target=work, daemon=True, name="fineweb-download")
        threads.append(worker)
        worker.start()
        try:
            while (item := q.get()) is not done:
                yield item
        finally:
            stop.set()
        if err:
            raise err[0]

    download_threads = []
    stream = prefetch(fineweb_rows(start_row), download_threads)
    buffered = []                                   # rows read for the tokenizer are tokenized too, not re-downloaded
    if tok_valid:
        TOK = Tokenizer.from_file(TOK_PATH)
    else:
        for sid in SHARD_IDS:                       # a new tokenizer invalidates every existing shard
            for f in shard_files(sid):
                if os.path.exists(f):
                    os.remove(f)
        corpus, chars = [], 0
        for row, batch in stream:                   # 1. the first ~80 MB trains the tokenizer
            buffered.append((row, batch))
            texts = [t for t in batch["text"] if len(t) >= MIN_DOC_CHARS]
            corpus += texts
            chars += sum(map(len, texts))
            if chars >= TOKENIZER_MB * 1e6:
                break
        log(f"tokenizer text: {len(corpus):,} documents, {chars / 1e6:.0f} MB ({time.time() - t0:.0f}s)")
        TOK = Tokenizer(models.BPE())
        TOK.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        TOK.decoder = decoders.ByteLevel()
        TOK.train_from_iterator(corpus, trainers.BpeTrainer(
            vocab_size=VOCAB, min_frequency=2, special_tokens=[EOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        TOK.save(TOK_PATH)
        save_json({"pipeline_sha256": PIPELINE_SHA, "trained_on_docs": len(corpus), "trained_on_mb": round(chars / 1e6)},
                  TOK_META)
        corpus = None
        log(f"tokenizer trained: {TOK.get_vocab_size():,} tokens ({time.time() - t0:.0f}s)")
    TOK_SHA = sha256_file(TOK_PATH)
    eos = TOK.token_to_id(EOS_TOKEN)
    encode_batch = getattr(TOK, "encode_batch_fast", TOK.encode_batch)   # _fast skips offsets: ~30% quicker

    def new_shard(k):
        sid = SHARD_IDS[k]
        return {"sid": sid, "buf": np.empty(VAL_TOKENS if sid.startswith("val") else SHARD_TOKENS, np.uint16),
                "filled": 0, "pieces": [], "text_sha": hashlib.sha256(), "dumps": {}, "lang": [], "rejected": 0,
                "rows": [None, None], "t0": time.time()}

    def touch(cur, text, dump, lang):              # a document's first piece in this shard
        cur["text_sha"].update(text.encode("utf-8"))
        cur["dumps"][dump] = cur["dumps"].get(dump, 0) + 1
        cur["lang"].append(lang)

    def finish(cur, nxt):
        sid = cur["sid"]
        b, m, d = shard_files(sid)
        data = cur["buf"].tobytes()
        with open(b + ".tmp", "wb") as f:
            f.write(data)
        os.replace(b + ".tmp", b)
        with gzip.open(d + ".tmp", "wt") as f:
            for piece in cur["pieces"]:
                f.write(json.dumps(piece) + "\n")
        os.replace(d + ".tmp", d)
        docs = len({pc["row"] for pc in cur["pieces"]})
        val = sid.startswith("val")
        manifest = {
            "shard_id": sid, "split": "validation" if val else "train", "parent_shard_id": None,
            "n_tokens": len(cur["buf"]), "n_sequences": VAL_SEQS if val else SHARD_SEQS, "seq_len": SEQ_LEN,
            "dtype": "uint16 token ids", "token_sha256": hashlib.sha256(data).hexdigest(),
            "text_sha256": cur["text_sha"].hexdigest(), "n_docs": docs, "docs_index": os.path.basename(d),
            "first_doc_continues_from_previous_shard": bool(cur["pieces"] and cur["pieces"][0]["doc_tok_offset"] > 0),
            "provenance": {"dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "split": "train",
                           "stream_rows": cur["rows"], "common_crawl_dumps": dict(sorted(cur["dumps"].items()))},
            "cleaning": {"upstream": "FineWeb's own pipeline (see its dataset card): URL filtering, text extraction, "
                                     "English language ID, quality and repetition filters, MinHash deduplication "
                                     "within each Common Crawl dump, PII anonymisation",
                         "here": f"rows shorter than {MIN_DOC_CHARS} characters dropped",
                         "rows_dropped": cur["rejected"]},
            "dedup": "FineWeb MinHash within each dump; not re-checked across dumps here",
            "contamination": "not checked against evaluation benchmarks",
            "eval_overlap": ("held out: every document here comes after every training document in the stream"
                             if val else "none: training shard"),
            "language": {"code": "en", "mean_fineweb_language_score": round(float(np.mean(cur["lang"])), 4)},
            "lane": "general English web text",
            "tokenizer": {"file": "tokenizer.json", "sha256": TOK_SHA, "vocab": VOCAB, "eos": EOS_TOKEN},
            "packing": PIPELINE["packing"], "pipeline_sha256": PIPELINE_SHA,
            "next": nxt, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_seconds": round(time.time() - cur["t0"], 1)}
        save_json(manifest, m)                      # written last: a shard exists only once its manifest does
        log(f"shard {sid}: {len(cur['buf']) / 1e6:.1f}M tokens from {docs:,} documents, "
            f"sha256 {manifest['token_sha256'][:12]}… ({time.time() - t0:.0f}s)")
        event("shard_written", shard=sid, tokens=len(cur["buf"]), docs=docs, sha256=manifest["token_sha256"])

    def all_rows():
        yield from buffered
        yield from stream

    k, cur, skip_tokens, done, t_print = first, new_shard(first), start_offset, False, time.time()
    total_needed = sum(SHARD_TOKENS if sid.startswith("train") else VAL_TOKENS for sid in SHARD_IDS[first:])
    written = 0
    for row0, batch in all_rows():
        texts = batch["text"]
        keep = [i for i, t in enumerate(texts) if len(t) >= MIN_DOC_CHARS]
        encs = dict(zip(keep, encode_batch([texts[i] for i in keep])))
        for i in range(len(texts)):
            row = row0 + i
            if row < start_row:
                continue
            if i not in encs:
                cur["rejected"] += 1
                continue
            ids = encs[i].ids + [eos]
            pos, skip_tokens = skip_tokens, 0       # only the very first document can start part-way
            touch(cur, texts[i], batch["dump"][i], batch["language_score"][i])
            while pos < len(ids):
                take = min(len(ids) - pos, len(cur["buf"]) - cur["filled"])
                cur["buf"][cur["filled"]:cur["filled"] + take] = ids[pos:pos + take]
                cur["pieces"].append({"row": row, "id": batch["id"][i], "url": batch["url"][i], "dump": batch["dump"][i],
                                      "tok_start": cur["filled"], "n_tok": take, "doc_tok_offset": pos,
                                      "doc_tokens": len(ids)})
                cur["rows"] = [row if cur["rows"][0] is None else cur["rows"][0], row]
                cur["filled"] += take
                pos += take
                if cur["filled"] < len(cur["buf"]):
                    continue
                finish(cur, {"row": row, "token_offset": pos} if pos < len(ids) else {"row": row + 1, "token_offset": 0})
                written += len(cur["buf"])
                k += 1
                if k == len(SHARD_IDS):
                    done = True
                    break
                cur = new_shard(k)
                if cur["sid"].startswith("val"):    # validation starts on the next whole document
                    break
                if pos < len(ids):
                    touch(cur, texts[i], batch["dump"][i], batch["language_score"][i])
            if done:
                break
        if done:
            break
        if time.time() - t_print > 30:
            t_print = time.time()
            got = written + cur["filled"]
            rate = got / max(1e-9, time.time() - t0)
            log(f"  data: {got / 1e6:,.0f}M / {total_needed / 1e6:,.0f}M tokens | {rate / 1e6:.2f}M tok/s | "
                f"~{(total_needed - got) / max(rate, 1) / 60:.1f} min left")
    stream.close()                                  # tell the download thread to stop ...
    for th in download_threads:                     # ... and wait for its last read, so nothing keeps running
        th.join(timeout=180)
        if th.is_alive():
            log("note: the download thread is still finishing its last read; it stops by itself.")
    if not done:
        raise RuntimeError("The FineWeb stream ended before the plan had enough tokens.")
    record_time("data_build", time.time() - t0)

# Verify every shard against its manifest before training on it, and read it from local disk, not Drive.
DATA_MANIFEST = {"pipeline": PIPELINE, "pipeline_sha256": PIPELINE_SHA, "tokenizer_sha256": TOK_SHA, "shards": []}
for sid in SHARD_IDS:
    b, m, d = shard_files(sid)
    man = json.load(open(m))
    local = os.path.join(LOCAL, os.path.basename(b))
    if os.path.abspath(b) != os.path.abspath(local) and (not os.path.exists(local) or os.path.getsize(local) != os.path.getsize(b)):
        shutil.copy(b, local)
    if sha256_file(local) != man["token_sha256"]:
        raise RuntimeError(f"{sid}: tokens do not match the SHA-256 in its manifest. Delete it and re-run to rebuild.")
    DATA_MANIFEST["shards"].append({k: man[k] for k in ("shard_id", "split", "n_tokens", "n_sequences", "n_docs",
                                                          "token_sha256", "text_sha256")})
save_json(DATA_MANIFEST, os.path.join(DATA_DIR, "manifest.json"))
log(f"{len(SHARD_IDS)} shards verified against their manifests (SHA-256) and copied to local disk")
record_time("data", time.time() - t0)

# %% [markdown]
# ### The loader, and which shard every step reads on which chip
#
# Training shards are read in a fixed shuffled **shard order**; inside a shard, sequences are read in a
# fixed shuffle seeded by the shard's number. A step never spans two shards. Both phase-2 runs read exactly
# the same shards and sequences, step for step. Each batch is split across the chips by row (one chip on a
# v5e-1, so every row goes to `TPU_0`); the placement below is read from the actual device arrays, not
# assumed. All of this goes into `ledger/run_manifest.json`.

# %%
def local_shard(sid):
    return os.path.join(LOCAL, sid + ".bin")


class ShardLoader:
    """Global step -> (shard, sequence indices inside it). Deterministic, so any step can be replayed."""
    def __init__(self, order, T, B, seed):
        assert SHARD_SEQS % B == 0
        self.order, self.T, self.B = order, T, B
        self.steps_per_shard = SHARD_SEQS // B
        self.maps = {sid: np.memmap(local_shard(sid), np.uint16, "r") for sid in order}
        self.perms = {sid: np.random.default_rng([seed, int(sid.split("_")[1])]).permutation(SHARD_SEQS)
                      for sid in order}

    def locate(self, step):
        k, j = divmod(step, self.steps_per_shard)
        sid = self.order[k % len(self.order)]
        return sid, self.perms[sid][j * self.B:(j + 1) * self.B]

    def get(self, step):
        sid, rows = self.locate(step)
        starts = rows.astype(np.int64) * self.T
        return self.maps[sid][starts[:, None] + np.arange(self.T + 1)].astype(np.int32)


SHARD_ORDER = [SHARD_IDS[i] for i in np.random.default_rng(SEED).permutation(N_TRAIN_SHARDS)]
LOADER = ShardLoader(SHARD_ORDER, SEQ_LEN, B, SEED)
val = np.fromfile(local_shard("val_00000"), np.uint16).astype(np.int32)
VAL = [jax.device_put(np.stack([val[j * SEQ_LEN: j * SEQ_LEN + SEQ_LEN + 1] for j in range(i, i + B)]), DATA)
       for i in range(0, VAL_SEQS, B)]

probe = jax.device_put(np.zeros((B, SEQ_LEN + 1), np.int32), DATA)      # where each batch row really lands
PLACEMENT = []
for sh in probe.addressable_shards:
    rows = sh.index[0]
    PLACEMENT.append({"device": f"{sh.device.platform.upper()}_{sh.device.id}", "kind": sh.device.device_kind,
                      "rows": [rows.start or 0, B if rows.stop is None else rows.stop]})
PLACEMENT.sort(key=lambda p: p["rows"][0])
del probe

sps = LOADER.steps_per_shard
SHARD_SCHEDULE = []
for k in range(math.ceil(2 * S / sps)):
    a, b = k * sps, min((k + 1) * sps, 2 * S)
    phase = "phase 1 (dense)" if b <= S else ("phase 2 (MoE and control)" if a >= S else "phase 1 → phase 2")
    SHARD_SCHEDULE.append({"steps": [a, b - 1], "shard": LOADER.locate(a)[0], "phase": phase})

log(f"every step: {B} sequences x {SEQ_LEN + 1} tokens; " +
    "; ".join(f"rows {p['rows'][0]}–{p['rows'][1] - 1} → {p['device']} ({p['kind']})" for p in PLACEMENT))
log(f"{sps} steps per shard. Which shard each step reads (phase 2 starts at step {S}):")
for e in SHARD_SCHEDULE:
    log(f"  steps {e['steps'][0]:>6} – {e['steps'][1]:>6}   {e['shard']}   {e['phase']}")

RUN_MANIFEST_PATH = os.path.join(LEDGER, "run_manifest.json")
RUN_MANIFEST = json.load(open(RUN_MANIFEST_PATH)) if os.path.exists(RUN_MANIFEST_PATH) else {"sessions": []}
RUN_MANIFEST.update({
    "fingerprint": FINGERPRINT, "plan": PLAN, "schedule": SCHED, "seed": SEED,
    "rng": f"model init PRNGKey({SEED}); router noise PRNGKey({SEED}) folded with the global step; "
           f"shard order and in-shard shuffles numpy default_rng seeded with {SEED} (and the shard number)",
    "data": {"pipeline_sha256": PIPELINE_SHA, "tokenizer_sha256": TOK_SHA, "shard_order": SHARD_ORDER,
             "steps_per_shard": sps, "validation_shard": "val_00000"},
    "devices": [{"id": d.id, "name": f"{d.platform.upper()}_{d.id}", "kind": d.device_kind,
                 "coords": list(getattr(d, "coords", []) or []), "process": d.process_index} for d in jax.devices()],
    "mesh": {"axes": list(MESH.axis_names), "shape": dict(MESH.shape)},
    "placement": PLACEMENT, "shard_schedule": SHARD_SCHEDULE,
    "runs": {"1_dense": {"global_steps": [0, S - 1], "model": "dense"},
             "2_moe": {"global_steps": [S, 2 * S - 1], "model": "MoE converted from 1_dense at step " + str(S)},
             "3_dense_control": {"global_steps": [S, 2 * S - 1], "model": "dense, continued from 1_dense"}},
    "checkpoint_contents": ["weights", "AdamW moments and step count (so the schedule continues)",
                            "router biases (MoE)", "loss curves", "this run's ledger lines",
                            "the data position (derived from the global step)"],
})
RUN_MANIFEST["sessions"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(T_START)),
                                 "device": DEVICE_KIND, "jax": jax.__version__})
save_json(RUN_MANIFEST, RUN_MANIFEST_PATH)
log("sample:", TOK.decode(LOADER.get(0)[0, :60].tolist())[:200].replace("\n", " "))

# %% [markdown]
# ## 7. The training loop
#
# One function runs every phase. Each step it reads a batch from the loader, trains, and (every 25 steps)
# writes one ledger line per step. It evaluates on the held-out shard 16 times per run and prints a
# progress line at least once a minute.
#
# **Resuming.** Every 5 minutes it writes a checkpoint (weights, optimizer, router biases, curves, and this
# run's ledger lines), keeping the previous one as a backup. If you press *Stop* or an error is raised, it
# first saves an emergency checkpoint at the exact step. On the next run it loads the newest readable
# checkpoint, rewrites the step ledger to match it exactly, and carries on. If a run passes 1.5× its
# planned time it saves and pauses with a message; running the notebook again continues it.

# %%
class PausedRun(Exception):
    pass


def evaluate(state, cfg=CFG):
    total = sum(float(EVAL_STEP(state["params"], state.get("bias"), b, cfg)) for b in VAL)
    return total / (VAL_SEQS * SEQ_LEN)


def batch_hash(batch):
    return hashlib.sha1(np.ascontiguousarray(batch).tobytes()).hexdigest()[:16]


def ckpt_files(rd):
    return os.path.join(rd, "ckpt.pkl"), os.path.join(rd, "ckpt_prev.pkl")


def write_checkpoint(name, rd, blob, first_step, reason):
    t = time.time()
    data = pickle.dumps(jax.device_get(blob), protocol=4)
    digest = hashlib.sha256(data).hexdigest()
    cur, prev = ckpt_files(rd)
    if os.path.exists(cur):
        os.replace(cur, prev)                      # keep the previous checkpoint as a backup
    with open(cur + ".tmp", "wb") as f:
        f.write(data)
    os.replace(cur + ".tmp", cur)
    g = first_step + blob["next"]
    sid, _ = LOADER.locate(g)
    rec = {"checkpoint": f"{name}@{g}", "global_step": g, "run_step": blob["next"], "reason": reason,
           "file": os.path.relpath(cur, OUT), "bytes": len(data), "sha256": digest,
           "resume_reads": {"shard": sid, "step_in_shard": g % LOADER.steps_per_shard},
           "lr_last": blob["curves"]["lr"][-1] if blob["curves"]["lr"] else None,
           "rng": f"PRNGKey({SEED}) folded with global step {g}", "ledger_lines": len(blob["ledger"]),
           "write_seconds": round(time.time() - t, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(LEDGER, f"{name}.checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    event("checkpoint", run=name, step=g, reason=reason, sha256=digest[:16], bytes=len(data))
    log(f"[{name}] checkpoint at step {g} ({reason}): {len(data) / 1e6:.0f} MB in {time.time() - t:.0f}s")


def read_checkpoint(name, rd):
    for path in ckpt_files(rd):
        if not os.path.exists(path):
            continue
        try:
            return load_pickle(path), path
        except Exception as e:                      # e.g. Drive had not finished uploading it
            log(f"[{name}] {os.path.basename(path)} is unreadable ({type(e).__name__}); trying the backup")
    return None, None


def shard_stats(ledger_lines):
    stats = {}
    for line in ledger_lines:
        r = json.loads(line)
        st = stats.setdefault(r["shard"], {"shard": r["shard"], "first_step": r["step"], "last_step": r["step"],
                                           "steps": 0, "tokens": 0, "loss_sum": 0.0, "devices": sorted(r["devices"])})
        st["last_step"], st["steps"], st["tokens"] = r["step"], st["steps"] + 1, st["tokens"] + r["tokens"]
        st["loss_sum"] += r["loss"]
    out = []
    for st in sorted(stats.values(), key=lambda x: x["first_step"]):
        st["mean_loss"] = round(st.pop("loss_sum") / st["steps"], 4)
        st["ppl"] = round(math.exp(st["mean_loss"]), 1)
        out.append(st)
    return out


def train_run(name, phase, state, n_steps, planned_s, keep_optimizer):
    """Train `state` for n_steps. Returns (final state or params, summary, curves). Skips if already finished;
    resumes from the newest checkpoint if one exists."""
    rd = os.path.join(OUT, "runs", name)
    os.makedirs(rd, exist_ok=True)
    fin, summ = os.path.join(rd, "final.pkl"), os.path.join(rd, "summary.json")
    steps_path = os.path.join(LEDGER, f"{name}.steps.jsonl")
    if os.path.exists(summ) and os.path.exists(fin):
        done = json.load(open(summ))
        if done["planned_steps"] != n_steps:
            raise RuntimeError(f"{rd} was trained with a different plan ({done['planned_steps']} steps, now {n_steps}). "
                               "Delete the runs folder to retrain, or set REPLAN = False to keep the old plan.")
        log(f"[{name}] finished in an earlier session: loading it")
        return load_pickle(fin), done, json.load(open(os.path.join(rd, "log.json")))

    moe = "bias" in state
    first_step = int(state["step"])
    curves = {"step": [], "train_loss": [], "time": [], "tok_s": [], "loader_ms": [], "lr": [], "gnorm": [],
              "val_step": [], "val_loss": [], "val_time": [], "resumes": []}
    if moe:
        curves.update(load=[], kept=[], bias=[])
    lines, start, t_train, compile_s = [], 0, 0.0, None
    t_enter = time.time()
    blob, path = read_checkpoint(name, rd)
    if blob is not None:
        state, curves, lines, start, t_train = blob["state"], blob["curves"], blob["ledger"], blob["next"], blob["t_train"]
        log(f"[{name}] RESUMING at step {first_step + start} ({start}/{n_steps} done) from {os.path.basename(path)}; "
            f"{len(lines)} ledger lines restored, {t_train / 60:.1f} min of training already done")
    with open(steps_path, "w") as f:               # the step ledger always matches the state we continue from
        f.writelines(lines)
    # Fresh device buffers owned by this run: the training step frees its input ("donation"), and a converted
    # MoE shares arrays (embeddings, attention) with the dense model it came from.
    state = put(jax.device_get(state))
    event("run_resume" if start else "run_start", run=name, phase=phase, step=first_step + start, n_steps=n_steps)
    budget = GUARD * planned_s * (n_steps - start) / n_steps   # this session's allowance for the remaining steps
    t_session, pending, loader_s = 0.0, [], 0.0

    def do_eval(s):
        curves["val_step"].append(first_step + s)
        curves["val_loss"].append(evaluate(state))
        curves["val_time"].append(t_train)

    def flush():
        """Fetch the finished steps' metrics and write their ledger lines."""
        nonlocal pending, t_train, t_session, loader_s
        if not pending:
            return None
        ms = jax.device_get([p[1] for p in pending])
        dt = time.perf_counter() - t_mark
        new_lines = []
        for (g, _, sid, rows, bh), m in zip(pending, ms):
            rec = {"run": name, "step": g, "phase": phase, "shard": sid,
                   "devices": {p["device"]: {"rows": p["rows"], "samples": rows[p["rows"][0]:p["rows"][1]].tolist()}
                               for p in PLACEMENT},
                   "loss": round(float(m["loss"]), 4), "ppl": round(math.exp(float(m["loss"])), 2),
                   "lr": float(f"{float(m['lr']):.4e}"), "gnorm": round(float(m["gnorm"]), 4),
                   "sample_loss": np.round(np.asarray(m["sample_loss"], np.float64), 3).tolist(),
                   "batch_sha1": bh, "tokens": B * SEQ_LEN}
            if moe:
                rec["dropped"] = round(1 - float(m["kept"]), 4)
                rec["busiest_expert_x_fair"] = round(float(np.max(m["load"])) * N_EXPERTS, 3)
            new_lines.append(json.dumps(rec) + "\n")
        with open(steps_path, "a") as f:
            f.writelines(new_lines)
        t_train += dt
        t_session += dt
        lines.extend(new_lines)
        curves["step"].append(pending[-1][0] + 1)
        curves["train_loss"].append(float(np.mean([x["loss"] for x in ms])))
        curves["time"].append(t_train)
        curves["tok_s"].append(len(ms) * B * SEQ_LEN / dt)
        curves["loader_ms"].append(1000 * loader_s / len(ms))
        curves["lr"].append(float(ms[-1]["lr"]))
        curves["gnorm"].append(float(np.mean([x["gnorm"] for x in ms])))
        if moe:
            curves["load"].append(np.mean([x["load"] for x in ms], axis=0).round(5).tolist())
            curves["kept"].append(float(np.mean([x["kept"] for x in ms])))
            curves["bias"].append(np.asarray(ms[-1]["bias"]).round(5).tolist())
        pending, loader_s = [], 0.0
        return ms

    def checkpoint(next_step, reason):
        write_checkpoint(name, rd, {"state": state, "curves": curves, "ledger": lines, "next": next_step,
                                    "t_train": t_train}, first_step, reason)

    if start == 0:
        do_eval(0)
        log(f"[{name}] {phase}: {n_steps} steps from global step {first_step}; val loss at start "
            f"{curves['val_loss'][-1]:.4f}; planned {planned_s / 60:.1f} min")
    eval_every = max(1, n_steps // 16)
    t_mark, t_print, t_ckpt = time.perf_counter(), time.time(), time.time()
    s = start
    try:
        for s in range(start, n_steps):
            g = first_step + s
            tl = time.perf_counter()
            host = LOADER.get(g)
            sid, rows = LOADER.locate(g)
            batch = jax.device_put(host, DATA)
            loader_s += time.perf_counter() - tl
            state, m = TRAIN_STEP(state, batch, CFG)
            pending.append((g, m, sid, rows, batch_hash(host)))
            if compile_s is None:                    # the first call may compile: keep it out of the timing
                jax.block_until_ready(m["loss"])
                compile_s = time.perf_counter() - t_mark
                if start:
                    curves["resumes"].append({"at_step": g, "latency_s": round(time.time() - t_enter, 1)})
                    event("resume_latency", run=name, step=g, seconds=round(time.time() - t_enter, 1))
                t_mark = time.perf_counter()
            if os.environ.get("S14_STOP_AT") == f"{name}:{s}":       # test hook: like pressing Stop
                raise KeyboardInterrupt("simulated Stop")
            if os.environ.get("S14_KILL_AT") == f"{name}:{s}":       # test hook: like the machine vanishing
                os._exit(1)
            if (s + 1) % LOG_EVERY and s + 1 != n_steps:
                continue
            flush()
            evaluated = (s + 1) % eval_every < LOG_EVERY or s + 1 == n_steps
            if evaluated:
                do_eval(s + 1)
            if evaluated or time.time() - t_print >= PRINT_EVERY_SEC:
                t_print = time.time()
                eta = t_session / (s + 1 - start) * (n_steps - s - 1) / 60
                extra = f" | dropped {1 - curves['kept'][-1]:.1%}" if moe else ""
                log(f"[{name}] step {g + 1:>6} ({s + 1}/{n_steps}, {(s + 1) / n_steps:.0%}) | shard {sid} | train "
                    f"{curves['train_loss'][-1]:.4f} | val {curves['val_loss'][-1]:.4f} | {curves['tok_s'][-1]:,.0f} "
                    f"tok/s | loader {curves['loader_ms'][-1]:.1f} ms/step{extra} | {t_train / 60:.1f} min, "
                    f"~{eta:.1f} min left")
            if t_session > budget and s + 1 < n_steps:
                checkpoint(s + 1, "paused: over the time guard")
                event("run_paused", run=name, step=g + 1, minutes=round(t_session / 60, 1))
                raise PausedRun(f"[{name}] PAUSED at step {g + 1} ({s + 1}/{n_steps}): this session spent "
                                f"{t_session / 60:.1f} min on it, over {GUARD}x the plan. Progress is saved. Run the "
                                f"notebook again to continue from step {g + 1}, or change GUARD.")
            if time.time() - t_ckpt >= CKPT_EVERY_MIN * 60 and s + 1 < n_steps:
                checkpoint(s + 1, "periodic")
                t_ckpt = time.time()
            t_mark = time.perf_counter()
    except PausedRun as e:
        log(str(e))
        raise
    except BaseException as e:
        log(f"[{name}] interrupted at step {first_step + s} ({type(e).__name__}: {e}); saving progress…")
        try:
            flush()
            steps_done = int(jax.device_get(state["step"])) - first_step
            if lines and json.loads(lines[-1])["step"] == first_step + steps_done - 1:
                checkpoint(steps_done, f"interrupted ({type(e).__name__})")
                log(f"[{name}] progress saved at step {first_step + steps_done}. Run the notebook again to continue "
                    "from exactly there.")
            else:
                log(f"[{name}] the last step's results were not recorded; the next session resumes from the last "
                    "periodic checkpoint instead.")
        except BaseException as e2:
            log(f"[{name}] could not save ({type(e2).__name__}); the next session resumes from the last checkpoint.")
        event("run_interrupted", run=name, step=first_step + s, error=type(e).__name__)
        raise

    host = jax.device_get(state)
    keep = host if keep_optimizer else {"params": host["params"], "bias": host.get("bias"), "step": host["step"]}
    save_pickle(keep, fin)
    stats = shard_stats(lines)
    save_json(stats, os.path.join(LEDGER, f"{name}.shards.json"))
    steps_done = curves["step"][-1] - first_step
    tail = max(1, len(curves["train_loss"]) // 20)
    summary = {"name": name, "phase": phase, "moe": moe, "steps": steps_done, "planned_steps": n_steps,
               "tokens": steps_done * B * SEQ_LEN, "first_global_step": first_step,
               "val_start": curves["val_loss"][0], "val_end": curves["val_loss"][-1],
               "train_end": float(np.mean(curves["train_loss"][-tail:])),
               "train_min": t_train / 60, "tok_s": float(np.median(curves["tok_s"])),
               "loader_ms": float(np.median(curves["loader_ms"])), "compile_s": compile_s,
               "resumes": curves["resumes"], "shards_read": [st["shard"] for st in stats],
               "params": count_params(host["params"]), "active_params": active_params(host["params"], CFG)}
    save_json(curves, os.path.join(rd, "log.json"))
    save_json(summary, summ)
    for f in ckpt_files(rd):
        if os.path.exists(f):
            os.remove(f)
    record_time(f"run_{name}", t_train)
    event("run_finished", run=name, steps=steps_done, val_end=summary["val_end"], minutes=round(t_train / 60, 1))
    log(f"[{name}] done: val {summary['val_start']:.4f} -> {summary['val_end']:.4f} in {t_train / 60:.1f} min "
        f"(resumed {len(curves['resumes'])} time(s)); shards read: {', '.join(summary['shards_read'])}")
    return keep, summary, curves

# %% [markdown]
# ## 8. Phase 1: train the dense model

# %%
dense0 = make_state(init_dense(CFG, jax.random.PRNGKey(SEED)), SCHED)
DENSE_FINAL, SUM_DENSE, LOG_DENSE = train_run("1_dense", "phase 1 dense", dense0, S, S * PLAN["t_dense"],
                                              keep_optimizer=True)
del dense0

# %% [markdown]
# ## 9. Convert the dense model into an MoE
#
# 1. Split every layer's FFN (see the top of the notebook): shared expert = first half of the neurons,
#    8 routed experts = copies of the second half plus 1% noise, new router. Adam's moments are split the
#    same way and the step count continues, so the optimizer carries on instead of starting cold.
# 2. **Pre-balance the router**: run the balancing rule on a real batch, layer by layer, so no expert's
#    capacity overflows on the first step.
# 3. **Continuity check** on the held-out set: the MoE, before it has taken a single step, should score the
#    same loss as the dense model it came from. If it is off by more than 0.1 the notebook stops here,
#    because something in the conversion is wrong.

# %%
t0 = time.time()
CONV_PATH = os.path.join(OUT, "runs", "conversion.json")
moe_done = os.path.exists(os.path.join(OUT, "runs", "2_moe", "summary.json"))
moe_resume = any(os.path.exists(f) for f in ckpt_files(os.path.join(OUT, "runs", "2_moe")))
if moe_done or moe_resume:
    MOE0 = None
    CONV = json.load(open(CONV_PATH))
    log("conversion already done in an earlier session")
else:
    dense_dev = put(DENSE_FINAL)
    moe_state = to_moe(dense_dev, CFG, jax.random.PRNGKey(SEED + 1))
    calib = jnp.asarray(LOADER.get(S - 1)[:, :-1])             # a phase-1 batch the model has already seen
    moe_state["bias"] = calibrate_bias(moe_state["params"], calib, CFG)
    MOE0 = put(moe_state)
    v_dense, v_moe = evaluate(dense_dev), evaluate(MOE0)
    _, st = jax.jit(forward, static_argnames="cfg")(MOE0["params"], VAL[0][:, :-1], CFG, MOE0["bias"])
    CONV = {"val_dense": v_dense, "val_moe_step0": v_moe, "delta": v_moe - v_dense,
            "dropped_at_conversion": 1 - float(np.mean([float(s["kept"]) for s in st])),
            "params_dense": count_params(DENSE_FINAL["params"]), "params_moe": count_params(MOE0["params"]),
            "active_moe": active_params(MOE0["params"], CFG)}
    save_json(CONV, CONV_PATH)
    del dense_dev
log(f"val loss, dense model at the end of phase 1:  {CONV['val_dense']:.4f}")
log(f"val loss, converted MoE before any training:  {CONV['val_moe_step0']:.4f}   (difference {CONV['delta']:+.4f}, "
      f"{CONV['dropped_at_conversion']:.1%} of expert choices over capacity)")
log(f"parameters: dense {CONV['params_dense'] / 1e6:.1f}M -> MoE {CONV['params_moe'] / 1e6:.1f}M total, "
      f"{CONV['active_moe'] / 1e6:.1f}M active per token")
if abs(CONV["delta"]) > 0.1:
    raise RuntimeError("The converted model does not reproduce the dense model (difference > 0.1). "
                       "Stopping before phase 2; check the conversion.")
record_time("conversion", time.time() - t0)

# %% [markdown]
# ## 10a. Phase 2: keep training the MoE
#
# Watch the `dropped` column: it is the share of expert choices that found their expert full. The bias
# balancing should keep it to a few percent or less.

# %%
if MOE0 is None and not moe_done:              # resuming: the checkpoint holds the state
    MOE0 = put(to_moe(put(DENSE_FINAL), CFG, jax.random.PRNGKey(SEED + 1)))
MOE_FINAL, SUM_MOE, LOG_MOE = train_run("2_moe", "phase 2 MoE", MOE0 if MOE0 is not None else DENSE_FINAL, S,
                                        S * PLAN["t_moe"], keep_optimizer=False)
MOE0 = None

# %% [markdown]
# ## 10b. Phase 2: the dense control
#
# The same dense checkpoint, the same tokens, the same learning-rate schedule, no conversion.

# %%
CTRL_FINAL, SUM_CTRL, LOG_CTRL = train_run("3_dense_control", "phase 2 dense control", DENSE_FINAL, S,
                                           S * PLAN["t_dense"], keep_optimizer=False)

# %% [markdown]
# ## 11. Results
#
# For scale: Session 13's baseline (also FineWeb with an 8K BPE, 16 layers × 320, 50M tokens) ended at a
# validation loss of 5.04. The tokenizer instance, validation slice and model differ, so treat that as a
# rough yardstick, not a like-for-like comparison.

# %%
def val_at(curve, step):
    return float(np.interp(step, curve["val_step"], curve["val_loss"]))

def first_time_below(curve, target, t_offset):
    """Training minutes (from the start of phase 2) at which the val loss first reaches target, or None."""
    for s, v, t in zip(curve["val_step"], curve["val_loss"], curve["val_time"]):
        if v <= target:
            return t / 60
    return None

def descending(curve):
    """Val loss improvement over the second half of a run: > 0 means it was still going down."""
    v = np.array(curve["val_loss"])
    return float(v[len(v) // 2] - v[-1])

RUN_ROWS = [("phase 1: dense", SUM_DENSE), ("phase 2: MoE", SUM_MOE), ("phase 2: dense control", SUM_CTRL)]
log(f"{'run':<24}{'params':>9}{'active':>9}{'steps':>7}{'tokens':>8}{'val start':>11}{'val end':>9}"
      f"{'tok/s':>10}{'minutes':>9}")
for label, s in RUN_ROWS:
    log(f"{label:<24}{s['params'] / 1e6:>8.1f}M{s['active_params'] / 1e6:>8.1f}M{s['steps']:>7}"
          f"{s['tokens'] / 1e6:>7.0f}M{s['val_start']:>11.4f}{s['val_end']:>9.4f}{s['tok_s']:>10,.0f}"
          f"{s['train_min']:>9.1f}" + (f"  (resumed {len(s['resumes'])}x)" if s.get("resumes") else ""))

gap = SUM_CTRL["val_end"] - SUM_MOE["val_end"]
t_reach = first_time_below(LOG_MOE, SUM_CTRL["val_end"], 0)
last = LOG_MOE["load"][-max(1, len(LOG_MOE["load"]) // 5):]
load_end = np.mean(np.array(last), axis=0)                         # [layers, experts]
fair = 1 / N_EXPERTS
dead = int((load_end < 0.1 * fair).sum())
findings = [
    f"**The conversion was seamless.** Held-out loss {CONV['val_dense']:.4f} for the dense model and "
    f"{CONV['val_moe_step0']:.4f} for the MoE made from it, before any MoE training (difference "
    f"{CONV['delta']:+.4f}).",
    f"**The MoE kept training and kept reducing the loss**: {SUM_MOE['val_start']:.4f} → {SUM_MOE['val_end']:.4f} "
    f"over {SUM_MOE['steps']:,} steps; still falling in the second half of the run "
    f"({descending(LOG_MOE):+.4f})." if descending(LOG_MOE) > 0 else
    f"**Check the MoE run**: its val loss did not improve over the second half ({descending(LOG_MOE):+.4f}).",
    f"**Against the dense control** (same start, same tokens, same schedule): MoE {SUM_MOE['val_end']:.4f} vs dense "
    f"{SUM_CTRL['val_end']:.4f}, " + (f"the MoE is {gap:.4f} lower." if gap > 0 else f"the dense model is {-gap:.4f} lower."),
    f"**Cost**: MoE {SUM_MOE['tok_s']:,.0f} tok/s vs dense {SUM_CTRL['tok_s']:,.0f} tok/s "
    f"({SUM_MOE['tok_s'] / SUM_CTRL['tok_s']:.0%} of the dense speed), with {SUM_MOE['active_params'] / 1e6:.1f}M active "
    f"of {SUM_MOE['params'] / 1e6:.1f}M parameters vs dense {SUM_CTRL['params'] / 1e6:.1f}M. "
    + (f"To reach the control's final loss ({SUM_CTRL['val_end']:.4f}) the MoE needed {t_reach:.1f} min of training, "
       f"against the control's {SUM_CTRL['train_min']:.1f} min "
       f"({'faster' if t_reach < SUM_CTRL['train_min'] else 'slower'} at equal wall-clock)." if t_reach is not None else
       "The MoE did not reach the control's final loss, so at equal wall-clock the dense model is ahead."),
    f"**Experts stayed healthy**: over the last 20% of training the busiest expert took "
    f"{load_end.max() / fair:.2f}x its fair share, the quietest {load_end.min() / fair:.2f}x; {dead} of "
    f"{load_end.size} experts were dead (< 10% of fair share); {1 - np.mean(LOG_MOE['kept'][-len(last):]):.2%} of "
    f"expert choices were over capacity.",
]
for f in findings:
    log("-", f.replace("**", ""))

# %%
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
C_DENSE, C_MOE, C_CTRL = "#6b7280", "#d97706", "#2563eb"
ema = lambda x, a=0.2: np.array([v for v in _ema(x, a)])
def _ema(x, a):
    m = None
    for v in x:
        m = v if m is None else a * v + (1 - a) * m
        yield m

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a in ax:
    for curve, c, lab in ((LOG_DENSE, C_DENSE, "phase 1: dense"), (LOG_CTRL, C_CTRL, "dense control"),
                          (LOG_MOE, C_MOE, "MoE (converted)")):
        a.plot(curve["step"], ema(curve["train_loss"]), color=c, alpha=0.3, lw=1)
        a.plot(curve["val_step"], curve["val_loss"], color=c, lw=2, marker="o", ms=3, label=lab + " (val)")
    a.axvline(S, color="k", ls="--", lw=1)
    a.set_xlabel("optimizer step")
ax[0].set_ylabel("loss (nats/token)")
ax[0].set_title("Whole run: faint = train loss (smoothed), bold = held-out loss")
ax[0].annotate("convert to MoE", (S, ax[0].get_ylim()[1]), xytext=(5, -14), textcoords="offset points", fontsize=9)
lo = min(LOG_MOE["val_loss"] + LOG_CTRL["val_loss"])
hi = CONV["val_dense"]
ax[1].set_xlim(S - S * 0.15, 2 * S + S * 0.02)
ax[1].set_ylim(lo - 0.25 * (hi - lo), hi + 0.6 * (hi - lo))
ax[1].set_title("Phase 2 close-up: MoE vs the dense control")
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "loss_curves.png"))
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 4))
for curve, c, lab in ((LOG_CTRL, C_CTRL, "dense control"), (LOG_MOE, C_MOE, "MoE")):
    ax.plot(np.array(curve["val_time"]) / 60, curve["val_loss"], color=c, lw=2, marker="o", ms=3, label=lab)
ax.set_xlabel("training minutes since the conversion")
ax.set_ylabel("held-out loss")
ax.set_title("Phase 2 at equal wall-clock time")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "loss_vs_time.png"))
plt.show()

steps_l = np.array(LOG_MOE["step"])
loads = np.array(LOG_MOE["load"])                                     # [logs, layers, experts]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
im = ax[0].imshow(load_end / fair, cmap="RdBu_r", vmin=0, vmax=2, aspect="auto")
ax[0].set_xticks(range(N_EXPERTS))
ax[0].set_yticks(range(N_LAYER))
ax[0].grid(False)
ax[0].set_xlabel("expert")
ax[0].set_ylabel("layer")
ax[0].set_title("Load ÷ fair share, last 20% of training")
fig.colorbar(im, ax=ax[0], fraction=0.046)
for li in range(N_LAYER):
    ax[1].plot(steps_l, loads[:, li].max(axis=1) / fair, lw=1, alpha=0.8, label=f"layer {li}")
ax[1].axhline(1, color="k", lw=0.8)
ax[1].axvline(S + SAMPLED_STEPS, color="k", ls=":", lw=0.8)
ax[1].annotate("hard top-k from here", (S + SAMPLED_STEPS, 1), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax[1].set_title("Busiest expert ÷ fair share, per layer")
ax[1].set_xlabel("optimizer step")
ax[1].legend(fontsize=6, ncol=2)
ax[2].plot(steps_l, 100 * (1 - np.array(LOG_MOE["kept"])), color=C_MOE)
ax[2].set_title("Expert choices over capacity (%)")
ax[2].set_xlabel("optimizer step")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "experts.png"))
plt.show()

# %% [markdown]
# ### The ledger: what every step trained on, on which chip, and what came back
#
# First, each run shard by shard (from `ledger/<run>.shards.json`). Then a lookup that answers the Session 6
# question, *what did we train on at step X?*: the shard, the chip and batch rows, the hardest and easiest
# sequence in that batch with the web pages they came from, and a **replay check** that re-reads the batch
# from the shard and compares its hash with the one recorded during training. The MoE and the dense control
# are looked up at the same step: same shard, same sequences, different models.

# %%
def read_steps(run):
    with open(os.path.join(LEDGER, f"{run}.steps.jsonl")) as f:
        return [json.loads(line) for line in f]

for run in ("1_dense", "2_moe", "3_dense_control"):
    stats = json.load(open(os.path.join(LEDGER, f"{run}.shards.json")))
    log(f"{run}: {len(stats)} shards")
    log(f"  {'shard':<13}{'steps':>17}  {'device':<8}{'tokens':>9}{'mean loss':>11}{'ppl':>8}")
    for st in stats:
        log(f"  {st['shard']:<13}{st['first_step']:>8} – {st['last_step']:<6}  {','.join(st['devices']):<8}"
            f"{st['tokens'] / 1e6:>8.1f}M{st['mean_loss']:>11.4f}{st['ppl']:>8.1f}")

_docs_cache = {}
def docs_for(shard, sample):
    """The documents (FineWeb id, URL) that make up one training sequence."""
    if shard not in _docs_cache:
        import gzip
        with gzip.open(shard_files(shard)[2], "rt") as f:
            _docs_cache[shard] = [json.loads(line) for line in f]
    lo, hi = sample * SEQ_LEN, sample * SEQ_LEN + SEQ_LEN + 1
    return [d for d in _docs_cache[shard] if d["tok_start"] < hi and d["tok_start"] + d["n_tok"] > lo]

def lookup(run, step):
    recs = read_steps(run)
    rec = recs[step - recs[0]["step"]]
    assert rec["step"] == step
    log(f"{run} @ step {step}: {rec['phase']} | shard {rec['shard']} | lr {rec['lr']:.2e} | loss {rec['loss']:.4f} "
        f"(perplexity {rec['ppl']}) | grad norm {rec['gnorm']}")
    samples = []
    for dev, d in rec["devices"].items():
        log(f"  {dev}: batch rows {d['rows'][0]}–{d['rows'][1] - 1} = sequences {d['samples'][:5]}… of {rec['shard']}")
        samples += d["samples"]
    order = np.argsort(rec["sample_loss"])
    for label, i in (("hardest", order[-1]), ("easiest", order[0])):
        docs = docs_for(rec["shard"], samples[i])
        log(f"  {label} sequence #{samples[i]} (loss {rec['sample_loss'][i]:.3f}) comes from {len(docs)} document(s):")
        for d in docs[:3]:
            log(f"      {d['url'][:100]}   (FineWeb {d['id']}, {d['dump']})")
    same = batch_hash(LOADER.get(step)) == rec["batch_sha1"]
    log(f"  replay: re-reading step {step} from {rec['shard']} gives the recorded batch: {'yes' if same else 'NO'}")
    return rec

probe_step = S + S // 2
r_moe, r_ctrl = lookup("2_moe", probe_step), lookup("3_dense_control", probe_step)
same_data = r_moe["shard"] == r_ctrl["shard"] and r_moe["devices"] == r_ctrl["devices"]
log(f"MoE and dense control at step {probe_step}: same shard and sequences: {'yes' if same_data else 'NO'}; "
    f"loss {r_moe['loss']:.4f} vs {r_ctrl['loss']:.4f}")

# %% [markdown]
# ### What the models write
#
# Samples from the dense model at the end of phase 1, the dense control and the MoE, with the same prompts
# and the same random seed. A ~20M-parameter model trained on web text for a few hundred million tokens
# writes fluent-looking but loosely connected sentences. The point is a sanity check that the MoE is a
# working language model, not just a falling number.

# %%
@functools.partial(jax.jit, static_argnames=("cfg", "n_new"))
def generate(params, bias, toks, start, key, cfg, n_new, temp=0.8, top_k=40):
    def body(i, carry):
        toks, key = carry
        pos = start + i
        logits, _ = forward(params, toks, cfg, bias)
        logits = jax.lax.dynamic_index_in_dim(logits, pos - 1, axis=1, keepdims=False) / temp
        kth = jax.lax.top_k(logits, top_k)[0][:, -1:]
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, jnp.where(logits < kth, -1e30, logits))
        return toks.at[:, pos].set(nxt), key
    return jax.lax.fori_loop(0, n_new, body, (toks, key))[0]

GEN_CFG = dataclasses.replace(CFG, group=SEQ_LEN)       # one group per sequence, so padding never takes a slot
EOS = TOK.token_to_id(EOS_TOKEN)
def sample(model, prompt, n=4, n_new=150):
    ids = TOK.encode(prompt).ids
    n_new = min(n_new, SEQ_LEN - len(ids))
    toks = np.zeros((n, SEQ_LEN), np.int32)
    toks[:, :len(ids)] = ids
    out = np.asarray(generate(put(model["params"]), None if model.get("bias") is None else put(model["bias"]),
                              jnp.asarray(toks), len(ids), jax.random.PRNGKey(7), GEN_CFG, n_new))
    texts = []
    for row in out:
        row = row[: len(ids) + n_new].tolist()
        row = row[: row.index(EOS, len(ids))] if EOS in row[len(ids):] else row
        texts.append(TOK.decode(row))
    return texts

samples = {}
for prompt in ("The best way to learn a new language is", "The city council voted on Tuesday to"):
    for label, model in (("dense, end of phase 1", DENSE_FINAL), ("dense control", CTRL_FINAL), ("MoE", MOE_FINAL)):
        samples[f"{label} | {prompt}"] = sample(model, prompt, n=2)
        log(f"--- {label} ---\n{samples[f'{label} | {prompt}'][0]}\n")
save_json(samples, os.path.join(OUT, "samples.json"))

# %% [markdown]
# ### Save everything
#
# Writes `results.md` (the table and findings, ready for the write-up) and `session14_outputs.zip`
# (console log, ledgers, shard manifests and document indexes, loss curves, summaries, figures, samples,
# plan; not the checkpoints or token shards) to the Drive folder. Send me the zip and I'll build the
# report from it.

# %%
table = ["| run | params | active | steps | tokens | val start | val end | tok/s | train min |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for label, s in RUN_ROWS:
    table.append(f"| {label}{' (resumed ' + str(len(s['resumes'])) + 'x)' if s.get('resumes') else ''} | {s['params'] / 1e6:.1f}M | "
                 f"{s['active_params'] / 1e6:.1f}M | {s['steps']:,} | {s['tokens'] / 1e6:.0f}M | {s['val_start']:.4f} | "
                 f"{s['val_end']:.4f} | {s['tok_s']:,.0f} | {s['train_min']:.1f} |")
total_min = (time.time() - T_START) / 60
record_time("notebook_session", time.time() - T_START)
results = "\n".join([
    "# Session 14: dense → MoE (FineWeb sample-10BT)", "",
    f"Hardware: {jax.device_count()} x {DEVICE_KIND}, bf16 matmuls. Plan: batch {B} x {SEQ_LEN}, {S} steps per phase, "
    f"warm-up {WARMUP} steps. This session: {total_min:.0f} min.", "",
    *table, "", *[f"{i + 1}. {f}" for i, f in enumerate(findings)], "",
    "![loss curves](figures/loss_curves.png)", "", "![loss vs time](figures/loss_vs_time.png)", "",
    "![experts](figures/experts.png)", ""])
with open(os.path.join(OUT, "results.md"), "w") as f:
    f.write(results)
save_json({"plan": PLAN, "sched": SCHED, "conversion": CONV, "dense": SUM_DENSE, "moe": SUM_MOE, "control": SUM_CTRL,
           "timings_s": TIMINGS, "findings": findings}, os.path.join(OUT, "summary.json"))

stage = os.path.join(LOCAL, "session14_outputs")
shutil.rmtree(stage, ignore_errors=True)
shutil.copytree(OUT, stage, ignore=shutil.ignore_patterns("*.pkl", "*.tmp", "*.bin", "*.zip", "_local"))
zip_path = shutil.make_archive(os.path.join(OUT, "session14_outputs"), "zip", stage)
log(f"wrote {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
log(f"notebook time this session: {total_min:.1f} min")
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    pass
