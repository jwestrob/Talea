# Talea performance branch — implementation and concordance record

**Branch:** `perf/vectorize-and-selective-attention`
**Base:** `ace799b` (github.com/jwestrob/Talea, `master`)
**Tree:** `/groups/banfield/projects/environmental/sr/srvp2020/Jacob/Talea/Talea_dev`
**Reference tree (untouched):** `../Talea_repo`
**Date:** 2026-07-28

Written to be quotable in a methods section. Every performance claim below is a
measurement from a script in `../perf/`, not an estimate; every numerical claim
is an assertion in `tests/test_concordance.py`.

---

## Headline

With `--capture-mode manual` (the default), this branch is **bit-identical** to
the reference implementation: a three-record CPU/fp32 run produces a prediction
JSONL with the same SHA-256,
`5de3114db73386e0db41451ff267a375d63b1317840052edc71ecdc04dda3729`.

Measured gains:

| Stage | Reference | This branch | Factor |
|---|---|---|---|
| CPU postprocess, L=400 | 1.33 s | 0.63 s | 2.12× |
| CPU postprocess, L=1000 | 3.31 s | 1.33 s | 2.48× |
| CPU postprocess, L=2046 | 7.93 s | 3.68 s | 2.16× |
| `best_harmonic_comb`, L=1024 | 26.1 ms | 1.65 ms | 15.9× |
| Peak forward VRAM, L=2046 (H200, measured) | 5.78 GB | 179 MB | **32.2×** |
| Peak forward+matrix VRAM, L=2046 | 6.06 GB | 453 MB | 13.4× |
| **End-to-end throughput, 96 records (H200, 16 CPU)** | **1.01 prot/s** | **4.77 prot/s** | **4.7×** |

End-to-end (job 1161569), 96 records with a GVDB-like length mix, H200 +
16 CPUs, model-load time excluded:

| configuration | prot/s | vs reference |
|---|---:|---:|
| reference, thread × 4 (old default) | 1.01 | 1.00× |
| dev, thread × 1 | 2.66 | 2.63× |
| dev, process × 2 | 2.91 | 2.88× |
| dev, process × 4 | 3.21 | 3.18× |
| dev, process × 8 | 3.54 | 3.51× |
| dev, process × 16 | 3.22 | 3.19× (oversubscribed: 16 workers + parent on 16 CPUs) |
| dev, process × 8, `--square-budget 16e6` | 4.72 | 4.67× |
| dev, process × 8, `--square-budget 64e6` | 4.77 | **4.72×** |

Two things to read off this. Pool scaling saturates around 8 workers and then
regresses — partly oversubscription, partly Amdahl: the teacher stage runs
serially in the parent and is 12–16% of per-record CPU time, capping pool
speedup near 6×. And **raising `square_budget` is worth 1.33×** (3.58 → 4.72)
once postprocessing is parallelized — which corrects an earlier assessment in
this document that it "buys little"; that was true only while the pipeline was
serial. The freed VRAM is what makes the larger budget affordable.

Caveat on the benchmark set: these are random sequences, which produce fewer
candidate periods and so less wedge/registration work than genuine repeat
proteins. Real solenoid-rich input will spend proportionally more time in
postprocessing, which should *favour* the parallel configurations further.

---

## GPU measurements (H200, job 1161567)

`torch.cuda.max_memory_allocated` over the ESM++ forward, batch of 1, fp16:

| L | reference | manual | reduction |
|---:|---:|---:|---:|
| 256 | 116.1 MB | 9.0 MB | 12.8× |
| 512 | 414.4 MB | 16.8 MB | 24.7× |
| 1024 | 1500.0 MB | 53.0 MB | 28.3× |
| 1732 | 4199.3 MB | 133.0 MB | 31.6× |
| 2046 | 5780.8 MB | 179.3 MB | 32.2× |

The predicted attention-tuple saving was 5.06 GB; the measured forward-peak drop
at L=2046 is 5.60 GB, the balance being the hidden-state tuple and the manual
kernel's transient score matrices.

**The forward is not the bottleneck.** At L=2046 the forward takes 0.038 s
against ~3.7 s of CPU postprocess — the pipeline is roughly **99% CPU**. Two
consequences: (i) `sdpa` mode's 1.53× on the forward (0.038 s → 0.025 s) is worth
well under 1% end to end, and (ii) raising `square_budget` to exploit the freed
VRAM buys little *while the pipeline is serial* — but 1.33× once postprocessing
is parallelized, as the throughput table above shows. The VRAM headroom is what
makes that larger budget affordable, so its value is both OOM headroom at the
length ceiling and a real throughput lever in the parallel configuration.

**GPU concordance.** A four-record CUDA/fp16 CLI run is byte-identical between
trees under `manual`: sha256 `0451e1e24c948930a38b9b5701dfe3ba0cfdee0db0cf2131b59c64a006725b7b`.
That result does **not** generalize to larger runs — see the next section.

---

## GPU reproducibility: a property of Talea, not of this branch (job 1161570)

A 96-record run showed every configuration producing a different SHA-256,
including two runs of an identical configuration. The obvious reading — that
this branch broke something — is wrong. The control settles it:

**Three runs of the unmodified `Talea_repo`, identical flags, identical input:**

| comparison | records differing | worst relative delta |
|---|---:|---:|
| ref run1 vs run2 | 54/96 | 3.81e-2 (`teacher_state_z`) |
| ref run1 vs run3 | 65/96 | 3.82e-2 |
| **dev run1 vs run2** | 83/96 | **3.76e-3** |
| ref run1 vs dev run1 | 73/96 | 3.96e-2 |

**The reference implementation is not reproducible on GPU.** Same code, same
input, same flags, three runs, three different outputs, disagreeing by up to
~3.8% on `teacher_state_z`. The cross-tree disagreement (3.96e-2) is the same
order as the reference's disagreement with itself (3.82e-2), so at this sample
size the two trees are indistinguishable from run-to-run noise.

Two corollaries worth attention independently of any optimization:

1. **Dev appears *more* stable than reference** — 3.76e-3 vs 3.81e-2, an order
   of magnitude tighter, though from only two runs. A plausible mechanism is C8:
   the reference calls `torch.cuda.empty_cache()` twice per batch, churning the
   allocator and so the cuBLAS workspace state that kernel selection depends on.
   Removing it may have removed a nondeterminism source. Not yet confirmed.

2. **Results depend on batch composition.** Same tree, `--square-budget` 4e6 vs
   1.6e7: **96/96 records differ, 11 structurally** (different registration unit
   counts), worst relative delta **11.94**. Batching is not a performance-only
   knob — it changes reported values. This has direct consequences for the
   sharded screens: a 64-shard run and a single-pass run over the same proteins
   will not agree, and neither will two shardings.

**What this means for the concordance claims here.** CPU/fp32 is deterministic
(verified: `Talea_repo` run twice, identical), so it is the correct environment
for bit-exactness validation, and there the branch is byte-identical. On GPU,
byte-identity is not achievable by anyone — including the reference against
itself — so the honest claim is bounded: *the difference between trees is the
same order as the reference's own run-to-run variation.*

**What this means for the science.** Published Talea scores are not exactly
reproducible from a GPU re-run, and score≥2 calls sitting within ~4% of the
threshold could flip between runs. Rank order should be largely stable, but that
is an assumption worth testing rather than assuming. The deployment contract's
"sealed_replay" language should be read against this.

### Localizing it (jobs 1161571–1161576)

Ruled out, each by direct measurement rather than reasoning:

| Hypothesis | Test | Result |
|---|---|---|
| fp16 split-k GEMM reductions | `allow_fp16_reduced_precision_reduction=False` | still varies |
| TF32 / cuDNN algorithm choice | `allow_tf32=False`, `cudnn.deterministic=True` | still varies |
| Within-process instability | 3 passes in one process (job 1161572) | **bit-identical** |
| ESM++ forward across processes | fixed batch, 3 processes (job 1161573) | **bit-identical**, incl. raw attention, pooled hidden, and the APC matrices |
| CPU postprocess across processes | 3 processes × 2 thread settings | **bit-identical** |

So the ESM++ forward, the matrix construction, and the entire CPU postprocess
are all deterministic — within a process and across processes.

**Confirmed cause (job 1161576): the teacher UNet.** Comparing three separate
processes on one H200, every ESM++ tensor is bit-identical while the teacher
output is not:

| tensor | across processes |
|---|---|
| `last_hidden`, `raw_attn13/35`, `hidden13/35` | identical, max\|Δ\| 0 |
| APC matrices `mat{0,1,2}_{13,35}` | identical, max\|Δ\| 0 |
| **`teacher_state{0,1,2}`** | **differ, max\|Δ\| 1.4e-4 – 2.7e-4** |
| **`teacher_class{0,1,2}`** | **differ, max\|Δ\| 1.9e-4 – 3.8e-4** |

`torch.conv2d` dispatches to cuDNN, which selects an algorithm per process;
different algorithms accumulate differently. ~2e-4 of probability then becomes
~3.8e-2 of `teacher_state_z`, because `standardized_probability_z` applies a
logit and divides by a control standard deviation — a large amplification of a
small perturbation.

**Mitigation, and its limit.** With `torch.use_deterministic_algorithms(True)`
plus `cudnn.deterministic=True` and a fixed cuBLAS workspace, two of the three
records became bit-identical across processes and the third shrank from 2.7e-4
to 5.2e-6 — roughly 50× tighter, but still not exact. Exposed as
**`--deterministic`** (C10), opt-in: it changes results relative to every run
made so far, including the frozen artifacts, which were themselves produced
without it.

Note the practical asymmetry this already establishes: **a single run is
internally consistent**, so within-run comparisons and rankings are sound. It is
*re-running* that does not reproduce. For the 64-shard GVDB screen, each shard
is a separate process, so shard boundaries are also potential discontinuities.

---

## Changes

### C1. Selective attention/hidden-state capture — `talea/capture.py` (new)

**Problem.** `iter_predictions` requested `output_attentions=True`. In the pinned
ESM++ revision (`modeling_esm_plusplus.py`) this has two effects. First,
`MultiHeadAttention._attn` short-circuits to `_manual_attn` whenever the flag is
set, so all 36 layers abandon the configured SDPA backend. Second,
`TransformerStack.forward` accumulates every layer's weights into a tuple held
live for the whole pass — at L=2046 with 18 heads in fp16 that is 144 MB/layer,
5.06 GB total, of which Talea uses two layers (288 MB). `output_hidden_states`
likewise retains 37 tensors where 2 are used.

**Change.** Per-module patches capture exactly the needed tensors; the stack is
called with both flags `False` so it accumulates nothing. Two modes:

- `manual` (default) — every block keeps `_manual_attn`, so the residual stream
  is unchanged and only the *retention* is removed. Bit-identical.
- `sdpa` (opt-in) — only the two captured blocks use `_manual_attn`; the other
  34 dispatch to SDPA. Faster, not bit-identical (see C1-risk).

**Index resolution (important, and counter-intuitive).** `hidden_states` for
this model does **not** begin with the embedding: the prepend in
`TransformerStack.forward` is gated on `esmfold2_hidden_states`, which Talea
never sets. So `hidden_states[i]` is block `i`'s output and `hidden_states[36]`
is the final `norm(x)`. Because `production.pooled_hidden_features_for_record`
reads `hidden_states[layer + 1]`, the frozen global head is pooled from **block
14's output for "layer 13"**, and from **`last_hidden_state` for "layer 35"**.
That off-by-one is part of the frozen feature contract; it is reproduced, not
corrected. `resolve_hidden_state_sources((13, 35), 36) == {13: 14, 35: None}` is
asserted in tests and verified against the model's own tuples in
`perf/validate_capture.py`. Attention indexing has no such offset —
`attentions[i]` is block `i`.

**C1-risk — `sdpa` mode is not recommended, and the reason is not the score.**
Changing the kernel for the 34 uncaptured layers perturbs their outputs by
~1e-6 (attention) and ~4e-5 (hidden). On the primary score that is invisible:
`talea_discovery_score` moves in the 5th–6th decimal, far below any threshold.

But the *auxiliary* outputs are not that stable. On a GPU/fp16 run over four
records (job 1161567), the `ank_like_repeat` record — a 29-aa ankyrin motif
repeated eight times, the most biologically realistic case in the set — changed
qualitatively:

| field | reference | sdpa |
|---|---:|---:|
| `talea_discovery_score` | 2.9854223191 | 2.9856440605 |
| `registration.n_units` | **6** | **4** |
| `registration.period` | **42.5** | **63.5** |
| `registration.phase_offset` | 31.0 | 51.0 |
| `wedge_call.period` | 42.5 | 63.5 |
| `wedge_call.score` | 12.78 | 10.47 |

A 1e-6 perturbation selected a different period candidate and therefore a
different repeat registration.

**Corrected 2026-07-29 — sdpa is not measurably less accurate.** The paragraph
above was written from that single 4-record run and over-read it. A 120-record
benchmark against constructed ground truth (job 1161577, see
`../perf/PERIOD_RECOVERY_BENCHMARK.md`) puts the two modes level:

| mode | exact period | 2× alias | median \|log2 err\| |
|---|---:|---:|---:|
| manual | 62/120 | 41 | 0.044 |
| sdpa | 59/120 | 42 | 0.074 |

Head to head: manual closer on 3, sdpa closer on 3, **tied on 114**. The
ankyrin case was one draw from a noisy process, not a demonstration that sdpa
degrades calls.

`manual` remains the default, but for **reproducibility, not accuracy**: it is
bit-identical to the frozen reference and sdpa is not, and sdpa's forward
speedup is worth <1% end to end, so there is nothing to trade for.

What that benchmark did establish is a property of Talea itself, present equally
in the reference: **the period call aliases upward in ~34% of cases** (41/120
exactly 2×, 0/120 at 0.5×), which halves the reported unit count. It is driven
by array length and, counterintuitively, is *worse* for low-divergence repeats.
See the benchmark note; it bears on period- and unit-derived analyses regardless
of which capture mode is used.

### C2. `diagonal_mean_profile` — `talea/attention.py`

**Problem.** A Python loop calling `np.diagonal` + `np.mean` once per offset —
49,590 `diagonal` and 52,200 `mean` calls for a single L=1000 protein, 38% of
postprocess time.

**Change.** Element `(i, i+k)` of a row-major L×L array lies at flat index
`i*(L+1)+k`, so padding to `L*(L+1)` and reshaping to `(L, L+1)` places diagonal
`k` in column `k`, valid over its first `L−k` rows; one `cumsum` down the columns
yields every diagonal sum.

**Gated at L ≤ 1024** (`DIAGONAL_FAST_PATH_MAXIMUM`). Above that the `(L, L+1)`
working grid stops fitting in cache and the vectorized form is *slower* — 0.9×
at L=2046 — so the original loop is retained there. Since analysis windows are
≤512, nearly all calls take the fast path. Measured: 21.7× (L=64), 15.9× (L=256),
3.1× (L=512), 3.0× (L=1024).

**Numerics.** Bit-exact at every size tested.

### C3. `best_harmonic_comb` — `talea/physical.py`

**Problem.** Doubly-nested Python loop, 42,412 `np.max` calls per L=1000
protein on 3-element slices; 39% of postprocess time.

**Change.** The tolerance-window maxima become one linear-time sliding maximum
of width `2·tolerance_bins+1` plus a single gather.

**The per-fundamental reduction is deliberately left as a loop.** This is the
subtle part and the one most worth recording. `np.dot` uses BLAS `ddot`, whose
SIMD-paired summation order no vectorized NumPy reduction reproduces: measured
against `np.dot` over 200,000 random inputs per length, `np.sum(a*b)` and
`np.einsum` differ in the last bit for 17–48% of cases at k ≤ 8, and a grouped
mat-vec differs by up to 5.3e-15. A first implementation used `einsum` and was
"within 1 ulp" on smooth synthetic spectra — but real APC attention spectra are
spiky and fp16-quantized, so **exact ties between adjacent fundamentals are
common**, and a last-bit difference flips the tie, changing the selected period
and shifting downstream scores by ~1e-3. Slicing a row of the gathered array is
bit-identical to `np.dot` on a contiguous copy, so the retained loop reproduces
the reference exactly.

Keeping exactness costs speed: 7.0–15.9× (vs 8.6–59× for the inexact `einsum`
form). That trade is taken deliberately.

Tie-break order is also reproduced: the reference replaces its incumbent only on
a *strict* improvement in `(score, n_harmonics)`, so ties resolve to the lowest
fundamental bin.

**Numerics.** Bit-exact, including on a 300-case tie-stress suite built from
quantized synthetic attention spectra.

### C4. `postprocess_workers` default 4 → 1 — `talea/cli.py`, `talea/inference.py`

**Problem.** The postprocess thread pool anti-scales. Postprocessing is tens of
thousands of small NumPy calls that hold the GIL, so threads convoy. Measured
(8 proteins, L=800, dev tree, 209 idle cores available):

| workers | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| scaling | 1.00× | 0.73× | 0.52× | 0.44× |

On the reference tree the same shape held (1.00 / 0.63 / 0.57). The previous
default of 4 was running ~1.9× slower than 1.

**Change.** Default 1. Structural fix (a `ProcessPoolExecutor`) is **not**
implemented here and remains open — see "Not done".

**Numerics.** None; worker count cannot affect per-record arithmetic.

### C5. Cached near-diagonal mask — `talea/inference.py`

`apc_head_mean` rebuilt `(positions[:,None] - positions[None,:]).abs()` per call
— two int64 L×L temporaries, 64 MB at L=2046, twice per record. Now cached on
`(length, separation, device)`. Bit-exact (asserted against the inline
construction).

### C6. Crop before symmetrize — `talea/inference.py`

`apc_head_mean` symmetrized the padded T×T then cropped to L×L. Both operations
are elementwise over the same index set, so cropping first is identical while
every subsequent reduction runs over L×L. Bit-exact (asserted at 3 padding
configurations × fp32/fp16).

### C7. Device-side fp16 cast — `talea/inference.py`

The storage contract is already fp16 (`decoder_matrix_storage_dtype`), but the
matrix was transferred as fp32 and quantized on the host. Casting first halves
the transfer (16 MB → 8 MB per matrix per layer at L=2046). Bit-exact — asserted
equal to the host-side round trip, since both are IEEE round-to-nearest-even.

### C9. Process-based postprocessing — `talea/postprocess_pool.py` (new)

**Problem.** C4 works around the GIL convoy by defaulting to one worker; it does
not let Talea use more than one core for the ~99% of wall time spent in
postprocessing.

**Change.** `--postprocess-mode process` runs postprocessing in a
`ProcessPoolExecutor`. Two details make the difference between a speedup and a
slowdown, and both are recorded because they are easy to reintroduce:

1. **The worker module must not import torch.** The first implementation put the
   worker entry point in `talea.inference`, whose cold import costs **16.06 s**
   (torch + transformers) against **1.13 s** for `talea.topology`. Every worker
   paid it, and the pool ran at **0.27×** of serial. Moving the entry point to
   the torch-free `talea/postprocess_pool.py` and preloading it took the same
   benchmark to **3.46×** on four workers. `tests/test_concordance.py` asserts
   the module's import graph stays clean.
2. **`forkserver`, not `spawn` or `fork`.** The forkserver is launched as a fresh
   interpreter, so workers never inherit the parent's CUDA context (plain `fork`
   after CUDA init is unsafe), and preloading the worker module means each worker
   forks from it almost for free. The preload list is deliberately *not*
   `['__main__']` — that is multiprocessing's default and it breaks any caller
   whose `__main__` is not importable, e.g. a REPL or a heredoc.

**Numerics.** Bit-exact — worker output is asserted identical to a direct
`process_attention_record` call, and the pooled result identical to serial.

### C10. `--deterministic` — `talea/inference.py`, `talea/cli.py`

Pins cuDNN to deterministic convolution algorithms, addressing the teacher-UNet
nondeterminism established above. Opt-in; see the GPU reproducibility section
for the measurement and the residual 5.2e-6 it does not remove.

### C8. `empty_cache()` / `gc.collect()` no longer per batch — `talea/inference.py`

Both ran unconditionally every batch (`empty_cache` twice). `empty_cache` returns
blocks to the driver and forces later `cudaMalloc`; `gc.collect()` walks every
generation. Now governed by `--cache-clear-interval` (default 0 = never), which
is safe once the 36-layer attention tuple is gone. No numerical effect.

---

## Concordance protocol

Five layers, all reproducible from `../perf/`:

| Layer | What | How | Result |
|---|---|---|---|
| L1 | replaced kernels vs in-tree originals | `tests/test_concordance.py`, 600 randomized spectra + 300 tie-stress + 11 matrix sizes | bit-exact |
| L2 | `process_attention_record`, every emitted numeric field | same, 3 lengths | bit-exact |
| L3 | `apc_head_mean` crop/cast/mask | same, padded × dtype grid | bit-exact |
| L4 | capture vs the model's own tuples | `perf/validate_capture.py`, real ESM++ forward | bit-exact (manual) |
| L5 | full CLI, both trees | `perf/validate_endtoend_cli.py`, 3 records | byte-identical JSONL |

The originals are kept **in-tree** as `_diagonal_mean_profile_reference` and
`_best_harmonic_comb_reference` rather than in the tests, so the optimized paths
always have an oracle and the replaced behaviour is documented where it lives.

**Controls that mattered:**

- *Reference reproducibility.* Before attributing any drift to this branch,
  `Talea_repo` was run twice on identical input and confirmed to produce
  identical scores. Without that control an observed difference cannot be
  assigned to a code change.
- *Matrix-level bisection.* When L5 first showed a 1e-3 drift, `perf/dump_matrices.py`
  compared all attention matrices and pooled hidden states between trees and
  found them bit-identical, localizing the drift to the CPU postprocess rather
  than the capture change. That is what exposed the `ddot` tie-break issue in C3.
- *A filename collision produced one wrong intermediate result.* Both dev CLI
  runs initially wrote to the same path, so the `sdpa` output overwrote the
  `manual` output and a post-hoc analysis read the wrong file. Fixed by naming
  outputs `<tree>-<mode>.jsonl`. Noted because the erroneous numbers appear in
  the session log.

Test-suite status: 73 pre-existing tests pass unchanged on both trees; 36
concordance tests added (109 total on the dev tree).

---

## Not done / open

- **`ProcessPoolExecutor` for postprocessing.** The GIL convoy (C4) is worked
  around, not fixed. Processes are the real fix — matrices are 8–16 MB, cheap
  against seconds of compute — and would make multi-core postprocessing actually
  scale. This is likely the single largest remaining throughput win.
- **`square_budget` retuning.** The default 4e6 against T²=4.19e6 forces batch
  size 1 at maximum length. That default was set against attention memory which
  C1 mostly eliminates, so it can likely be raised substantially. Not retuned
  here because it needs GPU measurement.
- **GPU-side measurement.** All timings here are CPU. The VRAM figures are
  arithmetic from the model config (36 layers, 18 heads, hidden 1152), not
  `nvidia-smi` readings, and the SDPA-mode speedup is unquantified. A GPU run is
  needed for both.
- **Prefix-sum hoisting in `wedge_start_scores`.** Recomputes `np.cumsum` per
  (period × width × separation) though prefix sums depend only on separation.
  Profiling showed this is currently negligible (<0.3 s of 5.1 s) because few
  candidate periods survive; worth hoisting only if a worst-case protein shows up.
- **Duplicate SHA-256.** Sequences are hashed at parse (`io.py`) and again at
  validation (`inference.py`). Left alone; the validation is a deliberate
  integrity check.
- **Streaming input.** `read_fasta` returns a list and `iter_predictions`
  re-materializes it plus pre-builds every batch, so a whole shard is resident
  before any GPU work starts.

---

## Reproducing

```bash
K=/groups/banfield/scratch/projects/human/nano/int/kinsey/projects/2026_07_21_Talea_solenoid_mining
T=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/Talea

# unit + concordance (no GPU, no model weights)
cd $T/Talea_dev && PYTHONPATH=. $K/.venv/bin/python -m pytest tests -q

# capture vs the model's own tuples (needs the cached ESM++ weights)
cd $T/perf
HF_HOME=$K/hf_cache HF_HUB_OFFLINE=1 $K/.venv/bin/python validate_capture.py

# full CLI, both trees, byte comparison
$K/.venv/bin/python validate_endtoend_cli.py
```
