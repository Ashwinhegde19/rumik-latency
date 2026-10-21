# rumik-oss-1 streaming latency

Time-to-first-audio (TTFA) and streaming throughput for
[`rumik-ai/rumik-oss-1`](https://huggingface.co/rumik-ai/rumik-oss-1)
(revision `0ed3c98684e14350c910129c0efa179841c41ad2`) on one Modal H100: the stock `transformers` baseline,
and an optimized streaming path (`rumik_fast.py`, benchmarked by `bench_fast.py`). The baseline code path
(`bench_modal.py`) is unchanged.

## Before / after (H100 80GB HBM3 SXM, warm, 10 texts x 10 runs)

| | baseline (stock HF) | optimized | change |
|---|---:|---:|---:|
| TTFA p50 / p95 (ms) | 158.3 / 182.2 | **32.1 / 32.5** | 4.9x |
| prefill (ms, p50) | 19.8 | 4.8 | |
| first 8 tokens complete (ms, p50) | 144.9 | 27.3 | |
| first-frame Mimi decode (ms, p50) | 13.0 | 4.5 | |
| first non-silent audio (ms, p50) | 599 | 110 | 5.4x |
| decode tok/s (steady state, p50) | 55.1 | **301** (290 on long texts) | 5.5x |
| per-token time (ms) | ~18 | 3.3 | |
| real-time factor (audio s / wall s, p50) | 0.55 | **2.84** | 5.2x |
| streamed playback without underrun | no (0.55x real time) | 100/100 runs | |
| Mimi streaming decode, steady state (ms per 80 ms frame) | n/a (11 ms per call, no streaming state) | 4.5 (concurrent with LM), 2.1 alone | |
| peak GPU memory allocated | 7.9 GB | 16.0 GB (stock model kept loaded for the gates, plus fused weight copies) | |

Baseline RTFx is the model-card path (`generate_audio` + full Mimi decode); optimized RTFx is the whole
streaming request (request start until every LM step, including run-ahead past the stop, and the last Mimi
frame are done). "Streamed playback without underrun": playback starts when frame 0 is ready, and every later
frame is ready before its playback time (max lateness 0 ms in all 100 runs).

Per text (optimized, p50 unless noted):

| text | TTFA p50 (ms) | TTFA p95 (ms) | first non-silent audio (ms) | audio (s) | request wall (s) | RTFx | tok/s | Mimi ms/frame | stall-free |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en_short | 31.8 | 32.4 | 110.1 | 2.9 | 1.1 | 2.7 | 304.9 | 4.5 | 10/10 |
| en_medium | 32.1 | 32.4 | 110.4 | 5.5 | 1.9 | 2.8 | 301.9 | 4.5 | 10/10 |
| en_long | 32.1 | 32.3 | 110.5 | 11.3 | 4.0 | 2.9 | 294.2 | 4.4 | 10/10 |
| hinglish_short | 32.2 | 32.3 | 110.4 | 3.6 | 1.3 | 2.8 | 304.1 | 4.5 | 10/10 |
| hinglish_medium | 32.1 | 32.2 | 110.4 | 6.5 | 2.3 | 2.9 | 300.3 | 4.4 | 10/10 |
| hinglish_long | 32.4 | 33.1 | 110.8 | 13.6 | 4.8 | 2.8 | 290.0 | 4.4 | 10/10 |
| hi_short (Devanagari) | 32.0 | 32.2 | 110.2 | 3.4 | 1.2 | 2.8 | 304.3 | 4.5 | 10/10 |
| hi_medium (Devanagari) | 32.1 | 32.3 | 110.1 | 6.0 | 2.1 | 2.9 | 301.1 | 4.5 | 10/10 |
| hi_long (Devanagari) | 32.2 | 32.8 | 110.5 | 14.8 | 5.3 | 2.8 | 289.0 | 4.4 | 10/10 |
| hi_en_mixed_medium | 31.9 | 32.3 | 110.3 | 5.9 | 2.1 | 2.9 | 301.1 | 4.5 | 10/10 |
| **all (n=100)** | **32.1** | **32.5** | **110.4** | 5.9 | 2.1 | **2.84** | **301.1** | 4.5 | 100/100 |

## Step-by-step (ablation, en_medium + hi_medium, 10 runs each)

| step | TTFA p50 / p95 (ms) | prefill (ms) | first 8 tokens (ms) | first-frame decode (ms) | tok/s |
|---|---:|---:|---:|---:|---:|
| 0. baseline (stock HF loop, HF Mimi fp32) | 158.3 / 182.2 | 19.8 | 144.9 | 13.0 | 55 |
| 1a. FastLM eager: static KV cache, audio-only head, on-GPU sampling/stop, no per-step sync | 224.0 / 257.6 * | 14.0 | 106.3 | 14.1 | 77 |
| 1b. + CUDA graphs (whole step captured) | 83.9 / 87.3 | 7.5 | 68.5 | 15.1 | 115 |
| 1c. + torch.compile (inductor) inside the graphs | 69.6 / 71.0 | 7.5 | 53.2 | 16.2 | 153 |
| 1d. + KV-length buckets (attend to 256/512/1024/2304 slots, not always 2304) | 48.1 / 49.1 | 7.4 | 32.2 | 15.7 | 260 |
| 1e. + fp32-accumulating reduction attention instead of SDPA mem-efficient kernel | 46.5 / 48.1 | 7.5 | 30.0 | 16.3 | 307 |
| 1f. + compiled prefill (prompt buckets 32/64/128/256) | 43.2 / 44.2 | 4.7 | 27.2 | 15.7 | 307 |
| 2. + streaming Mimi (graph-captured, own stream, overlaps LM) | **32.0 / 32.3** | 4.7 | 27.2 | 4.5 | 302 |
| 3. weight-only int8 / fp8 | not needed (skipped) | | | | |

\* Step 1a is CPU-bound (about 13 ms per step to issue eager kernels). The driver enqueues the next
8-token chunk before decoding the first frame, which adds a whole chunk of CPU time to its TTFA. Its tok/s
is the meaningful number. Steps 1a-1f decode the first frame with HF `MimiModel.decode`, as the baseline does.

Where the time went: a kernel profile of 1c showed 3.4 of 6.4 ms per step in PyTorch's mem-efficient
attention kernel. Batch 1, 4 KV heads and a 4-row query mean only 4 thread blocks, each scanning the
whole 2304-slot static cache. Buckets plus a reduction-style attention bring attention to about 0.1 ms. The
final step is about 3.05 ms of GPU time: about 2.5 ms of GEMV at about 2.5 TB/s (gate/up 1.26 ms,
down 0.65 ms, qkv+o 0.46 ms), the rest small fused kernels. bf16 weights are now close to the HBM bandwidth
floor, so int8 or fp8 would be the next lever (see "What remains").

## Method (optimized path)

`rumik_fast.py`:

- **FastLM** re-implements the Cohere2 decoder forward for batch 1, using the HF module weights unchanged.
  It matches HF numerics: fp32 LayerNorm, interleaved (GPT-J) RoPE in fp32 on the sliding-window layers only,
  NoPE global layers, parallel attention + MLP, and the final norm feeding both the head and the stop
  predictor (HF's `hidden_states[-1]` is post-norm). Other details:
  - Static KV cache of 2304 slots. This is below the 4096 sliding window, so sliding and global layers see
    identical keys, exactly as the stock DynamicCache does for these lengths.
  - Fused qkv and gate/up weights.
  - The LM head is restricted to the 16,385 audio ids (16,384 codec ids plus `</audio>`). The stock loop
    masks every other id to -inf anyway, so this is exact: no codebook-specific pruning.
  - Sampling is the same as `_constrained_sample`: temperature, top-k 30 over the same ids, `</audio>`
    masked for the first 8 steps, and the stop head checked before sampling at step >= 8.
  - Sampling also consumes random numbers the same way: `torch.multinomial(p, 1)` draws `Exp(1)` noise
    over the full vocabulary and takes `argmax(p / q)`.
  - The sampled token and the stop flag go to a device buffer, so there is no per-token host sync.
  - One CUDA-graph step is "forward previous token + sample next", so the first frame needs prefill + 7
    steps. The prefill graph includes sampling token 0.
- **StreamingMimi** decodes one codec frame (8 codes) into 1920 samples with explicit state: causal-conv
  history, transposed-conv overlap-add tails, and a KV cache for the 8-layer decoder transformer (HF Mimi
  uses plain causal attention there). The whole frame step is one CUDA graph. With TF32 off, the eager
  version matches `MimiModel.decode` of the full clip at 128 dB SNR (max abs diff 4e-7), so it is exact.
- **StreamingTTS** is the request driver. The LM runs ahead on the default stream in 8-token chunks, 2
  chunks ahead, with one pinned-memory copy and one event per chunk. The host checks each chunk once for
  stop or `</audio>`, assembles frames with the same resync rules as `audio_tokens_to_codes`, and replays
  the Mimi graph on a second stream.
- **Capture streams matter.** The LM and Mimi graphs must be captured on different streams. PyTorch keys
  cuBLAS workspaces by stream. Captured on the shared default capture stream, both graphs baked in the same
  workspace and corrupted each other when replayed concurrently: streamed audio was only 22-39 dB SNR
  against the full decode.

Measurement follows the baseline rules. Everything runs server-side in one `modal run` container, and
model load, graph capture, compile and 3 warm-ups happen before timing. The request starts before
tokenization. TTFA is when the first 1920-sample PCM chunk is on the GPU, taken with `perf_counter` after
the host syncs on that chunk's event. Per-chunk and per-frame times come from CUDA events on the same
timeline. "First non-silent audio" is the ready time of the first frame above -40 dBFS. tok/s is the
steady-state rate over complete chunks after the first.

## Quality gates (final configuration `c_kvb_f32_pf`)

**(a) Teacher-forced logit agreement vs the stock HF model.** The token sequence is the stock model's own
seeded generation (seed 1000). Logits are compared over the 16,385 audio ids at every position. The
reference is the stock incremental loop (DynamicCache, as `generate_audio`). The "noise floor" row compares
two stock paths with each other: a single full-sequence forward vs incremental decode.

| text | positions | top-1 agree | mean KL (T=1) | mean KL (T=0.8) | max KL | top-30 overlap | stop decision agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| en_medium: optimized | 513 | 96.7% | 6.8e-4 | 9.4e-4 | 4.7e-3 | 98.9% | 100% |
| en_medium: stock noise floor | 513 | 96.9% | 5.6e-4 | 7.7e-4 | 3.3e-3 | 98.9% | 100% |
| hinglish_medium: optimized | 721 | 97.1% | 8.2e-4 | 1.2e-3 | 4.7e-3 | 99.0% | 100% |
| hinglish_medium: stock noise floor | 721 | 96.8% | 6.6e-4 | 9.5e-4 | 4.1e-3 | 99.0% | 100% |
| hi_medium: optimized | 649 | 98.2% | 7.4e-4 | 1.0e-3 | 3.3e-3 | 98.8% | 100% |
| hi_medium: stock noise floor | 649 | 97.4% | 6.4e-4 | 9.0e-4 | 2.9e-3 | 98.9% | 100% |
| hi_en_mixed_medium: optimized | 625 | 97.3% | 7.3e-4 | 1.0e-3 | 3.9e-3 | 99.0% | 100% |
| hi_en_mixed_medium: stock noise floor | 625 | 97.6% | 6.2e-4 | 8.7e-4 | 5.7e-3 | 99.0% | 100% |

The optimized path sits at 1.1-1.25x the KL between two stock code paths, with the same top-1 and top-30
agreement. Step 1c (SDPA over the full static cache, eager prefill) is exactly at the noise floor
(6.0-6.4e-4), but it runs at 153 tok/s with 70 ms TTFA. Every bucketed variant lands at 6.8-8.2e-4:
SDPA, bf16-matmul and fp32-reduction attention, each with compiled prefill. So the small excess comes from
the bucketed decode kernels and/or the compiled prefill; I didn't isolate which. Per-variant numbers are
in `results/attn_variants_profile.json`.

**(b) Seeded generation.** With the same seed, tokens are identical for the first 2-270 tokens (median
about 50), then diverge. This is expected: bf16 kernels differ, so the sampled index flips at near-ties
(the stock path itself flips top-1 on about 3% of positions between its own two code paths). Lengths stay
comparable (stock vs optimized frames: 64/71, 73/69, 90/80, 87/97, 81/75, 67/68, 78/72, 81/78). The
optimized path is deterministic: identical tokens with and without the concurrent Mimi stream, 8/8.

**(c) Streaming Mimi vs full `MimiModel.decode` of the same codes.** As deployed (TF32 on, like the
baseline), SNR is 61.8-67.7 dB (max abs diff about 1.5e-3, 8 clips). The HF full decode with TF32 on vs
off differs by the same amount (65.0 dB). The streamed output is closer to an fp32 full decode (69.6 dB)
than HF's own TF32 decode is.

**(d) ASR (Whisper-large-v3).** CER against the input text; 10 texts x 3 seeds; stock `generate_audio`
plus HF Mimi vs optimized streaming, same seeds.

| | all | en | hinglish | hi | hi_en_mixed |
|---|---:|---:|---:|---:|---:|
| baseline mean CER (median) | 0.179 (0.044) | 0.015 (0.000) | 0.401 (0.067) | 0.051 (0.044) | 0.385 |
| optimized mean CER (median) | 0.183 (0.050) | 0.015 (0.000) | 0.407 (0.074) | 0.058 (0.053) | 0.385 |

Read this comparatively, not in absolute terms. Whisper writes romanized Hinglish and code-mixed text in
whatever script it picks, which inflates those columns equally for both paths. The differences are within
sample-to-sample variation (different sampled token sequences). Listening samples: `samples/optimized/`.

## Leading silence (about 3 frames before speech)

- The silence comes from the model, not the codec. Frame 0's semantic code (codebook 0) is 1049 in almost
  every run, and frame 1's is 127 or 958. Frames 0-1 decode to -45 to -70 dBFS; frame 2 is often a ramp.
  The first frame above -40 dBFS is frame 3 at the median (mean 2.7). This looks like learned onset silence
  from the training data.
- Codec warm-up is not the cause. Decoding a mid-utterance speech segment from a fresh Mimi state gives the
  same levels as decoding it in context (within about 1.5 dB on the first frame).
- The prompt doesn't change it. Dropping the `<description=...>` prefix (the `server.py` format) gives a
  median of frame 3 (mean 2.2). Speaker Aisha gives frame 3 (mean 2.8). The tokenized prompt is correct:
  it starts with `<BOS_TOKEN>`, `<text>` and ends with `<audio>` (id 277393).
- Effect on latency: first audible audio is ready at 110 ms (p50; 138 ms p95) instead of 599 ms. A player
  that skips leading frames below -40 dBFS would start speech at 110 ms. A player that plays them would
  start at 32 ms with about 240 ms of silence first. Suppressing the silence codes during sampling would
  remove it, but it changes the model's distribution, so it is not applied.

## Independent quality audit

A second pass re-checked the optimized path without relying on the gates above. The code was reviewed line by line against the pinned HF modeling code, and 160 matched-seed pairs (10 texts x 16 seeds, same H100, same sampling settings) were generated through both paths and scored.

- Intelligibility: Whisper-large-v3 and MMS-1b-all CER/WER show no distinguishable difference for English, Hinglish, Hindi or code-mixed text (paired bootstrap 95% CI plus sign-flip test).
- Naturalness: UTMOSv2 2.84 vs 2.86 and UTMOS22, no difference.
- Speaker: ECAPA and WavLM-SV cosine similarity to a stock reference, 0.873 vs 0.872.
- Duration, leading and trailing silence, clipping, loudness and spectrum: no difference. A stock-vs-stock null check (two seed batches) shows the same spread.
- Teacher forcing on 15,228 positions spanning every KV bucket: against an fp32 reference the optimized path is closer than stock bf16 (KL 5.7e-4 vs 7.9e-4, closer in 20/20 sequences). KV buckets have no measurable effect.
- Streaming Mimi vs full HF decode of the same tokens: median 66 dB SNR, no drift on clips over 10 s.
- Code review: nothing changes the output distribution. Known limits: prompts over 255 tokens are rejected, sampling settings are fixed at graph capture, one request per engine instance.
- Not covered: human listening, speakers other than Ira, IndicConformer (gated).

Scripts are in `eval/`, results in `results/audit/` (`report.txt`, `summary.json`); the generated clips are not checked in. `samples/listening_pack/` holds stock/optimized pairs for the same text and seed.

## What remains / next levers

- The bf16 decode step (3.05 ms GPU) is about 80% GEMV at about 2.5 TB/s. Weight-only int8 (or fp8) on the
  36 decoder layers would roughly halve the per-token time, to about 1.8-2 ms. That would bring TTFA to
  about 22-25 ms, and first speech at frame 3 to about 70 ms. It needs its own teacher-forced gate. It was
  not required for either target, so it was not done.
- The Mimi frame step is 2.1 ms alone, 4.5 ms concurrent with the LM. It is a graph of fp32 eager kernels.
  Compiling it, or running it in bf16 (quality-gated), would shave 1-3 ms off TTFA.
- The first frame still waits for 8 sequential LM steps (about 22 ms of the 32 ms). Going below about
  20 ms needs faster steps, not pipeline changes.
- The run-ahead wastes up to 16 LM steps (about 50 ms of GPU) after the stop. It is harmless for latency
  but counts against throughput when serving back-to-back requests.

## Reproduce (optimized)

```bash
# full benchmark + quality gates + ablation + leading-silence + Whisper CER (about 30 min on H100, about $2)
.venv/bin/modal run bench_fast.py --parts quality,variants,bench,mimicheck,silence,asr --variant c_kvb_f32_pf \
  --runs 10 --warmup 3 --n-tf-texts 4 --seeds 2 --tf-extra compile \
  --plan "eager:hf_first,graphs:hf_first,compile:hf_first,c_kvb:hf_first,c_kvb_f32:hf_first,c_kvb_f32_pf:hf_first,c_kvb_f32_pf:stream" \
  --out results/optimized.json
# smoke test
.venv/bin/modal run bench_fast.py --parts quality,variants --runs 2 --warmup 1 --n-tf-texts 1 --seeds 1 \
  --plan "c_kvb_f32_pf:stream" --out results/fast_smoke.json
```

The image adds `build-essential` (needed by triton/inductor). Inductor and triton caches persist in the
`rumik-hf-cache` volume, so FastLM builds take 5-40 s after the first run.

Files: `rumik_fast.py` (FastLM, StreamingMimi, StreamingTTS), `bench_fast.py` (Modal app),
`results/optimized.json` (environment, per-run timings, summaries, quality gates, ablation, silence, ASR
transcripts), `results/attn_variants_profile.json` (kernel profiles and teacher-forced comparison of the
attention variants), `samples/optimized/*.wav` (seed 1000 of each text, streamed output).

Cost of this phase: about $3.9 of Modal H100 time (smoke runs plus one full run). Prior work cost $3.28.

---

# Baseline (stock transformers path)

## Results (H100 80GB HBM3 SXM, bf16, warm)

Sampling: temperature 0.8, top-k 30, max 2048 new tokens, speaker `Ira`. 3 warm-up runs, then 10 timed
streaming runs per text (seeds 1000..1009) and 5 non-streaming runs per text (seeds 1000..1004, so they
produce exactly the same tokens as the matching streaming runs; verified 50/50).

| text | prompt tok | TTFA p50 (ms) | TTFA p95 (ms) | first non-silent audio p50 (ms) | audio (s, p50) | non-streaming total p50 (s) | RTFx p50 | decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en_short | 27 | 174.6 | 197.4 | 614 | 2.88 | 5.52 | 0.54 | 53.1 |
| en_medium | 36 | 161.7 | 168.0 | 618 | 5.52 | 10.15 | 0.54 | 55.0 |
| en_long | 52 | 155.2 | 164.8 | 584 | 11.60 | 20.97 | 0.56 | 55.2 |
| hinglish_short | 33 | 157.9 | 165.8 | 746 | 3.44 | 6.29 | 0.54 | 54.4 |
| hinglish_medium | 41 | 159.0 | 178.5 | 584 | 7.20 | 13.74 | 0.55 | 56.3 |
| hinglish_long | 79 | 156.5 | 166.6 | 600 | 13.36 | 24.58 | 0.55 | 55.0 |
| hi_short (Devanagari) | 27 | 156.0 | 162.0 | 590 | 3.52 | 6.33 | 0.57 | 56.8 |
| hi_medium (Devanagari) | 34 | 156.0 | 167.3 | 591 | 6.08 | 11.08 | 0.55 | 56.4 |
| hi_long (Devanagari) | 56 | 159.1 | 172.9 | 607 | 14.16 | 25.49 | 0.56 | 55.3 |
| hi_en_mixed_medium | 35 | 160.1 | 178.0 | 602 | 6.24 | 11.34 | 0.54 | 53.8 |
| **short (all langs)** | | 158.3 | 189.7 | 602 | 3.36 | 6.15 | 0.56 | 54.6 |
| **medium** | | 160.1 | 174.5 | 597 | 6.16 | 11.21 | 0.55 | 55.3 |
| **long** | | 156.5 | 169.0 | 594 | 13.36 | 24.58 | 0.56 | 55.2 |
| **all (n=100 / 50)** | | **158.3** | **182.2** | **599** | 6.16 | 11.21 | **0.55** | **55.1** |

RTFx = audio seconds / wall seconds for the full non-streaming request (model-card path, including Mimi
decode). Values below 1.0 mean slower than real time. Conventional RTF (wall / audio) is about 1.8.

TTFA breakdown (all runs, p50): prefill **19.8 ms** -> first complete codec frame (8 tokens) at
**144.9 ms** -> Mimi decode of that frame **13.0 ms** -> TTFA **158.3 ms**. TTFA does not depend on
text length or language in any meaningful way; prompts are only 27-79 tokens.

Other numbers:

- Per-step cost (instrumented runs, mean): LM forward 17.3-18.1 ms, of which 17.0-17.7 ms is spent
  on the CPU just issuing the forward (time until `model(...)` returns, before `cuda.synchronize()`);
  stop-predictor + host sync 0.19 ms; masked sampling + host sync 0.42 ms. Per-token wall p50 about 18 ms.
- Mimi decode is about 11 ms regardless of length (1 frame / 80 ms audio: 10.7 ms; 125 frames / 10 s
  audio: 11.6 ms); full-clip decode inside a request: 13.5 ms p50.
- Peak GPU memory: 7.9 GB allocated, 8.7 GB reserved. Model 3.38 B parameters.
- The stop predictor ended every run; no run hit `max_new_tokens`.
- The first audible frame (above -40 dBFS) is typically frame 3, i.e. the model emits about 240 ms of
  near-silence first, which is why "first non-silent audio" is about 440 ms later than TTFA.

## What this means

- The LM is the whole story. At about 55 tokens/s and 100 tokens per second of audio (8 codebooks x
  12.5 Hz), generation runs at 0.55x real time. A naive streaming player would underrun after the first
  chunk; for gap-free playback with the stock loop, the client would have to buffer roughly
  `0.8 x audio_duration` before starting (about 5 s for a 6 s utterance).
- Each forward is CPU / launch bound: the GPU finishes almost as soon as the CPU stops issuing kernels.
  A 3.4 B model at batch 1 on an H100 should need only a few ms of GPU time per token.
- Prefill (about 20 ms) and Mimi decode (about 11-13 ms) are small next to the first 8 decode steps
  (about 125 ms).

Low-hanging fruit observed (not applied here):

1. CUDA graphs / static KV cache (`torch.compile(mode="reduce-overhead")` or a static-cache decode
   step) to remove the about 17 ms of per-token launch overhead; this alone should push past real time.
2. `output_hidden_states=True` on every step returns all 37 layer outputs just to feed the stop
   predictor from the last one; computing the stop head from the final hidden state directly is cheaper.
3. Full 277k-vocab logits are computed and then masked to the 16,385 audio ids
   (`full_like` + `index_copy_` every step); slicing the LM head to audio rows saves work.
4. Two blocking GPU->CPU syncs per token (`bool(stop > 0.5)`, `bool(token == end)`); these can be
   checked every N steps or kept on device.
5. `attention_mask` grows via `torch.cat` every step; the DynamicCache also reallocates.
6. Mimi decode is about 11 ms even for one frame (also launch bound); a stateful streaming decoder
   (Kyutai `moshi` `mimi.streaming()`) or a CUDA-graphed decoder would cut both per-chunk cost and
   remove re-decode overlap once chunks are streamed continuously.
7. The model emits about 3 near-silent frames before speech; worth checking whether that is prompt- or
   sampling-dependent, since it adds about 440 ms to perceived latency at the current decode speed.

## Measurement method

All timing is server-side inside one Modal container (`modal run`, single function call). The model
is loaded and warmed up (3 streaming + 3 non-streaming runs on `en_medium`) before any timed run, so
cold start, image pull, and weight download are excluded. Every timestamp is taken with
`time.perf_counter()` after `torch.cuda.synchronize()` (the stock decode loop already forces a host
sync per token, so per-token timestamps are exact).

- **Request start**: before tokenization of the prompt.
- **Streaming loop (TTFA)**: a copy of `RumikOSSForCausalLM.generate_audio` with identical operations
  (prefill with `output_hidden_states=True`, stop predictor, audio-id masking, top-k sampling,
  DynamicCache). Tokens are assembled into Mimi frames incrementally using the same rules as
  `audio_tokens_to_codes`. As soon as the first complete frame (8 tokens) exists, it is decoded with
  `MimiModel.decode`, and TTFA is taken after a sync. Mimi's convolutions are causal, so decoding the
  first frame alone is valid for the first chunk; continuous chunk-by-chunk decoding is not part of
  this baseline. Generation then continues, so the first-chunk decode is included in streaming total.
- **First non-silent audio** (secondary): after the timed run, the run's full token sequence is decoded
  (untimed) and the first 80 ms frame above -40 dBFS RMS is located; the metric is that frame's
  token-completion time in the streaming run plus the measured first-chunk decode time.
- **Non-streaming**: exactly the model-card path: `model.generate_audio(...)` then
  `mimi.decode(model.audio_tokens_to_codes(tokens))`, synced; same seed as the matching streaming run.
- **decode tok/s**: steady-state rate from per-token timestamps, excluding prefill and the first frame.
- **Breakdown**: 3 extra runs with a sync after each forward to split CPU enqueue vs GPU completion;
  these runs are not used for the headline numbers.
- p50/p95 use linear interpolation over the runs in each group.

## Environment

- GPU: NVIDIA H100 80GB HBM3 (SXM, 700 W), driver 610.57.04, CUDA 12.8
- torch 2.8.0+cu128, transformers 4.57.6, Python 3.12 (Modal `debian_slim`)
- LM dtype bf16, attention `sdpa`; Mimi fp32
- Full pins: `requirements.txt` (compiled from `requirements.in` with `uv pip compile`)

## Reproduce

```bash
uv venv .venv && uv pip install --python .venv modal
.venv/bin/modal token new            # once, if not authenticated
.venv/bin/modal run bench_modal.py   # about 36 min on H100
# smoke test:
.venv/bin/modal run bench_modal.py --runs 1 --ns-runs 1 --warmup 1 --out results/smoke.json
```

Options: `--runs` (streaming runs per text), `--ns-runs` (non-streaming runs per text),
`--warmup`, `--chunk-frames` (frames in the first decoded chunk, default 1 = 80 ms).
The model is public; no Hugging Face token is needed. Weights are cached in the Modal volume
`rumik-hf-cache`.

## Files

- `bench_modal.py`: Modal app (benchmark + summary)
- `texts.json`: fixed text set (10 sentences; English, romanized Hinglish, Devanagari Hindi, mixed)
- `requirements.in`, `requirements.txt`: dependency spec and lock
- `results/baseline.json`: environment, per-run raw timings, summary per text / language / length
- `samples/*.wav`: first non-streaming run (seed 1000) of each text, 24 kHz mono PCM16

The model is licensed CC-BY-NC 4.0 (research / non-commercial use).
