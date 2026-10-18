# rumik-oss-1 latency baseline

Baseline time-to-first-audio (TTFA) and synthesis throughput for
[`rumik-ai/rumik-oss-1`](https://huggingface.co/rumik-ai/rumik-oss-1)
(revision `0ed3c98684e14350c910129c0efa179841c41ad2`) on one Modal H100, with **no optimizations**:
stock `transformers` code path, bf16 weights, default (SDPA) attention, eager mode, Mimi codec in fp32
as in the model card example.

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
