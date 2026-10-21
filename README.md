# rumik-latency

Streaming inference for [rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1), Rumik's open Indic
text-to-speech model: **158 ms → 32 ms to first audio** on one H100, with quality verified unchanged.

The model is not changed. No retraining, no quantization, bf16 throughout. Only the code that runs it is new.

## Results

H100 80GB SXM, warm, 10 texts (English, romanized Hinglish, Devanagari Hindi, code-mixed) × 10 runs.

| | stock `transformers` path | this repo |
|---|---:|---:|
| time to first audio (p50 / p95) | 158 / 182 ms | **32.1 / 32.5 ms** |
| first audible speech (p50) | 599 ms | **110 ms** |
| decode speed | 55 tokens/s | **301 tokens/s** |
| real-time factor (audio s / wall s) | 0.55× (playback stalls) | **2.84×** (0 stalls in 100 runs) |
| time per token | ~18 ms | 3.3 ms |

Time to first audio is almost identical across languages and lengths (31.8–32.4 ms per text).

## What changed

- **CUDA graphs.** The stock loop spent ~17 of its 18 ms per token on CPU-side kernel launches and two
  GPU→CPU syncs. One decode step (forward + sampling + stop check, all on the GPU) is captured once and replayed.
- **Static KV cache** (2304 slots), the precondition for graph replay.
- **KV-length buckets** (256 / 512 / 1024 / 2304). After the graphs, attention was the bottleneck: a profile
  showed 3.4 of 6.4 ms per step scanning the whole cache with 4 thread blocks. One graph per bucket, masked,
  so the result is exact. Biggest single win.
- **`torch.compile`** inside the graphs, fp32-reduction decode attention, compiled prefill.
- **Audio-only output head.** 16,385 rows instead of 277,404. Exact, since the stock loop masks the rest anyway.
- **Streaming Mimi decoder.** Frame-by-frame codec decode with explicit state, graph-captured on its own
  CUDA stream so it overlaps the language model. 13 ms → 4.5 ms per frame.

Sampling reproduces the stock `_constrained_sample` exactly, including random-number consumption, so a seed
gives the same tokens as the stock loop up to bf16 kernel differences.

The final step is ~3.05 ms of GPU time, ~2.5 ms of it reading weights at ~2.5 TB/s: close to the bf16
memory-bandwidth floor. Weight-only int8/fp8 is the next lever (~22–25 ms), not done here.

## Quality

Verified two ways, both against the unmodified stock path:

- **Teacher forcing** on 15,228 positions across every cache bucket: optimized-vs-stock next-token KL
  (1.04e-3) is within the stock-vs-stock noise floor (9.0e-4), and against an fp32 reference the optimized
  path is closer than stock bf16 (5.7e-4 vs 7.9e-4, in 20/20 sequences).
- **160 matched-seed pairs** (same text, same seed, both paths) scored with Whisper-large-v3 and MMS-1b CER,
  UTMOS naturalness, and ECAPA / WavLM speaker similarity: no distinguishable difference in any language group
  (paired bootstrap 95% CI). Streamed codec output matches a full decode at 66 dB SNR.

Full tables and method: [docs/REPORT.md](docs/REPORT.md). Stock/optimized pairs for listening:
[`samples/listening_pack/`](samples/listening_pack/). Not yet done: a human listening test for Hindi and Hinglish.

## Use the fast path

```python
import torch, soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel
from rumik_fast import FastLM, StreamingMimi, StreamingTTS

repo = "rumik-ai/rumik-oss-1"  # tested at revision 0ed3c98684e14350c910129c0efa179841c41ad2
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
mimi = MimiModel.from_pretrained(repo, subfolder="codec").eval().cuda()

lm = FastLM(model, use_compile=True, kv_buckets=(256, 512, 1024), attn="f32",
            compile_prefill=True, prefill_buckets=(32, 64, 128, 256),
            max_new_tokens=2048, min_new_tokens=8, temperature=0.8, top_k=30)  # graph capture + compile, once
tts = StreamingTTS(lm, tok, smimi=StreamingMimi(mimi))

prompt = '<text>Ira: <description="warm, Indian English accent, steady pace"> Namaste, aaj ka weather bahut accha hai.<audio>'
r = tts.run(prompt, seed=1000, keep_audio=True)
print(f"first audio at {r['ttfa_s'] * 1e3:.1f} ms, {r['audio_seconds']:.1f} s of speech")
sf.write("out.wav", r["_pcm"].float().clamp(-1, 1).numpy(), 24000)
```

`run()` decodes each 80 ms frame as soon as its 8 tokens exist; a server would hand `tts.pin_pcm[k]` to the
client per frame. Needs an NVIDIA GPU, `torch==2.8.0`, `transformers==4.57.6` (see `requirements.txt`).

## Reproduce the benchmarks

Both benchmarks run on [Modal](https://modal.com) (`pip install modal`, `modal setup`); the image is built from
`requirements.txt` and the weights are cached in a Modal volume.

```bash
# optimized path, smoke test (a few minutes on an H100)
modal run bench_fast.py --parts quality,variants --runs 2 --warmup 1 --n-tf-texts 1 --seeds 1 \
  --plan "c_kvb_f32_pf:stream" --out results/fast_smoke.json

# optimized path, full benchmark + quality gates + ablation (about 30 min, about $2)
modal run bench_fast.py --parts quality,variants,bench,mimicheck,silence,asr --variant c_kvb_f32_pf \
  --runs 10 --warmup 3 --n-tf-texts 4 --seeds 2 --tf-extra compile \
  --plan "eager:hf_first,graphs:hf_first,compile:hf_first,c_kvb:hf_first,c_kvb_f32:hf_first,c_kvb_f32_pf:hf_first,c_kvb_f32_pf:stream" \
  --out results/optimized.json

# stock baseline (about 36 min)
modal run bench_modal.py
```

## Limitations

- Prompts over ~255 tokens are rejected (prefill buckets); `max_new_tokens` above 2048 needs a larger cache.
- Temperature and top-k are fixed when the graphs are captured.
- One request at a time per engine instance; no batching.
- Measured on H100 only, with the Ira voice only.
- The model emits ~3 near-silent frames before speech (first audible sound at ~110 ms). That is the model's
  output, not this code; see the report.

## Layout

| | |
|---|---|
| `rumik_fast.py` | the optimized engine: `FastLM`, `StreamingMimi`, `StreamingTTS` |
| `bench_fast.py` | Modal benchmark, ablation and quality gates for the optimized path |
| `bench_modal.py` | Modal benchmark for the stock path (unchanged baseline) |
| `eval/` | independent quality audit: matched-seed generation, judges, statistics |
| `results/` | raw timings and audit data (`baseline.json`, `optimized.json`, `audit/report.txt`) |
| `samples/` | generated audio: stock, optimized, and matched listening pairs |
| `docs/REPORT.md` | full write-up: ablation, method, quality gates, leading silence, audit, next levers |

## Model licence

rumik-oss-1 weights are CC-BY-NC 4.0 (research / non-commercial). This repo contains no weights.
