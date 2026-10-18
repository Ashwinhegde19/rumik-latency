"""Baseline latency benchmark for rumik-ai/rumik-oss-1 on a Modal GPU.

Runs everything inside a single GPU container: model load and warm-up happen
before any timed run, so cold start is excluded from every reported number.

    modal run bench_modal.py                 # full benchmark (H100)
    modal run bench_modal.py --runs 1 --ns-runs 1 --warmup 1 --out results/smoke.json   # smoke test

Outputs (written locally by the entrypoint):
    results/baseline.json      raw per-run timings + summary + environment
    samples/*.wav              one full (non-streaming) synthesis per text
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO_ID = "rumik-ai/rumik-oss-1"
REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"  # pinned HF commit
GPU = "H100"
HF_CACHE = "/hf-cache"

LOCAL_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(requirements=[str(LOCAL_DIR / "requirements.txt")])
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(LOCAL_DIR / "texts.json"), "/root/texts.json")
)
volume = modal.Volume.from_name("rumik-hf-cache", create_if_missing=True)
app = modal.App("rumik-latency-baseline", image=image)

SILENCE_DBFS = -40.0
GEN_KW = dict(max_new_tokens=2048, min_new_tokens=8, temperature=0.8, top_k=30, do_sample=True)


@app.function(gpu=GPU, volumes={HF_CACHE: volume}, timeout=60 * 60)
def run_benchmark(runs: int = 10, ns_runs: int = 5, warmup: int = 3, chunk_frames: int = 1) -> dict:
    import io
    import platform
    import subprocess
    from time import perf_counter

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    t_dl = perf_counter()
    root = Path(snapshot_download(REPO_ID, revision=REVISION, token=False))
    volume.commit()
    t_dl = perf_counter() - t_dl

    t_load = perf_counter()
    tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        root, trust_remote_code=True, dtype=torch.bfloat16
    ).eval().cuda()
    # Mimi loaded exactly as in the model card one-shot example (checkpoint dtype, fp32).
    mimi = MimiModel.from_pretrained(root / "codec").eval().cuda()
    torch.cuda.synchronize()
    t_load = perf_counter() - t_load

    cfg = model.config
    Q = int(cfg.num_quantizers)
    FIRST, LAST = int(cfg.first_unit_id), int(cfg.last_unit_id)
    END = int(cfg.audio_end_token_id)
    SR = 24000
    SAMPLES_PER_FRAME = int(SR / cfg.frame_rate_hz)  # 1920
    texts = json.loads(Path("/root/texts.json").read_text())

    def build_prompt(t: dict) -> str:
        return f'<text>{t["speaker"]}: <description="{t["description"]}"> {t["text"]}<audio>'

    def sync():
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # (a) minimal streaming loop. Mirrors RumikOSSForCausalLM.generate_audio
    # op-for-op, and decodes the first `chunk_frames` Mimi frames as soon as
    # they are complete. TTFA = request start -> first PCM chunk on GPU, synced.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def streaming_run(prompt: str, seed: int, breakdown: bool = False) -> dict:
        sync()
        t0 = perf_counter()
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        input_ids, mask = inputs.input_ids, inputs.attention_mask
        torch.manual_seed(seed)
        out = model(input_ids=input_ids, attention_mask=mask, use_cache=True,
                    output_hidden_states=True, return_dict=True)
        cache = out.past_key_values
        scores = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1:, :]
        sync()
        t_prefill = perf_counter() - t0

        allowed = model.audio_token_ids(device=input_ids.device)
        emitted = input_ids
        tokens: list[int] = []
        token_times: list[float] = []
        frame_done_times: list[float] = []
        cur = 0  # position within the round robin (frame assembly, as audio_tokens_to_codes)
        n_frames = 0
        t_first_audio = None
        first_chunk_decode = None
        stopped_by = "max_new_tokens"
        bd = {"fwd_enqueue": [], "fwd_total": [], "stop_pred": [], "sample": []} if breakdown else None

        for step in range(GEN_KW["max_new_tokens"]):
            ts = perf_counter()
            if step >= GEN_KW["min_new_tokens"]:
                stop = torch.sigmoid(model.stop_predictor(hidden).squeeze(-1))
                if bool((stop > 0.5).all()):
                    stopped_by = "stop_predictor"
                    break
            if breakdown:
                tsp = perf_counter(); bd["stop_pred"].append(tsp - ts)
            restricted = torch.full_like(scores, torch.finfo(scores.dtype).min)
            restricted.index_copy_(1, allowed, scores.index_select(1, allowed))
            if step < GEN_KW["min_new_tokens"]:
                restricted[:, END] = torch.finfo(scores.dtype).min
            token = model._constrained_sample(restricted, True, GEN_KW["temperature"], GEN_KW["top_k"])
            emitted = torch.cat((emitted, token), dim=1)
            is_end = bool((token == END).all())  # host sync (same as reference)
            tid = int(token)
            now = perf_counter()
            if breakdown:
                bd["sample"].append(now - tsp)
            tokens.append(tid)
            token_times.append(now - t0)
            if is_end:
                stopped_by = "audio_end_token"
                break
            # incremental frame assembly (same rules as audio_tokens_to_codes)
            if FIRST <= tid <= LAST:
                q = (tid - FIRST) % Q
                if q == cur:
                    cur += 1
                    if cur == Q:
                        cur = 0
                        n_frames += 1
                        frame_done_times.append(now - t0)
                else:
                    cur = 1 if q == 0 else 0
            else:
                cur = 0
            if t_first_audio is None and n_frames >= chunk_frames:
                td = perf_counter()
                codes = model.audio_tokens_to_codes(tokens).to("cuda")
                pcm = mimi.decode(codes).audio_values[0, 0]
                sync()
                t_first_audio = perf_counter() - t0
                first_chunk_decode = perf_counter() - td
                first_chunk_samples = int(pcm.shape[-1])
            mask = torch.cat((mask, torch.ones_like(token)), dim=1)
            tf = perf_counter()
            out = model(input_ids=token, attention_mask=mask, past_key_values=cache,
                        use_cache=True, output_hidden_states=True, return_dict=True)
            if breakdown:
                bd["fwd_enqueue"].append(perf_counter() - tf)
                sync()
                bd["fwd_total"].append(perf_counter() - tf)
            cache = out.past_key_values
            scores = out.logits[:, -1, :]
            hidden = out.hidden_states[-1][:, -1:, :]
        sync()
        t_gen_done = perf_counter() - t0

        n_audio_tokens = len([t for t in tokens if t != END])
        # untimed: full decode of this run's tokens to find the first non-silent 80 ms frame
        full = mimi.decode(model.audio_tokens_to_codes(tokens).to("cuda")).audio_values[0, 0].float()
        nfr = full.shape[-1] // SAMPLES_PER_FRAME
        rms = full[: nfr * SAMPLES_PER_FRAME].view(nfr, SAMPLES_PER_FRAME).pow(2).mean(-1).sqrt()
        frame_dbfs = (20 * torch.log10(rms.clamp_min(1e-9))).cpu().tolist()
        k = next((i for i, d in enumerate(frame_dbfs) if d > SILENCE_DBFS), None)
        ttfa_ns = (frame_done_times[k] + first_chunk_decode
                   if k is not None and k < len(frame_done_times) and first_chunk_decode else None)
        res = {
            "seed": seed,
            "prompt_tokens": int(input_ids.shape[1]),
            "audio_tokens": n_audio_tokens,
            "frames": n_frames,
            "audio_seconds": n_frames / cfg.frame_rate_hz,
            "stopped_by": stopped_by,
            "prefill_s": t_prefill,
            "first_token_s": token_times[0] if token_times else None,
            "first_frame_tokens_s": frame_done_times[0] if frame_done_times else None,
            "ttfa_s": t_first_audio,
            "first_chunk_decode_s": first_chunk_decode,
            "first_chunk_frames": chunk_frames,
            "first_nonsilent_frame": k,
            "ttfa_nonsilent_s": ttfa_ns,
            "gen_total_s": t_gen_done,
            "frame_done_times_s": frame_done_times,
        }
        # steady-state decode rate (excludes prefill and the inline first-chunk decode)
        if len(token_times) > 16:
            tt = np.array(token_times)
            gaps = np.diff(tt[8:])
            res["decode_tok_per_s"] = float(len(gaps) / gaps.sum())
            res["per_token_ms_p50"] = float(np.median(gaps) * 1e3)
        if breakdown:
            res["breakdown_ms_p50"] = {k: float(np.median(v) * 1e3) for k, v in bd.items() if v}
            res["breakdown_ms_mean"] = {k: float(np.mean(v) * 1e3) for k, v in bd.items() if v}
        return res

    # ------------------------------------------------------------------
    # (b) non-streaming: exactly the model-card path (generate_audio + full Mimi decode).
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def nonstreaming_run(prompt: str, seed: int, keep_audio: bool = False) -> dict:
        sync()
        t0 = perf_counter()
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        torch.manual_seed(seed)
        ids = model.generate_audio(**inputs, **GEN_KW)
        sync()
        t_gen = perf_counter() - t0
        audio_tokens = ids[0].tolist()[inputs.input_ids.shape[1]:]
        codes = model.audio_tokens_to_codes(audio_tokens).to("cuda")
        td = perf_counter()
        wav = mimi.decode(codes).audio_values[0, 0]
        sync()
        t_end = perf_counter()
        n_tok = len([t for t in audio_tokens if t != END])
        audio_s = wav.shape[-1] / SR
        res = {
            "seed": seed,
            "audio_tokens": n_tok,
            "frames": int(codes.shape[-1]),
            "audio_seconds": audio_s,
            "gen_s": t_gen,
            "mimi_decode_s": t_end - td,
            "total_s": t_end - t0,
            "tok_per_s_incl_prefill": n_tok / t_gen,
            "rtfx": audio_s / (t_end - t0),
        }
        if keep_audio:
            buf = io.BytesIO()
            sf.write(buf, wav.float().clamp(-1, 1).cpu().numpy(), SR, format="WAV", subtype="PCM_16")
            res["_wav"] = buf.getvalue()
        return res

    # ------------------------------------------------------------------
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,power.limit,clocks.max.sm",
                                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "gpu_mem_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": str(transformers.__version__),
        "python": platform.python_version(),
        "model_dtype": str(next(model.parameters()).dtype),
        "mimi_dtype": str(next(mimi.parameters()).dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "model_params_b": sum(p.numel() for p in model.parameters()) / 1e9,
        "repo": REPO_ID,
        "revision": REVISION,
        "generation": GEN_KW,
        "download_s": t_dl,
        "load_s": t_load,
    }
    print(json.dumps(env, indent=2))

    # warm-up (not recorded)
    wt = next(t for t in texts if t["id"] == "en_medium")
    for i in range(warmup):
        streaming_run(build_prompt(wt), seed=10_000 + i)
        nonstreaming_run(build_prompt(wt), seed=10_000 + i)
    sync()
    torch.cuda.reset_peak_memory_stats()

    # Mimi decode micro-benchmark (decode cost vs chunk length)
    mimi_bench = {}
    with torch.inference_mode():
        for nf in [1, 2, 4, 8, 25, 125]:
            codes = torch.randint(0, 2048, (1, Q, nf), device="cuda")
            for _ in range(3):
                mimi.decode(codes)
            sync()
            ts = []
            for _ in range(20):
                t = perf_counter(); mimi.decode(codes); sync(); ts.append(perf_counter() - t)
            mimi_bench[str(nf)] = {"p50_ms": float(np.median(ts) * 1e3), "audio_ms": nf * 80}

    results = []
    wavs = {}
    for t in texts:
        prompt = build_prompt(t)
        for r in range(runs):
            seed = 1000 + r
            s = streaming_run(prompt, seed)
            # non-streaming (model-card path) runs on the same seed -> identical tokens
            n = nonstreaming_run(prompt, seed, keep_audio=(r == 0)) if r < ns_runs else None
            if n is not None and "_wav" in n:
                wavs[t["id"]] = n.pop("_wav")
            results.append({"text_id": t["id"], "group": t["group"], "lang": t["lang"], "run": r,
                            "streaming": s, "nonstreaming": n,
                            "same_tokens": (s["audio_tokens"] == n["audio_tokens"]) if n else None})
            print(f'{t["id"]:18s} run {r:2d}  ttfa {s["ttfa_s"]*1e3:7.1f} ms  prefill {s["prefill_s"]*1e3:6.1f} ms  '
                  f'audio {s["audio_seconds"]:5.2f}s  gen {s["gen_total_s"]:6.2f}s  '
                  f'tok/s {s.get("decode_tok_per_s", 0):.1f}' + (f'  ns_total {n["total_s"]:6.2f}s rtfx {n["rtfx"]:.2f}' if n else ''),
                  flush=True)

    mem = {"max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
           "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}

    # per-step breakdown on one medium text (extra syncs; not used for headline numbers)
    breakdown = [streaming_run(build_prompt(wt), seed=2000 + i, breakdown=True) for i in range(3)]

    return {"env": env, "memory": mem, "mimi_decode_bench": mimi_bench, "texts": texts,
            "runs": results, "breakdown_runs": breakdown, "wavs": wavs,
            "config": {"runs": runs, "ns_runs": ns_runs, "warmup": warmup, "chunk_frames": chunk_frames}}


def summarize(data: dict) -> dict:
    import statistics

    def pct(xs, p):
        xs = sorted(xs)
        if not xs:
            return None
        k = (len(xs) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        return xs[f] + (xs[c] - xs[f]) * (k - f)

    def stats(xs):
        if not xs:
            return {"p50": None, "p95": None, "mean": None, "n": 0}
        return {"p50": pct(xs, 50), "p95": pct(xs, 95), "mean": statistics.fmean(xs), "n": len(xs)}

    groups: dict[str, list] = {}
    for r in data["runs"]:
        groups.setdefault(r["text_id"], []).append(r)
        groups.setdefault(f'lang:{r["lang"]}', []).append(r)
        groups.setdefault(f'len:{r["group"]}', []).append(r)
        groups.setdefault("ALL", []).append(r)
    out = {}
    for g, rs in groups.items():
        s = [r["streaming"] for r in rs]
        n = [r["nonstreaming"] for r in rs if r["nonstreaming"]]
        out[g] = {
            "ttfa_ms": stats([x["ttfa_s"] * 1e3 for x in s if x["ttfa_s"] is not None]),
            "ttfa_nonsilent_ms": stats([x["ttfa_nonsilent_s"] * 1e3 for x in s if x.get("ttfa_nonsilent_s")]),
            "first_nonsilent_frame": stats([x["first_nonsilent_frame"] for x in s if x.get("first_nonsilent_frame") is not None]),
            "prefill_ms": stats([x["prefill_s"] * 1e3 for x in s]),
            "first_frame_tokens_ms": stats([x["first_frame_tokens_s"] * 1e3 for x in s if x["first_frame_tokens_s"]]),
            "first_chunk_decode_ms": stats([x["first_chunk_decode_s"] * 1e3 for x in s if x["first_chunk_decode_s"]]),
            "decode_tok_per_s": stats([x["decode_tok_per_s"] for x in s if "decode_tok_per_s" in x]),
            "nonstreaming_total_s": stats([x["total_s"] for x in n]),
            "nonstreaming_mimi_decode_ms": stats([x["mimi_decode_s"] * 1e3 for x in n]),
            "audio_seconds": stats([x["audio_seconds"] for x in n]),
            "rtfx": stats([x["rtfx"] for x in n]),
            "prompt_tokens": s[0]["prompt_tokens"],
            "hit_max_tokens": sum(x["stopped_by"] == "max_new_tokens" for x in s),
            "same_tokens_both_modes": f'{sum(bool(r["same_tokens"]) for r in rs if r["nonstreaming"])}/{len(n)}',
            "streaming_gen_total_s": stats([x["gen_total_s"] for x in s]),
            "streaming_rtfx": stats([x["audio_seconds"] / x["gen_total_s"] for x in s]),
        }
    return out


@app.local_entrypoint()
def main(runs: int = 10, ns_runs: int = 5, warmup: int = 3, chunk_frames: int = 1, out: str = "results/baseline.json"):
    t0 = time.time()
    data = run_benchmark.remote(runs=runs, ns_runs=ns_runs, warmup=warmup, chunk_frames=chunk_frames)
    wavs = data.pop("wavs")
    data["summary"] = summarize(data)
    data["wall_clock_remote_call_s"] = time.time() - t0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    sd = Path("samples"); sd.mkdir(exist_ok=True)
    for k, b in wavs.items():
        (sd / f"{k}.wav").write_bytes(b)
    print(f"\nwrote {out} and {len(wavs)} wavs to samples/  (remote call {data['wall_clock_remote_call_s']:.0f}s)")
    print(f'{"group":24s} {"TTFA p50":>9s} {"p95":>8s} {"TTFA-ns":>8s} {"prefill":>8s} {"tok/s":>7s} {"RTFx":>6s} {"audio s":>8s} {"total s":>8s}')
    for g, s in data["summary"].items():
        print(f'{g:24s} {s["ttfa_ms"]["p50"]:9.1f} {s["ttfa_ms"]["p95"]:8.1f} {s["ttfa_nonsilent_ms"]["p50"] or 0:8.1f} {s["prefill_ms"]["p50"]:8.1f} '
              f'{s["decode_tok_per_s"]["p50"]:7.1f} {s["rtfx"]["p50"]:6.2f} {s["audio_seconds"]["p50"]:8.2f} '
              f'{s["nonstreaming_total_s"]["p50"]:8.2f}')
