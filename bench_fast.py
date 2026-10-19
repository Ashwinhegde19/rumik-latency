"""Optimized streaming path for rumik-ai/rumik-oss-1 on a Modal GPU: quality gates + benchmark.

Same container rules as bench_modal.py (model load, graph capture and warm-up
happen before any timed run; cold start excluded). The baseline path is
untouched; this file only adds the optimized path from rumik_fast.py.

    modal run bench_fast.py --parts quality,variants,silence --runs 3      # smoke
    modal run bench_fast.py --parts quality,bench --runs 10 --out results/optimized.json

Parts:
    quality   teacher-forced logit agreement vs the stock HF model (en / hinglish / hi),
              seeded generation comparison vs stock generate_audio, streaming Mimi vs
              full MimiModel.decode, optimized samples
    variants  per-step ablation (FastLM eager -> + CUDA graphs -> + torch.compile;
              HF first-frame decode -> streaming Mimi) on two texts
    bench     full benchmark of the chosen variant on all texts
    silence   leading-silence investigation (prompt variants, codec warm-up)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO_ID = "rumik-ai/rumik-oss-1"
REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"  # pinned HF commit (same as baseline)
GPU = "H100"
HF_CACHE = "/hf-cache"
LOCAL_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # triton / inductor need a host C compiler
    .uv_pip_install(requirements=[str(LOCAL_DIR / "requirements.txt")])
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "TORCHINDUCTOR_CACHE_DIR": f"{HF_CACHE}/inductor-cache", "TRITON_CACHE_DIR": f"{HF_CACHE}/triton-cache"})
    .add_local_file(str(LOCAL_DIR / "texts.json"), "/root/texts.json")
    .add_local_python_source("rumik_fast")
)
volume = modal.Volume.from_name("rumik-hf-cache", create_if_missing=True)
app = modal.App("rumik-latency-fast", image=image)

GEN_KW = dict(max_new_tokens=2048, min_new_tokens=8, temperature=0.8, top_k=30)
KVB = (256, 512, 1024)
PFB = (32, 64, 128, 256)
# FastLM configurations (step-by-step ablation)
VARIANTS = {
    "eager": dict(use_graphs=False),                                   # static cache + on-GPU sampling, no graphs
    "graphs": dict(use_graphs=True),                                   # + CUDA graphs
    "compile": dict(use_compile=True),                                 # + torch.compile (inductor) inside the graphs
    "c_kvb": dict(use_compile=True, kv_buckets=KVB),                   # + KV-length buckets
    "c_kvb_man": dict(use_compile=True, kv_buckets=KVB, attn="manual"),
    "c_kvb_man_pf": dict(use_compile=True, kv_buckets=KVB, attn="manual", compile_prefill=True, prefill_buckets=PFB),
    "c_kvb_pf": dict(use_compile=True, kv_buckets=KVB, compile_prefill=True, prefill_buckets=PFB),
    "c_kvb_f32": dict(use_compile=True, kv_buckets=KVB, attn="f32"),
    "c_kvb_f32_pf": dict(use_compile=True, kv_buckets=KVB, attn="f32", compile_prefill=True, prefill_buckets=PFB),
}
SR = 24000


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "mean": sum(xs) / len(xs), "n": len(xs)}


def summarize_runs(rs: list[dict]) -> dict:
    return {
        "ttfa_ms": stats([r["ttfa_s"] * 1e3 for r in rs if r["ttfa_s"]]),
        "ttfa_nonsilent_ms": stats([r["ttfa_nonsilent_s"] * 1e3 for r in rs if r.get("ttfa_nonsilent_s")]),
        "first_nonsilent_frame": stats([r.get("first_nonsilent_frame") for r in rs]),
        "prefill_ms": stats([r["prefill_s"] * 1e3 for r in rs]),
        "first_frame_tokens_ms": stats([r["first_frame_tokens_s"] * 1e3 for r in rs if r["first_frame_tokens_s"]]),
        "first_chunk_decode_ms": stats([r["first_chunk_decode_s"] * 1e3 for r in rs if r["first_chunk_decode_s"]]),
        "decode_tok_per_s": stats([r.get("decode_tok_per_s") for r in rs]),
        "per_token_ms": stats([r.get("per_token_ms") for r in rs]),
        "mimi_frame_ms": stats([r.get("mimi_frame_ms_p50") for r in rs]),
        "mimi_frame_ms_max": stats([r.get("mimi_frame_ms_max") for r in rs]),
        "audio_seconds": stats([r["audio_seconds"] for r in rs]),
        "total_s": stats([r["total_s"] for r in rs]),
        "rtfx": stats([r.get("rtfx") for r in rs]),
        "max_lateness_ms": stats([r["max_lateness_s"] * 1e3 for r in rs if "max_lateness_s" in r]),
        "stall_free": f'{sum(bool(r.get("stall_free")) for r in rs)}/{len(rs)}',
        "hit_max_tokens": sum(r["stopped_by"] == "max_new_tokens" for r in rs),
    }


@app.function(gpu=GPU, volumes={HF_CACHE: volume}, timeout=60 * 60)
def run_fast(parts: str = "quality,variants", runs: int = 3, warmup: int = 3,
             variant: str = "c_kvb_f32_pf", n_tf_texts: int = 3, seeds: int = 2,
             plan: str = "eager:hf_first,graphs:hf_first,graphs:stream,compile:hf_first,compile:stream",
             tf_extra: str = "") -> dict:
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

    from rumik_fast import FastLM, StreamingMimi, StreamingTTS

    parts_set = set(parts.split(","))
    root = Path(snapshot_download(REPO_ID, revision=REVISION, token=False))
    tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    mimi = MimiModel.from_pretrained(root / "codec").eval().cuda()  # fp32, as baseline
    cfg = model.config
    Q, END = int(cfg.num_quantizers), int(cfg.audio_end_token_id)
    texts = json.loads(Path("/root/texts.json").read_text())
    tmap = {t["id"]: t for t in texts}

    def build_prompt(t: dict, speaker=None, description=True) -> str:
        sp = speaker or t["speaker"]
        if description:
            return f'<text>{sp}: <description="{t["description"]}"> {t["text"]}<audio>'
        return f'<text>{sp}: {t["text"]}<audio>'

    def wav_bytes(x) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, x.float().clamp(-1, 1).cpu().numpy(), SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,power.limit,clocks.max.sm,driver_version",
                                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "transformers": str(transformers.__version__),
        "python": platform.python_version(), "repo": REPO_ID, "revision": REVISION, "generation": GEN_KW,
    }
    print(json.dumps(env), flush=True)
    out: dict = {"env": env, "parts": parts, "variant": variant}
    wavs: dict[str, bytes] = {}

    p0 = build_prompt(tmap["en_medium"])
    ids0 = tok(p0).input_ids
    out["prompt_tokens_example"] = {"prompt": p0, "ids": ids0, "pieces": tok.convert_ids_to_tokens(ids0)}

    lms: dict = {}

    def get_lm(name: str) -> FastLM:
        if name not in lms:
            for k in list(lms):
                del lms[k]
            torch.cuda.empty_cache()
            t = perf_counter()
            kw = dict(max_new_tokens=GEN_KW["max_new_tokens"], min_new_tokens=GEN_KW["min_new_tokens"],
                      temperature=GEN_KW["temperature"], top_k=GEN_KW["top_k"])
            try:
                lms[name] = FastLM(model, **VARIANTS[name], **kw)
            except Exception as e:  # keep the run alive; record why
                import traceback
                traceback.print_exc()
                out.setdefault("build_errors", {})[name] = repr(e)[:2000]
                if name in ("eager", "graphs"):
                    raise
                lms[name] = FastLM(model, use_graphs=True, **kw)
            torch.cuda.synchronize()
            out.setdefault("build_s", {})[name] = perf_counter() - t
            print(f"built FastLM[{name}] in {perf_counter() - t:.1f}s", flush=True)
        return lms[name]

    def asr_check() -> dict:
        """Whisper-large-v3 CER: stock generate_audio + HF Mimi (model-card path) vs optimized streaming,
        same texts and seeds (3 per text). Comparative only: romanized Hinglish refs vs Whisper output
        are noisy in absolute terms."""
        import unicodedata

        from transformers import pipeline

        def resample_24k_16k(x: np.ndarray) -> np.ndarray:
            up = np.zeros(2 * len(x), dtype=np.float64)
            up[::2] = 2 * x
            n = np.arange(129) - 64
            fc = 7600 / 48000
            h = 2 * fc * np.sinc(2 * fc * n) * np.hanning(129)
            return np.convolve(up, h, mode="same")[::3].astype(np.float32)

        def norm(t: str) -> str:
            t = unicodedata.normalize("NFC", t.lower())
            t = "".join(" " if unicodedata.category(c).startswith("P") else c for c in t)
            return "".join(t.split())

        def cer(ref: str, hyp: str) -> float:
            r, h = norm(ref), norm(hyp)
            d = list(range(len(h) + 1))
            for i in range(1, len(r) + 1):
                prev, d[0] = d[0], i
                for j in range(1, len(h) + 1):
                    cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                    prev, d[j] = d[j], cur
            return d[len(h)] / max(1, len(r))

        lm = get_lm(variant)
        tts = StreamingTTS(lm, tok, smimi=smimi)
        clips = []
        for t_ in texts:
            prompt = build_prompt(t_)
            for seed in (1000, 1001, 1002):
                inputs = tok(prompt, return_tensors="pt").to("cuda")
                torch.manual_seed(seed)
                with torch.inference_mode():
                    ids = model.generate_audio(**inputs, **GEN_KW, do_sample=True)
                    toks = ids[0].tolist()[inputs.input_ids.shape[1]:]
                    base = mimi.decode(model.audio_tokens_to_codes(toks).cuda()).audio_values[0, 0].float().cpu().numpy()
                r = tts.run(prompt, seed, mimi_mode="stream", keep_audio=True)
                clips.append((t_, seed, "baseline", base))
                clips.append((t_, seed, "optimized", r["_pcm"].numpy()))
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                       torch_dtype=torch.float16, device="cuda")
        rows = []
        for t_, seed, kind, x in clips:
            lang = "hi" if t_["lang"] in ("hi", "hi_en_mixed") else "en"
            hyp = asr({"raw": resample_24k_16k(x), "sampling_rate": 16000},
                      generate_kwargs={"language": lang, "task": "transcribe"})["text"]
            rows.append({"text_id": t_["id"], "lang": t_["lang"], "seed": seed, "kind": kind,
                         "hyp": hyp, "cer": cer(t_["text"], hyp), "audio_s": len(x) / SR})
        summ = {}
        for kind in ("baseline", "optimized"):
            for g in ["ALL", "en", "hinglish", "hi", "hi_en_mixed"]:
                xs = [r["cer"] for r in rows if r["kind"] == kind and (g == "ALL" or r["lang"] == g)]
                summ[f"{kind}/{g}"] = {"mean_cer": sum(xs) / len(xs), "median_cer": pct(xs, 50), "n": len(xs)}
        print("asr", json.dumps(summ), flush=True)
        return {"model": "openai/whisper-large-v3", "summary": summ, "rows": rows}

    t = perf_counter()
    smimi = StreamingMimi(mimi)
    out["build_s_mimi"] = perf_counter() - t

    # ------------------------------------------------------------------ quality
    if "quality" in parts_set:
        try:
            q: dict = {}
            aud = model.audio_token_ids(device="cuda")

            @torch.inference_mode()
            def stock_generate(prompt, seed):
                inputs = tok(prompt, return_tensors="pt").to("cuda")
                torch.manual_seed(seed)
                ids = model.generate_audio(**inputs, **GEN_KW, do_sample=True)
                return [int(x) for x in ids[0].tolist()[inputs.input_ids.shape[1]:]]

            @torch.inference_mode()
            def stock_tf(prompt, seq):
                """Stock incremental decode (DynamicCache, as generate_audio): logits over audio ids + stop prob."""
                inputs = tok(prompt, return_tensors="pt").to("cuda")
                o = model(**inputs, use_cache=True, output_hidden_states=True, return_dict=True)
                cache, mask = o.past_key_values, inputs.attention_mask
                L, S = [o.logits[0, -1].index_select(0, aud).float()], [torch.sigmoid(model.stop_predictor(o.hidden_states[-1][:, -1:]).float()).view(())]
                for t_ in seq[:-1]:
                    tt = torch.tensor([[t_]], device="cuda")
                    mask = torch.cat((mask, torch.ones_like(tt)), 1)
                    o = model(input_ids=tt, attention_mask=mask, past_key_values=cache, use_cache=True,
                              output_hidden_states=True, return_dict=True)
                    cache = o.past_key_values
                    L.append(o.logits[0, -1].index_select(0, aud).float())
                    S.append(torch.sigmoid(model.stop_predictor(o.hidden_states[-1][:, -1:]).float()).view(()))
                return torch.stack(L), torch.stack(S)

            @torch.inference_mode()
            def stock_full(prompt, seq):
                """Stock single full-sequence forward (no cache): noise-floor reference."""
                ids = tok(prompt).input_ids
                x = torch.tensor([ids + seq[:-1]], device="cuda")
                o = model(input_ids=x, output_hidden_states=True, return_dict=True)
                P = len(ids)
                lg = o.logits[0, P - 1:].index_select(-1, aud).float()
                st = torch.sigmoid(model.stop_predictor(o.hidden_states[-1][:, P - 1:]).float()).view(-1)
                return lg, st

            @torch.inference_mode()
            def fast_tf(lm, prompt, seq):
                ids = tok(prompt).input_ids
                lm.prefill(ids)
                L, S = [lm.logits_buf.float().clone()], [lm.stop_prob().clone()]
                for t_ in seq[:-1]:
                    lm.forward_token(t_)
                    L.append(lm.logits_buf.float().clone())
                    S.append(lm.stop_prob().clone())
                return torch.stack(L), torch.stack(S)

            def compare(ref, test):
                (lr, sr), (lt, st) = ref, test
                n = min(len(lr), len(lt))
                lr, lt, sr, st = lr[:n], lt[:n], sr[:n], st[:n]
                res = {"positions": n,
                       "top1_agree": float((lr.argmax(-1) == lt.argmax(-1)).float().mean()),
                       "max_abs_logit_diff": float((lr - lt).abs().max()),
                       "stop_decision_agree": float(((sr > 0.5) == (st > 0.5)).float().mean()),
                       "stop_prob_max_abs_diff": float((sr - st).abs().max())}
                for T in (1.0, 0.8):
                    lp, lq = torch.log_softmax(lr / T, -1), torch.log_softmax(lt / T, -1)
                    kl = (lp.exp() * (lp - lq)).sum(-1)
                    res[f"kl_mean_T{T}"] = float(kl.mean())
                    res[f"kl_max_T{T}"] = float(kl.max())
                # top-30 set overlap (the sampling support)
                a, b = lr.topk(30, -1).indices, lt.topk(30, -1).indices
                res["top30_overlap"] = float(torch.tensor([len(set(x.tolist()) & set(y.tolist())) / 30 for x, y in zip(a, b)]).mean())
                return res

            tf_ids = ["en_medium", "hinglish_medium", "hi_medium", "hi_en_mixed_medium"][:n_tf_texts]
            q["teacher_forced"] = {}
            q["seeded"] = {}
            q["mimi_stream_vs_full"] = {}
            # stock references first (shared by every FastLM variant)
            refs = {}
            for tid in tf_ids:
                prompt = build_prompt(tmap[tid])
                seqs = {}
                for s_ in range(seeds):
                    seed = 1000 + s_
                    t = perf_counter()
                    seqs[seed] = stock_generate(prompt, seed)
                    print(f"stock gen {tid} seed {seed}: {len(seqs[seed])} tok in {perf_counter() - t:.1f}s", flush=True)
                seq = [x for x in seqs[1000] if x != END] + [END]
                ref = stock_tf(prompt, seq)
                full = stock_full(prompt, seq)
                refs[tid] = (prompt, seqs, seq, ref)
                q["teacher_forced"][tid] = {"stock_full_vs_stock_incremental (noise floor)": compare(ref, full)}
            tf_variants = list(dict.fromkeys([v for v in tf_extra.split(",") if v] + [variant]))  # main variant last
            for vname in tf_variants:
                lm = get_lm(vname)
                for tid in tf_ids:
                    prompt, _, seq, ref = refs[tid]
                    q["teacher_forced"][tid][f"fast[{vname}]_vs_stock_incremental"] = compare(ref, fast_tf(lm, prompt, seq))
            for tid in tf_ids:
                print(tid, json.dumps(q["teacher_forced"][tid]), flush=True)
            # seeded generation + streaming Mimi exactness with the main variant
            tts = StreamingTTS(lm, tok, smimi=smimi)
            for tid in tf_ids:
                prompt, seqs, _, _ = refs[tid]
                q["seeded"][tid] = []
                for seed, sseq in seqs.items():
                    r = tts.run(prompt, seed, mimi_mode="stream", keep_audio=True)
                    r_none = tts.run(prompt, seed, mimi_mode="none")  # LM alone, no concurrent Mimi
                    a = [x for x in sseq if x != END]
                    b = [x for x in r["tokens"] if x != END]
                    n_same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    q["seeded"][tid].append({"seed": seed, "stock_len": len(a), "fast_len": len(b),
                                             "identical": a == b, "common_prefix": n_same,
                                             "fast_same_with_and_without_concurrent_mimi": r_none["tokens"] == r["tokens"],
                                             "stock_frames": len(a) // Q, "fast_frames": r["frames"]})
                    # streaming Mimi vs full HF decode of the same codes
                    codes = model.audio_tokens_to_codes(r["tokens"]).cuda()
                    with torch.inference_mode():
                        full_pcm = mimi.decode(codes).audio_values[0, 0].float()
                    sp = r["_pcm"].cuda()
                    n = min(len(sp), len(full_pcm))
                    err = (sp[:n] - full_pcm[:n])
                    q["mimi_stream_vs_full"][f"{tid}/{seed}"] = {
                        "samples": n, "max_abs_diff": float(err.abs().max()),
                        "snr_db": float(10 * torch.log10(full_pcm[:n].pow(2).sum() / err.pow(2).sum().clamp_min(1e-20))),
                        "len_stream": len(sp), "len_full": len(full_pcm)}
                    if seed == 1000:
                        wavs[f"{tid}_seed{seed}_stream.wav"] = wav_bytes(r["_pcm"])
                print(tid, json.dumps(q["seeded"][tid]), json.dumps({k: v for k, v in q["mimi_stream_vs_full"].items() if k.startswith(tid)}), flush=True)
            out["quality"] = q
        except Exception as e:  # never lose the other parts
            import traceback
            traceback.print_exc()
            out["quality_error"] = repr(e)[:2000]

    # ------------------------------------------------------------------ variants (ablation)
    if "variants" in parts_set:
        vt = ["en_medium", "hi_medium"]
        res: dict = {}
        for item in plan.split(","):
            name, mm = item.split(":")
            lm = get_lm(name)
            # eager is CPU-bound: running ahead would only delay the first chunk check
            tts = StreamingTTS(lm, tok, smimi=smimi, hf_mimi=mimi, lookahead=0 if name == "eager" else 2)
            for i in range(warmup):
                tts.run(build_prompt(tmap["en_medium"]), 10_000 + i, mimi_mode=mm)
            rs = []
            for tid in vt:
                for r in range(runs):
                    x = tts.run(build_prompt(tmap[tid]), 1000 + r, mimi_mode=mm)
                    x.pop("tokens")
                    x.pop("frame_ready_s", None)
                    x.pop("frame_dbfs", None)
                    rs.append(x)
            res[f"{name}+{mm}"] = summarize_runs(rs)
            s = res[f"{name}+{mm}"]
            print(f'{name:8s} {mm:9s} TTFA p50 {s["ttfa_ms"]["p50"]:.1f} p95 {s["ttfa_ms"]["p95"]:.1f}  prefill {s["prefill_ms"]["p50"]:.1f}  '
                  f'tok/s {s["decode_tok_per_s"]["p50"]:.1f}  mimi/frame {s["mimi_frame_ms"]["p50"]}', flush=True)
        out["variants"] = res
        # standalone per-step timings (synced) for the chosen variant
        lm = get_lm(variant)
        with torch.inference_mode():
            lm.prefill(ids0)
            torch.cuda.synchronize()
            ts = []
            for _ in range(50):
                t = perf_counter(); lm.step(1); torch.cuda.synchronize(); ts.append(perf_counter() - t)
            smimi.reset()
            codes = torch.randint(0, 2048, (1, Q, 1), device="cuda")
            tm = []
            for _ in range(30):
                t = perf_counter(); smimi.decode_frame(codes); torch.cuda.synchronize(); tm.append(perf_counter() - t)
            tp = []
            for _ in range(10):
                t = perf_counter(); lm.prefill(ids0); torch.cuda.synchronize(); tp.append(perf_counter() - t)
        out["micro"] = {"step_ms_p50": float(np.median(ts) * 1e3), "mimi_frame_ms_p50": float(np.median(tm) * 1e3),
                        "prefill_ms_p50": float(np.median(tp) * 1e3), "prompt_tokens": len(ids0)}
        print("micro", out["micro"], flush=True)

    # ------------------------------------------------------------------ full benchmark
    if "bench" in parts_set:
        lm = get_lm(variant)
        tts = StreamingTTS(lm, tok, smimi=smimi)
        for i in range(warmup):
            tts.run(build_prompt(tmap["en_medium"]), 10_000 + i, mimi_mode="stream")
        torch.cuda.reset_peak_memory_stats()
        runs_out = []
        for t_ in texts:
            for r in range(runs):
                x = tts.run(build_prompt(t_), 1000 + r, mimi_mode="stream", keep_audio=(r == 0))
                if "_pcm" in x:
                    wavs[f'{t_["id"]}.wav'] = wav_bytes(x.pop("_pcm"))
                x.pop("tokens")
                runs_out.append({"text_id": t_["id"], "group": t_["group"], "lang": t_["lang"], "run": r, "streaming": x})
                print(f'{t_["id"]:18s} run {r:2d}  ttfa {x["ttfa_s"]*1e3:6.1f} ms  first-speech {(x["ttfa_nonsilent_s"] or 0)*1e3:6.1f} ms  '
                      f'audio {x["audio_seconds"]:5.2f}s  total {x["total_s"]:5.2f}s  tok/s {x.get("decode_tok_per_s", 0):.1f}  '
                      f'rtfx {x["rtfx"]:.2f}  mimi {x["mimi_frame_ms_p50"]:.2f} ms  stall_free {x["stall_free"]}', flush=True)
        groups: dict = {}
        for r in runs_out:
            for g in (r["text_id"], f'lang:{r["lang"]}', f'len:{r["group"]}', "ALL"):
                groups.setdefault(g, []).append(r["streaming"])
        out["bench"] = {"runs": runs_out, "summary": {g: summarize_runs(v) for g, v in groups.items()},
                        "memory": {"max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                                   "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}}

    # ------------------------------------------------------------------ kernel profile
    if "profile" in parts_set:
        from torch.profiler import ProfilerActivity, profile
        prof_out = {}
        for name in dict.fromkeys(i.split(":")[0] for i in plan.split(",")):
            lm = get_lm(name)
            with torch.inference_mode():
                lm.prefill(ids0)
                lm.step(30)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    lm.step(20)
                    torch.cuda.synchronize()
            ka = prof.key_averages()
            rows = sorted(ka, key=lambda e: -getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)))
            tot = sum(getattr(e, "device_time_total", 0) for e in ka)
            prof_out[name] = {"total_ms_per_step": tot / 20 / 1e3, "top": [
                {"name": e.key[:120], "ms_per_step": getattr(e, "device_time_total", 0) / 20 / 1e3, "calls_per_step": e.count / 20}
                for e in rows[:25]]}
            print(name, json.dumps(prof_out[name], indent=1), flush=True)
        out["profile"] = prof_out

    # ------------------------------------------------------------------ Mimi streaming exactness without TF32
    if "mimicheck" in parts_set:
        lm = get_lm(variant)
        tts = StreamingTTS(lm, tok, smimi=smimi)
        r = tts.run(build_prompt(tmap["hi_medium"]), 1000, mimi_mode="stream", keep_audio=True)
        codes = model.audio_tokens_to_codes(r["tokens"]).cuda()

        def snr(ref, x):
            n = min(len(ref), len(x))
            e = x[:n] - ref[:n]
            return {"snr_db": float(10 * torch.log10(ref[:n].pow(2).sum() / e.pow(2).sum().clamp_min(1e-30))),
                    "max_abs_diff": float(e.abs().max())}
        mc = {}
        with torch.inference_mode():
            full_tf32 = mimi.decode(codes).audio_values[0, 0].float()
            mc["graph_stream_vs_full (tf32 on, as deployed)"] = snr(full_tf32, r["_pcm"].cuda())
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            full_fp32 = mimi.decode(codes).audio_values[0, 0].float()
            se = StreamingMimi(mimi, use_graphs=False)
            stream_fp32 = se.decode_all(codes)
            mc["eager_stream_vs_full (tf32 off)"] = snr(full_fp32, stream_fp32)
            mc["full_tf32_vs_full_fp32 (noise floor)"] = snr(full_fp32, full_tf32)
            mc["graph_stream_tf32_vs_full_fp32"] = snr(full_fp32, r["_pcm"].cuda())
            torch.backends.cudnn.allow_tf32 = True
        print("mimicheck", json.dumps(mc), flush=True)
        out["mimicheck"] = mc

    # ------------------------------------------------------------------ leading silence
    if "silence" in parts_set:
        lm = get_lm(variant)
        tts = StreamingTTS(lm, tok, smimi=smimi)
        sil: dict = {"variants": {}}
        cases = {
            "standard": dict(),
            "no_description": dict(description=False),
            "speaker_Aisha": dict(speaker="Aisha"),
        }
        for cname, kw in cases.items():
            rows = []
            for tid in ["en_short", "hinglish_short", "hi_short"]:
                for s in range(5):
                    r = tts.run(build_prompt(tmap[tid], **kw), 3000 + s, mimi_mode="stream")
                    rows.append({"text": tid, "seed": 3000 + s, "first_nonsilent_frame": r["first_nonsilent_frame"],
                                 "dbfs_first6": [round(d, 1) for d in r["frame_dbfs"][:6]],
                                 "codes_first3": [r["tokens"][i * Q:(i + 1) * Q] for i in range(3)],
                                 "frames": r["frames"]})
            sil["variants"][cname] = {"rows": rows, "first_nonsilent_frame": stats([x["first_nonsilent_frame"] for x in rows])}
            print(cname, sil["variants"][cname]["first_nonsilent_frame"], flush=True)
        # codec warm-up: decode a speech segment from a fresh decoder state vs in context
        r = tts.run(build_prompt(tmap["en_medium"]), 3100, mimi_mode="stream", keep_audio=True)
        codes = model.audio_tokens_to_codes(r["tokens"]).cuda()
        k0 = (r["first_nonsilent_frame"] or 0) + 5
        with torch.inference_mode():
            ctx = smimi.decode_all(codes)
            fresh = smimi.decode_all(codes[..., k0:k0 + 6])
        db = lambda x: [round(float(20 * torch.log10(f.pow(2).mean().sqrt().clamp_min(1e-9))), 1) for f in x.view(-1, 1920)]  # noqa: E731
        sil["codec_warmup"] = {"start_frame": k0, "in_context_dbfs": db(ctx[k0 * 1920:(k0 + 6) * 1920]), "fresh_state_dbfs": db(fresh)}
        print("codec warm-up", sil["codec_warmup"], flush=True)
        out["silence"] = sil

    # ------------------------------------------------------------------ ASR CER (baseline vs optimized)
    if "asr" in parts_set:
        try:
            out["asr"] = asr_check()
        except Exception as e:  # never lose the other parts
            import traceback
            traceback.print_exc()
            out["asr_error"] = repr(e)[:2000]

    volume.commit()  # keep inductor / triton caches for the next run
    out = json.loads(json.dumps(out))  # guarantee plain JSON types
    out["wavs"] = wavs
    return out


@app.local_entrypoint()
def main(parts: str = "quality,variants", runs: int = 3, warmup: int = 3, variant: str = "c_kvb_f32_pf",
         n_tf_texts: int = 3, seeds: int = 2, out: str = "results/fast_smoke.json",
         plan: str = "eager:hf_first,graphs:hf_first,graphs:stream,compile:hf_first,compile:stream",
         tf_extra: str = ""):
    t0 = time.time()
    data = run_fast.remote(parts=parts, runs=runs, warmup=warmup, variant=variant, n_tf_texts=n_tf_texts, seeds=seeds,
                           plan=plan, tf_extra=tf_extra)
    wavs = data.pop("wavs")
    data["wall_clock_remote_call_s"] = time.time() - t0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    sd = Path("samples/optimized")
    sd.mkdir(parents=True, exist_ok=True)
    for k, b in wavs.items():
        (sd / k).write_bytes(b)
    print(f"\nwrote {out} and {len(wavs)} wavs to {sd}/ (remote call {data['wall_clock_remote_call_s']:.0f}s)")
    if "bench" in data:
        print(f'{"group":24s} {"TTFA p50":>9s} {"p95":>8s} {"speech":>8s} {"prefill":>8s} {"tok/s":>7s} {"RTFx":>6s} {"mimi ms":>8s} {"stall-free":>10s}')
        for g, s in data["bench"]["summary"].items():
            print(f'{g:24s} {s["ttfa_ms"]["p50"]:9.1f} {s["ttfa_ms"]["p95"]:8.1f} {s["ttfa_nonsilent_ms"]["p50"] or 0:8.1f} '
                  f'{s["prefill_ms"]["p50"]:8.1f} {s["decode_tok_per_s"]["p50"]:7.1f} {s["rtfx"]["p50"]:6.2f} '
                  f'{s["mimi_frame_ms"]["p50"]:8.2f} {s["stall_free"]:>10s}')
