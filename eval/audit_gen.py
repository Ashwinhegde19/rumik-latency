"""Matched-seed quality audit: stock HF path vs the optimized streaming path, same H100, texts and seeds.

    .venv/bin/modal run eval/audit_gen.py --smoke                                  # harness check
    .venv/bin/modal run eval/audit_gen.py --seeds 1000-1007 --tf-seeds 1000,1001   # full run

In one container:
  * stock path, as bench_modal.py / the model card: generate_audio (bf16, SDPA) + MimiModel.decode of the whole
    clip (fp32). Run by several worker processes sharing the GPU (eval/audit_worker.py), plus the main process
    once it is free. Forward hooks record the stock incremental logits for the teacher-forcing references.
  * optimized path, exactly as benchmarked: FastLM(c_kvb_f32_pf) + StreamingMimi + StreamingTTS.run(..., "stream").
    The same tokens are also decoded with the HF full-clip decoder to measure the streaming codec on its own.
  * determinism checks (optimized rerun with and without the concurrent Mimi stream; one stock job twice).
  * teacher forcing on the stock sequences: stock incremental (recorded during generation), stock full forward,
    optimized FastLM, FastLM without KV-length buckets, and an fp32 copy of the model as a high-precision reference.

Nothing in rumik_fast.py / bench_*.py is modified; this file only imports rumik_fast.
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "rumik-ai/rumik-oss-1"
REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"  # same pin as bench_modal.py / bench_fast.py
HF_CACHE = "/hf-cache"
SR = 24000

# identical to bench_fast.py (checked against its source in the local entrypoint)
GEN_KW = dict(max_new_tokens=2048, min_new_tokens=8, temperature=0.8, top_k=30)
KVB = (256, 512, 1024)
PFB = (32, 64, 128, 256)
FINAL = dict(use_compile=True, kv_buckets=KVB, attn="f32", compile_prefill=True, prefill_buckets=PFB)
NO_KVB = dict(use_compile=True, attn="f32", compile_prefill=True, prefill_buckets=PFB)  # same, KV buckets off

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install(requirements=[str(ROOT / "requirements.txt")])
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "TORCHINDUCTOR_CACHE_DIR": f"{HF_CACHE}/inductor-cache", "TRITON_CACHE_DIR": f"{HF_CACHE}/triton-cache"})
    .add_local_file(str(ROOT / "texts.json"), "/root/texts.json")
    .add_local_file(str(ROOT / "rumik_fast.py"), "/root/rumik_fast.py")
    .add_local_file(str(ROOT / "eval" / "audit_worker.py"), "/root/audit_worker.py")
)
volume = modal.Volume.from_name("rumik-hf-cache", create_if_missing=False)
app = modal.App("rumik-quality-audit", image=image)


@app.function(gpu="H100", cpu=8.0, memory=65536, volumes={HF_CACHE: volume}, timeout=3600)
def run_audit(cfg: dict) -> dict:
    import copy
    import io
    import pickle
    import platform
    import subprocess
    import sys
    import traceback
    from time import perf_counter

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    sys.path.insert(0, "/root")
    from audit_worker import StockRunner, run_claimed_jobs
    from rumik_fast import FastLM, StreamingMimi, StreamingTTS

    t_start = perf_counter()
    log = lambda *a: print(f"[{perf_counter() - t_start:7.1f}s]", *a, flush=True)  # noqa: E731
    errors: list = []
    root = snapshot_download(REPO_ID, revision=REVISION, token=False)
    texts = json.loads(Path("/root/texts.json").read_text())
    tmap = {t["id"]: t for t in texts}

    def build_prompt(t: dict) -> str:  # bench_modal.py / bench_fast.py prompt format
        return f'<text>{t["speaker"]}: <description="{t["description"]}"> {t["text"]}<audio>'

    text_ids = cfg["texts"] or [t["id"] for t in texts]
    seeds, tf_seeds = list(cfg["seeds"]), set(cfg["tf_seeds"])
    order = sorted(text_ids, key=lambda i: -len(tmap[i]["text"]))  # longest first (load balance)
    jobs = [{"key": f"{tid}_seed{s}", "text_id": tid, "seed": s, "prompt": build_prompt(tmap[tid]),
             "record": s in tf_seeds} for tid in order for s in seeds]
    dup = dict(jobs[-1], key=jobs[-1]["key"] + "_dup", record=False)  # stock run-to-run determinism
    work = Path("/tmp/audit")
    (work / "claims").mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    (work / "jobs.json").write_text(json.dumps(jobs + [dup]))
    procs = [subprocess.Popen([sys.executable, "/root/audit_worker.py", "--rank", str(r), "--root", root,
                               "--work", str(work), "--gen-kw", json.dumps(GEN_KW), "--delay", str(8 * r)])
             for r in range(cfg["workers"])]
    log(f"{len(jobs)} jobs, {cfg['workers']} stock workers started")

    # ------------------------------------------------------------------ main process: optimized path
    tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    mimi = MimiModel.from_pretrained(f"{root}/codec").eval().cuda()
    cfgm = model.config
    END = int(cfgm.audio_end_token_id)
    kw = dict(max_new_tokens=GEN_KW["max_new_tokens"], min_new_tokens=GEN_KW["min_new_tokens"],
              temperature=GEN_KW["temperature"], top_k=GEN_KW["top_k"])
    smimi = StreamingMimi(mimi)
    t = perf_counter()
    lm = FastLM(model, **FINAL, **kw)
    torch.cuda.synchronize()
    build_final_s = perf_counter() - t
    tts = StreamingTTS(lm, tok, smimi=smimi)
    for i in range(2):  # warm-up, as the benchmark does (not recorded)
        tts.run(build_prompt(tmap["en_medium"]), 10_000 + i, mimi_mode="stream")
    log(f"optimized path ready (FastLM build {build_final_s:.1f}s)")

    env = {"gpu": torch.cuda.get_device_name(0),
           "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "torch": str(torch.__version__), "cuda": torch.version.cuda, "transformers": str(transformers.__version__),
           "python": platform.python_version(), "revision": REVISION, "gen_kw": GEN_KW,
           "final_variant": {k: list(v) if isinstance(v, tuple) else v for k, v in FINAL.items()},
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}

    def wav_bytes(x: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, np.clip(x, -1, 1), SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def codec_check(full: np.ndarray, stream: np.ndarray) -> dict:
        n = min(len(full), len(stream))
        f, s = full[:n].astype(np.float64), stream[:n].astype(np.float64)
        e = s - f
        snr = lambda a, b: float(10 * np.log10(max(np.sum(a ** 2), 1e-30) / max(np.sum(b ** 2), 1e-30)))  # noqa: E731
        per_s = [snr(f[i:i + SR], e[i:i + SR]) for i in range(0, n - SR // 2, SR)]
        nf = n // 1920
        E = (e[: nf * 1920] ** 2).reshape(nf, 1920)
        prof = E.mean(0)
        return {"len_full": len(full), "len_stream": len(stream), "snr_db": snr(f, e),
                "snr_db_first10s": snr(f[: 10 * SR], e[: 10 * SR]),
                "snr_db_after10s": snr(f[10 * SR:], e[10 * SR:]) if n > 10 * SR + SR // 2 else None,
                "snr_db_per_second": per_s, "max_abs_diff": float(np.abs(e).max()),
                "err_energy_boundary_ratio": float(prof[[0, 1919]].mean() / prof.mean()),
                "err_energy_pct_rank_boundary": float((prof < prof[[0, 1919]].mean()).mean())}

    opt: dict = {}
    wavs: dict = {}
    t = perf_counter()
    for j in jobs:
        try:
            r = tts.run(j["prompt"], j["seed"], mimi_mode="stream", keep_audio=True)
            pcm = r.pop("_pcm").numpy()
            with torch.inference_mode():
                codes = model.audio_tokens_to_codes(r["tokens"]).cuda()
                full = mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()
            wavs[f'{j["key"]}_optimized.wav'] = wav_bytes(pcm)
            toks = [int(x) for x in r["tokens"]]
            opt[j["key"]] = {"tokens": toks, "frames": r["frames"], "stopped_by": r["stopped_by"],
                             "n_audio_tokens": r["audio_tokens"], "ttfa_s": r["ttfa_s"], "total_s": r["total_s"],
                             "first_nonsilent_frame": r.get("first_nonsilent_frame"),
                             "codec_check": codec_check(full, pcm)}
        except Exception:
            errors.append({"where": f'opt {j["key"]}', "err": traceback.format_exc()[-3000:]})
    log(f"optimized generation done: {len(opt)} clips in {perf_counter() - t:.1f}s")

    # determinism of the optimized path: rerun the first seed of every text with and without the Mimi stream
    det = []
    for tid in order:
        key = f"{tid}_seed{seeds[0]}"
        if key not in opt:
            continue
        try:
            p = build_prompt(tmap[tid])
            r_stream = tts.run(p, seeds[0], mimi_mode="stream", keep_audio=True)
            r_none = tts.run(p, seeds[0], mimi_mode="none")
            a = opt[key]["tokens"]
            b2 = wav_bytes(r_stream.pop("_pcm").numpy())
            i1, _ = sf.read(io.BytesIO(wavs[f"{key}_optimized.wav"]), dtype="int16")
            i2, _ = sf.read(io.BytesIO(b2), dtype="int16")
            det.append({"key": key, "rerun_stream_tokens_equal": [int(x) for x in r_stream["tokens"]] == a,
                        "rerun_lm_only_tokens_equal": [int(x) for x in r_none["tokens"]] == a,
                        "rerun_wav_bytes_identical": b2 == wavs[f"{key}_optimized.wav"],
                        "rerun_pcm_max_abs_diff_int16": int(np.abs(i1.astype(np.int32) - i2).max())
                        if len(i1) == len(i2) else None})
        except Exception:
            errors.append({"where": f"determinism {key}", "err": traceback.format_exc()[-3000:]})
    log("determinism:", json.dumps(det))

    # ------------------------------------------------------------------ main joins the stock work
    runner = StockRunner(tok, model, mimi)
    n_main = run_claimed_jobs(runner, work, GEN_KW, "main")
    log(f"main process ran {n_main} stock jobs; waiting for workers")
    deadline = perf_counter() + cfg.get("stock_deadline_s", 2400)
    keys = [j["key"] for j in jobs] + [dup["key"]]
    while perf_counter() < deadline:
        have = {p.stem for p in (work / "out").glob("*.pkl")}
        if all(k in have for k in keys) or all(pr.poll() is not None for pr in procs):
            break
        time.sleep(2)
    worker_status = {}
    for r_, pr in enumerate(procs):
        worker_status[r_] = {"returncode": pr.poll(),
                             "fatal": (work / "out" / f"fatal_{r_}.txt").read_text()[-2000:]
                             if (work / "out" / f"fatal_{r_}.txt").exists() else None,
                             "done_jobs": (work / "out" / f"done_{r_}").read_text()
                             if (work / "out" / f"done_{r_}").exists() else None}
        if pr.poll() is None:
            pr.kill()
    stock: dict = {}
    tf_refs: dict = {}
    for k in keys:
        f = work / "out" / f"{k}.pkl"
        if not f.exists():  # fallback: run it here (still the stock path)
            try:
                res = runner.run_job(next(j for j in jobs + [dup] if j["key"] == k), GEN_KW)
                res["worker"] = "main-fallback"
            except Exception:
                errors.append({"where": f"stock fallback {k}", "err": traceback.format_exc()[-3000:]})
                continue
        else:
            res = pickle.loads(f.read_bytes())
        if "error" in res:
            errors.append({"where": f"stock {k}", "err": res["error"]})
            continue
        wb = res.pop("wav")
        if "tf_ref" in res:
            tf_refs[k] = res.pop("tf_ref")
        if k == dup["key"]:
            stock["_dup"] = res
            continue
        wavs[f"{k}_stock.wav"] = wb
        stock[k] = res
    runner.close()
    torch.cuda.synchronize()
    log(f"stock results: {len(stock)} (+dup) ; workers {worker_status}")
    dkey = dup["key"][:-4]
    stock_det = {"key": dkey, "tokens_equal": stock.get("_dup", {}).get("tokens") == stock.get(dkey, {}).get("tokens"),
                 "workers": [stock.get("_dup", {}).get("worker"), stock.get(dkey, {}).get("worker")]}

    # ------------------------------------------------------------------ teacher forcing
    tf_out: dict = {}
    try:
        aud = model.audio_token_ids(device="cuda")
        dev = torch.device("cuda")

        @torch.inference_mode()
        def full_forward(m, ids, seq):
            x = torch.tensor([ids + seq], device=dev)
            o = m.model(input_ids=x)
            h = o.last_hidden_state[0, len(ids) - 1:]                       # [N, H] post-norm
            lg = m.lm_head(h).index_select(-1, aud) * m.logit_scale
            st = m.stop_predictor(h[:, None, :]).reshape(-1)
            return lg.float(), st.float()

        @torch.inference_mode()
        def fast_tf(lmx, ids, seq):
            lmx.prefill(ids)
            L = [lmx.logits_buf.clone()]
            S = [lmx.stop_head(lmx.hidden_buf).reshape(-1)[0].float()]
            for t_ in seq:
                lmx.forward_token(t_)
                L.append(lmx.logits_buf.clone())
                S.append(lmx.stop_head(lmx.hidden_buf).reshape(-1)[0].float())
            return torch.stack(L).float(), torch.stack(S)

        K_MC = 256

        def samp_dist(L, steps):
            s = L.clone()
            s[steps < GEN_KW["min_new_tokens"], -1] = float("-inf")
            s = s / GEN_KW["temperature"]
            kth = s.topk(GEN_KW["top_k"], -1).values[:, -1:]
            s = s.masked_fill(s < kth, float("-inf"))
            return torch.softmax(s, -1)

        def compare(ref, test, T_prompt):
            (lr, sr), (lt, st) = ref, test
            N = min(lr.shape[0], lt.shape[0])
            lr, lt, sr, st = lr[:N], lt[:N], sr[:N], st[:N]
            steps = torch.arange(N, device=dev)
            res = {}
            for T in (1.0, 0.8):
                lp, lq = torch.log_softmax(lr / T, -1), torch.log_softmax(lt / T, -1)
                res[f"kl_T{T}"] = (lp.exp() * (lp - lq)).sum(-1)
            P, Q = samp_dist(lr, steps), samp_dist(lt, steps)
            res["tvd"] = 0.5 * (P - Q).abs().sum(-1)
            g = torch.Generator(device=dev).manual_seed(1234)  # same noise for every comparison
            fl = []
            for a in range(0, N, 16):
                b = min(N, a + 16)
                E = torch.empty(b - a, K_MC, P.shape[1], device=dev).exponential_(generator=g)
                fl.append(((P[a:b, None] / E).argmax(-1) != (Q[a:b, None] / E).argmax(-1)).float().mean(-1))
            res["flip"] = torch.cat(fl)
            res["top1"] = (lr.argmax(-1) == lt.argmax(-1)).float()
            ta, tb = lr.topk(30, -1).indices, lt.topk(30, -1).indices
            res["top30_overlap"] = (ta[:, :, None] == tb[:, None, :]).any(-1).float().mean(-1)
            ok = (steps >= 8) & torch.isfinite(sr) & torch.isfinite(st)
            stop_agree = ((sr > 0) == (st > 0))[ok].float()
            dprob = (torch.sigmoid(sr) - torch.sigmoid(st)).abs()[ok]
            keep = ("kl_T0.8", "tvd", "flip", "top1")
            out = {"positions": N, "per_pos": {k: [round(float(x), 6) for x in res[k].tolist()] for k in keep},
                   "stop_agree": float(stop_agree.mean()) if ok.any() else None,
                   "stop_prob_absdiff_mean": float(dprob.mean()) if ok.any() else None,
                   "stop_prob_absdiff_max": float(dprob.max()) if ok.any() else None,
                   "stop_flip_positions": [int(x) for x in steps[ok][((sr > 0) != (st > 0))[ok]].tolist()]}
            out["mean"] = {k: float(v.mean()) for k, v in res.items()}
            return out

        tf_keys = [k for k in tf_refs]
        log(f"teacher forcing on {len(tf_keys)} stock sequences")
        lm_nokvb = None
        try:
            t = perf_counter()
            lm_nokvb = FastLM(model, **NO_KVB, **kw)
            torch.cuda.synchronize()
            log(f"built FastLM without KV buckets in {perf_counter() - t:.1f}s")
        except Exception:
            errors.append({"where": "build nokvb", "err": traceback.format_exc()[-3000:]})
        # high-precision reference: fp32 copy of the stock model (no TF32), full forward
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        ref32 = copy.deepcopy(model).float().eval()
        for k in tf_keys:
            s = stock[k]
            d = torch.load(io.BytesIO(tf_refs[k]))
            seq = [x for x in s["tokens"] if x != END]  # every token generate_audio fed back
            ids = tok(build_prompt(tmap[s["text_id"]])).input_ids
            L_inc = d["logits"].to(dev).float()
            S_inc = torch.full((L_inc.shape[0],), float("nan"), device=dev)
            so = d["stop_logits_from_step8"].to(dev).float()
            n_s = max(0, min(len(so), L_inc.shape[0] - 8))
            S_inc[8:8 + n_s] = so[:n_s]
            if L_inc.shape[0] != len(seq) + 1:
                errors.append({"where": f"tf {k}", "err": f"len mismatch {L_inc.shape[0]} vs {len(seq) + 1}"})
            paths = {"stock_inc": (L_inc, S_inc), "stock_full": full_forward(model, ids, seq),
                     "fast": fast_tf(lm, ids, seq), "fp32": full_forward(ref32, ids, seq)}
            if lm_nokvb is not None:
                paths["fast_nokvb"] = fast_tf(lm_nokvb, ids, seq)
            pairs = [("stock_inc", "stock_full"), ("stock_inc", "fast"), ("stock_inc", "fast_nokvb"),
                     ("stock_full", "fast"), ("fp32", "stock_inc"), ("fp32", "stock_full"), ("fp32", "fast"),
                     ("fp32", "fast_nokvb"), ("fast", "fast_nokvb")]
            steps_ = torch.arange(len(seq) + 1)
            pos_ = len(ids) - 1 + steps_  # absolute position of the token fed at this step (prefill: T-1)
            bucket = torch.where(steps_ == 0, torch.full_like(pos_, -1),
                                 torch.where(pos_ < 256, 256, torch.where(pos_ < 512, 512,
                                                                          torch.where(pos_ < 1024, 1024, 2304))))
            tf_out[k] = {"text_id": s["text_id"], "seed": s["seed"], "prompt_tokens": len(ids),
                         "bucket": bucket.tolist(),
                         "stock_stop_step": len(seq) if s["stopped_by"] == "stop_predictor" else None,
                         "cmp": {f"{a}|{b}": compare(paths[a], paths[b], len(ids))
                                 for a, b in pairs if a in paths and b in paths}}
            # stop decision of each path at the step where stock stopped
            fin = {}
            for name, (L_, S_) in paths.items():
                v = S_[len(seq)] if len(seq) < len(S_) else None
                fin[name] = float(v) if v is not None and torch.isfinite(v) else None
            tf_out[k]["stop_logit_at_stock_stop"] = fin
            m = {c: round(v["mean"]["kl_T0.8"], 6) for c, v in tf_out[k]["cmp"].items()}
            log(f"tf {k}: N={len(seq) + 1} kl_T0.8 {m}")
        del ref32
    except Exception:
        errors.append({"where": "teacher forcing", "err": traceback.format_exc()[-3000:]})

    out = {"env": env, "cfg": cfg, "jobs": [{k: v for k, v in j.items() if k != "prompt"} for j in jobs],
           "stock": {k: v for k, v in stock.items() if k != "_dup"}, "opt": opt,
           "determinism": {"optimized": det, "stock": stock_det}, "tf": tf_out, "errors": errors,
           "worker_status": {str(k): v for k, v in worker_status.items()}, "build_final_s": build_final_s,
           "wall_s": perf_counter() - t_start}
    out = json.loads(json.dumps(out))  # plain JSON types only
    volume.commit()  # keep inductor / triton caches
    out["wavs"] = wavs
    return out


def _bench_fast_constants() -> dict:
    """GEN_KW / VARIANTS from bench_fast.py, read from its source (not imported)."""
    tree = ast.parse((ROOT / "bench_fast.py").read_text())
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("GEN_KW", "KVB", "PFB", "VARIANTS")
                                                for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "bench_fast", "exec"), {"dict": dict}, ns)
    return ns


def _parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@app.local_entrypoint()
def main(seeds: str = "1000-1007", tf_seeds: str = "1000,1001", texts: str = "", workers: int = 4,
         out: str = "results/audit/gen.json", wav_dir: str = "results/audit/wavs", smoke: bool = False):
    ref = _bench_fast_constants()
    assert ref["GEN_KW"] == GEN_KW, (ref["GEN_KW"], GEN_KW)
    assert ref["VARIANTS"]["c_kvb_f32_pf"] == FINAL, (ref["VARIANTS"]["c_kvb_f32_pf"], FINAL)
    if smoke:
        seeds, tf_seeds, texts, workers = "1000", "1000", "en_short,hi_short", 2
        out, wav_dir = "results/audit/smoke_gen.json", "results/audit/smoke_wavs"
    cfg = {"seeds": _parse_seeds(seeds), "tf_seeds": _parse_seeds(tf_seeds),
           "texts": [t for t in texts.split(",") if t], "workers": workers}
    t0 = time.time()
    data = run_audit.remote(cfg)
    wavs = data.pop("wavs")
    data["wall_clock_remote_call_s"] = time.time() - t0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, ensure_ascii=False))
    wd = Path(wav_dir)
    wd.mkdir(parents=True, exist_ok=True)
    for k, b in wavs.items():
        (wd / k).write_bytes(b)
    print(f"wrote {out} and {len(wavs)} wavs to {wd}/ (remote call {data['wall_clock_remote_call_s']:.0f}s)")
    print("errors:", json.dumps(data["errors"], indent=1)[:4000])
    print("determinism:", json.dumps(data["determinism"]))
