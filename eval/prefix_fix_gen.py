#!/usr/bin/env python3
"""Prefix experiment: does rumik-oss-1 skip its leading silence when the silence codes are already in context?

Every generated sequence starts with ~3 near-silent frames (code 1049 at frame 0 in 320/320; first audible
speech typically at frame 3). Here the known silence frames are appended to the prompt and processed in
prefill, as if the model had already written them. They are decoded through the streaming Mimi decoder to
warm its state but never form part of the output audio. The model's own first generated frame is then
measured: is it speech (first_nonsilent_frame == 0) or more silence?

Conditions (prefix source = a real generated start from results/audit/gen.json, optimized path):
  k0   no prefix (control; same seeds as the audit, so tokens should reproduce the audit's optimized clips)
  k1   first 1 frame of prefix A        k2  first 2 frames of A        k3  all 3 frames of A
  k4   4 frames from a run that had >= 4 silent frames (longer than typical)
  k3B  3 frames from a second, median-typicality run (does the exact member of the "silence family" matter?)

    .venv/bin/modal run eval/prefix_fix_gen.py --smoke          # harness check (2 texts, 2 conditions, 1 seed)
    .venv/bin/modal run eval/prefix_fix_gen.py                  # full run (10 texts; 8 seeds k0/k3, 4 seeds others)

Nothing in rumik_fast.py is modified; the run loop below is a copy of StreamingTTS.run with the prefix added.
"""
from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "rumik-ai/rumik-oss-1"
REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"  # same pin as bench_fast.py / audit_gen.py
HF_CACHE = "/hf-cache"
SR = 24000
GEN_KW = dict(max_new_tokens=2048, min_new_tokens=8, temperature=0.8, top_k=30)
KVB = (256, 512, 1024)
PFB = (32, 64, 128, 256)
FINAL = dict(use_compile=True, kv_buckets=KVB, attn="f32", compile_prefill=True, prefill_buckets=PFB)

CONDS = {"k0": ("A", 0), "k1": ("A", 1), "k2": ("A", 2), "k3": ("A", 3), "k4": ("A4", 4), "k3B": ("B", 3)}
MAIN_CONDS = ("k0", "k3")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install(requirements=[str(ROOT / "requirements.txt")])
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "TORCHINDUCTOR_CACHE_DIR": f"{HF_CACHE}/inductor-cache", "TRITON_CACHE_DIR": f"{HF_CACHE}/triton-cache"})
    .add_local_file(str(ROOT / "texts.json"), "/root/texts.json")
    .add_local_file(str(ROOT / "rumik_fast.py"), "/root/rumik_fast.py")
)
volume = modal.Volume.from_name("rumik-hf-cache", create_if_missing=False)
app = modal.App("rumik-prefix-fix", image=image)


def pick_prefixes(gen: dict) -> dict:
    """Real generated starts from the audit's optimized runs, ranked by per-position typicality."""
    opt = gen["opt"]
    toks = {k: r["tokens"] for k, r in opt.items()}
    n = len(toks)
    pos = [collections.Counter() for _ in range(32)]
    for t in toks.values():
        for i in range(32):
            pos[i][t[i]] += 1

    def typ(t, nf):  # log-probability of the start under the per-position empirical distribution
        return sum(math.log(pos[i][t[i]] / n) for i in range(nf * 8))

    eq3 = sorted((k for k, r in opt.items() if r["first_nonsilent_frame"] == 3), key=lambda k: -typ(toks[k], 3))
    ge4 = sorted((k for k, r in opt.items() if (r["first_nonsilent_frame"] or 0) >= 4), key=lambda k: -typ(toks[k], 4))
    out = {"A": {"source": eq3[0], "ids": toks[eq3[0]][:24]},
           "B": {"source": eq3[len(eq3) // 2], "ids": toks[eq3[len(eq3) // 2]][:24]},
           "A4": {"source": ge4[0], "ids": toks[ge4[0]][:32]}}
    out["_pool"] = {"n_runs": n, "n_first_nonsilent_eq3": len(eq3), "n_first_nonsilent_ge4": len(ge4)}
    return out


@app.function(gpu="H100", cpu=4.0, memory=32768, volumes={HF_CACHE: volume}, timeout=3600)
def run_gen(cfg: dict) -> dict:
    import io
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
    from rumik_fast import FastLM, FrameAssembler, StreamingMimi, StreamingTTS

    t_start = perf_counter()
    log = lambda *a: print(f"[{perf_counter() - t_start:7.1f}s]", *a, flush=True)  # noqa: E731
    errors: list = []
    root = snapshot_download(REPO_ID, revision=REVISION, token=False)
    texts = json.loads(Path("/root/texts.json").read_text())
    tmap = {t["id"]: t for t in texts}

    def build_prompt(t: dict) -> str:  # bench_modal.py / bench_fast.py / audit_gen.py prompt format
        return f'<text>{t["speaker"]}: <description="{t["description"]}"> {t["text"]}<audio>'

    tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    mimi = MimiModel.from_pretrained(f"{root}/codec").eval().cuda()
    cfgm = model.config
    kw = dict(max_new_tokens=GEN_KW["max_new_tokens"], min_new_tokens=GEN_KW["min_new_tokens"],
              temperature=GEN_KW["temperature"], top_k=GEN_KW["top_k"])
    smimi = StreamingMimi(mimi)
    t = perf_counter()
    lm = FastLM(model, **FINAL, **kw)
    torch.cuda.synchronize()
    build_s = perf_counter() - t
    tts = StreamingTTS(lm, tok, smimi=smimi)
    Q = lm.Q
    # room for the prefix frames in front of a maximal generation
    mf = lm.max_new // Q + 2 + 8
    tts.pin_codes = torch.zeros(mf, 1, Q, 1, dtype=torch.long).pin_memory()
    tts.pin_pcm = torch.zeros(mf, tts.spf, dtype=torch.float32).pin_memory()
    for i in range(2):  # warm-up, as the benchmark does (not recorded)
        tts.run(build_prompt(tmap["en_medium"]), 10_000 + i, mimi_mode="stream")
    log(f"optimized path ready (FastLM build {build_s:.1f}s)")

    prefixes = cfg["prefixes"]
    for name, p in prefixes.items():
        if name.startswith("_"):
            continue
        for i, tid in enumerate(p["ids"]):
            assert lm.first <= tid <= lm.last and (tid - lm.first) % Q == i % Q, (name, i, tid)

    @torch.inference_mode()
    def run_prefixed(prompt: str, seed: int, prefix_ids: list[int]) -> dict:
        """StreamingTTS.run with `prefix_ids` appended to the prompt. Prefix frames warm the decoder only."""
        n_pre = len(prefix_ids) // Q
        assert len(prefix_ids) == n_pre * Q
        torch.cuda.synchronize()
        t0 = perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        ids = tok(prompt).input_ids + list(prefix_ids)
        torch.manual_seed(seed)
        with torch.cuda.stream(tts.mstream):
            tts.smimi.reset()
            for j in range(n_pre):
                codes = [(x - lm.first) // Q for x in prefix_ids[j * Q:(j + 1) * Q]]
                tts.pin_codes[j].view(-1).copy_(torch.tensor(codes))
                pcm = tts.smimi.decode_frame(tts.pin_codes[j])
                tts.pin_pcm[j].copy_(pcm, non_blocking=True)
        lm.prefill(ids)
        ev_pre = torch.cuda.Event(enable_timing=True)
        ev_pre.record()
        n_sampled = 1
        chunk_ev: list = []
        max_new = lm.max_new

        def enqueue_chunk(c):
            nonlocal n_sampled
            a, b = Q * c, min(Q * (c + 1), max_new)
            lm.step(b - n_sampled)
            n_sampled = b
            lm.copy_out(a, b)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            chunk_ev.append(e)

        n_chunks_max = (max_new + Q - 1) // Q
        next_chunk = 0
        while next_chunk <= tts.lookahead and next_chunk < n_chunks_max:
            enqueue_chunk(next_chunk)
            next_chunk += 1

        asm = FrameAssembler(lm.first, lm.last, Q)
        tokens: list[int] = []
        frames_codes: list[list[int]] = []
        stopped_by = "max_new_tokens"
        frame_ev: list = []
        t_first_audio = None
        first_chunk_decode = None
        n_frames = n_pre
        c = 0
        done = False
        while not done and c < len(chunk_ev):
            chunk_ev[c].synchronize()
            recs = lm.pin_rec[Q * c: min(Q * (c + 1), max_new)].tolist()
            if next_chunk < n_chunks_max:
                enqueue_chunk(next_chunk)
                next_chunk += 1
            new_frames = []
            for r in recs:
                if r == -1:
                    stopped_by, done = "stop_predictor", True
                    break
                tokens.append(r)
                if r == lm.end_id:
                    stopped_by, done = "audio_end_token", True
                    break
                f = asm.push(r)
                if f is not None:
                    new_frames.append(f)
            for f in new_frames:
                kf = n_frames
                frames_codes.append(f)
                tts.pin_codes[kf].view(-1).copy_(torch.tensor(f))
                with torch.cuda.stream(tts.mstream):
                    es, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    es.record()
                    pcm = tts.smimi.decode_frame(tts.pin_codes[kf])
                    ee.record()
                    tts.pin_pcm[kf].copy_(pcm, non_blocking=True)
                frame_ev.append((es, ee))
                if kf == n_pre:
                    ee.synchronize()
                    t_first_audio = perf_counter() - t0
                    first_chunk_decode = es.elapsed_time(ee) / 1e3
                n_frames += 1
            c += 1
        t_last_tokens = perf_counter() - t0
        torch.cuda.synchronize()
        t_end = perf_counter() - t0

        ms = lambda e: ev0.elapsed_time(e) / 1e3  # noqa: E731
        chunk_t = [ms(e) for e in chunk_ev[:c]]
        n_gen = n_frames - n_pre
        pcm_gen = tts.pin_pcm[n_pre:n_frames]
        dbfs = (20 * torch.log10(pcm_gen.pow(2).mean(-1).sqrt().clamp_min(1e-9))).tolist()
        pre_dbfs = (20 * torch.log10(tts.pin_pcm[:n_pre].pow(2).mean(-1).sqrt().clamp_min(1e-9))).tolist()
        k_ns = next((i for i, d in enumerate(dbfs) if d > tts.silence_dbfs), None)
        ready = [ms(ee) for _, ee in frame_ev]
        gpu_ms = [es.elapsed_time(ee) for es, ee in frame_ev]
        lateness = [ready[i] - (ready[0] + 0.08 * i) for i in range(n_gen)] if ready else [0.0]
        res = {
            "seed": seed, "k": n_pre, "prompt_tokens": len(ids) - len(prefix_ids), "prefix_tokens": len(prefix_ids),
            "audio_tokens": len([x for x in tokens if x != lm.end_id]), "frames": n_gen,
            "audio_seconds": n_gen * 0.08, "stopped_by": stopped_by,
            "prefill_s": ms(ev_pre), "first_frame_tokens_s": chunk_t[0] if chunk_t else None,
            "ttfa_s": t_first_audio, "first_chunk_decode_s": first_chunk_decode,
            "gen_total_s": t_last_tokens, "total_s": t_end, "tokens": tokens,
            "frame_dbfs": dbfs, "prefix_frame_dbfs": pre_dbfs,
            "first_nonsilent_frame": k_ns,
            "ttfa_nonsilent_s": ready[k_ns] if k_ns is not None else None,
            "first4_cb0": [f[0] for f in frames_codes[:4]],
            "first_frame_codes": frames_codes[0] if frames_codes else None,
            "mimi_frame_ms_p50": float(sorted(gpu_ms)[len(gpu_ms) // 2]) if gpu_ms else None,
            "max_lateness_s": max(lateness), "stall_free": max(lateness) <= 1e-3,
            "rtfx": (n_gen * 0.08) / t_end,
        }
        if len(chunk_t) > 3:
            res["decode_tok_per_s"] = Q * (len(chunk_t) - 2) / (chunk_t[-2] - chunk_t[0])
        res["_pcm"] = pcm_gen.reshape(-1).clone()
        return res

    def wav_bytes(x: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, np.clip(x, -1, 1), SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # ------------------------------------------------------------------ determinism: k0 loop == StreamingTTS.run
    det = {}
    try:
        j0 = cfg["jobs"][0]
        p0 = build_prompt(tmap[j0["text_id"]])
        r_orig = tts.run(p0, j0["seed"], mimi_mode="stream")
        r_mine = run_prefixed(p0, j0["seed"], [])
        det = {"job": j0["text_id"], "seed": j0["seed"], "tokens_equal": r_orig["tokens"] == r_mine["tokens"],
               "n_tokens": (len(r_orig["tokens"]), len(r_mine["tokens"])),
               "first_nonsilent_frame": (r_orig.get("first_nonsilent_frame"), r_mine["first_nonsilent_frame"])}
        log(f"determinism (k0 loop vs StreamingTTS.run): {det}")
    except Exception:
        errors.append({"where": "determinism", "err": traceback.format_exc()[-3000:]})

    # ------------------------------------------------------------------ jobs
    results: list = []
    wavs: dict = {}
    for i, j in enumerate(cfg["jobs"]):
        try:
            pname, k = CONDS[j["cond"]]
            prefix_ids = prefixes[pname]["ids"][: k * Q]
            r = run_prefixed(build_prompt(tmap[j["text_id"]]), j["seed"], prefix_ids)
            pcm = r.pop("_pcm").numpy()
            key = f'{j["text_id"]}_seed{j["seed"]}_{j["cond"]}'
            wavs[f"{key}.wav"] = wav_bytes(pcm)
            r.update({"key": key, "cond": j["cond"], "text_id": j["text_id"], "prefix": pname})
            results.append(r)
            if i % 10 == 0 or i == len(cfg["jobs"]) - 1:
                log(f"{i + 1}/{len(cfg['jobs'])} {key}: frames {r['frames']} first_nonsilent {r['first_nonsilent_frame']} "
                    f"cb0 {r['first4_cb0']} ttfa_ns {r['ttfa_nonsilent_s']}")
        except Exception:
            errors.append({"where": f"job {j}", "err": traceback.format_exc()[-3000:]})

    env = {"gpu": torch.cuda.get_device_name(0),
           "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "transformers": str(transformers.__version__),
           "python": platform.python_version(), "revision": REVISION, "gen_kw": GEN_KW, "final_variant": FINAL,
           "silence_dbfs": tts.silence_dbfs, "build_final_s": build_s}
    plain = json.loads(json.dumps({"env": env, "cfg": cfg, "determinism": det, "results": results, "errors": errors,
                                   "wall_s": perf_counter() - t_start}, default=str))  # JSON types only: the client has no torch
    plain["wavs"] = wavs
    return plain


def _seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


@app.local_entrypoint()
def main(smoke: bool = False, out_dir: str = "results/prefix_fix", seeds_main: str = "1000-1007",
         seeds_side: str = "1000-1003", texts: str = "", conds: str = ""):
    gen = json.loads((ROOT / "results" / "audit" / "gen.json").read_text())
    prefixes = pick_prefixes(gen)
    all_texts = [t["id"] for t in json.loads((ROOT / "texts.json").read_text())]
    text_ids = [x for x in texts.split(",") if x] or all_texts
    cond_ids = [x for x in conds.split(",") if x] or list(CONDS)
    sm, ss = _seeds(seeds_main), _seeds(seeds_side)
    if smoke:
        text_ids, cond_ids, sm, ss = ["en_short", "hinglish_short"], ["k0", "k3"], [1000], [1000]
        out_dir = out_dir.rstrip("/") + "_smoke"
    jobs = [{"text_id": tid, "cond": c, "seed": s}
            for tid in sorted(text_ids, key=lambda i: -len(i))  # any fixed order
            for c in cond_ids for s in (sm if c in MAIN_CONDS else ss)]
    cfg = {"prefixes": prefixes, "jobs": jobs, "conds": {c: CONDS[c] for c in cond_ids}}
    print(f"{len(jobs)} jobs; prefixes: " + ", ".join(f"{k} <- {v['source']}" for k, v in prefixes.items() if not k.startswith('_')))
    t0 = time.time()
    data = run_gen.remote(cfg)
    data["wall_clock_remote_call_s"] = time.time() - t0
    wavs = data.pop("wavs")
    od = ROOT / out_dir
    (od / "wavs").mkdir(parents=True, exist_ok=True)
    for k, b in wavs.items():
        (od / "wavs" / k).write_bytes(b)
    # k0 should reproduce the audit's optimized tokens for the same text/seed (same code, config, GPU class)
    opt = gen["opt"]
    rep = [(r["key"], r["tokens"] == opt[f'{r["text_id"]}_seed{r["seed"]}']["tokens"])
           for r in data["results"] if r["cond"] == "k0" and f'{r["text_id"]}_seed{r["seed"]}' in opt]
    data["k0_reproduces_audit_tokens"] = {"n": len(rep), "equal": sum(1 for _, e in rep if e)}
    (od / "gen.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"wrote {od / 'gen.json'} and {len(wavs)} wavs (remote {data['wall_clock_remote_call_s']:.0f}s, "
          f"gpu wall {data['wall_s']:.0f}s); determinism {data['determinism']}; k0 vs audit {data['k0_reproduces_audit_tokens']}; "
          f"errors {len(data['errors'])}")
    by = collections.defaultdict(list)
    for r in data["results"]:
        by[r["cond"]].append(r)
    for c in cond_ids:
        rs = by.get(c, [])
        if not rs:
            continue
        fns = collections.Counter(r["first_nonsilent_frame"] for r in rs)
        cb0 = collections.Counter(r["first4_cb0"][0] if r["first4_cb0"] else None for r in rs)
        tt = sorted(r["ttfa_nonsilent_s"] for r in rs if r["ttfa_nonsilent_s"] is not None)
        print(f"  {c:4s} n={len(rs):3d} first_nonsilent_frame {dict(sorted(fns.items(), key=lambda kv: (kv[0] is None, kv[0])))} "
              f"first-frame cb0 top {cb0.most_common(3)} ttfa_nonsilent p50 {tt[len(tt) // 2] * 1e3 if tt else None:.0f} ms")
    if data["errors"]:
        print(json.dumps(data["errors"])[:3000])
