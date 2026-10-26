"""Does the Mimi ENCODER emit a fixed first code (codebook 0 = 1049) for any input?

rumik-oss-1 always starts its generated token stream with Mimi code 1049 (codebook 0,
frame 0). Two hypotheses:
  H1 (encoder artefact): the causal encoder's frame-0 receptive field is mostly
     zero padding, so frame 0 of *every* training clip encoded to 1049 and the LM
     learned "frame 0 = 1049" as a rule, independent of content.
  H2 (honest silence): 1049 is simply the code for silence, and the training
     recordings genuinely started with silence.

Test: run MimiModel.encode (same codec checkpoint / revision as the TTS, fp32,
num_quantizers=8) on ~30 varied clips (rumik TTS output, real LibriSpeech speech,
synthetic tone / noise / clicks / silence) in several variants that control what
the first 80 ms contains (as-is, leading silence trimmed, cut mid-word at
0.9/1.3/2.1 s, 1 s silence prepended, time-reversed) and look at the frame-0 code.

    modal run eval/encoder_first_code.py            # ~3 min on a T4

Outputs: results/encoder_first_code.json and results/encoder_first_code.md
(the .md gets the auto-generated tables; the verdict paragraph is written by hand).
Also counts where codes 1049 / 127 / 958 occur in the 320 generated sequences of
results/audit/gen.json (frame 0 / 1 / 2 / elsewhere).
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import time
from pathlib import Path

import modal

REPO_ID = "rumik-ai/rumik-oss-1"
REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"  # pinned HF commit (same as bench_modal.py)
GPU = "T4"  # cheapest; the Mimi encoder is tiny
HF_CACHE = "/hf-cache"
SR = 24000
FRAME = 1920  # 80 ms at 24 kHz = one Mimi frame at 12.5 Hz
NQ = 8  # codebooks used by the TTS (config.num_quantizers)
TRIM_DBFS = -30.0  # "audible" threshold on |x| for trimming / cutting
MID_OFFSETS_S = (0.9, 1.3, 2.1)
LOUD80_DBFS = -30.0  # RMS of the first 80 ms above this => "loud onset"
FIRST_UNIT_ID, Q_TTS = 261008, 8  # token id = FIRST_UNIT_ID + code*8 + q

PROJECT = Path(__file__).resolve().parent.parent

# Same image definition as bench_modal.py (minus the texts.json mount) so the built layers are reused.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(requirements=[str(PROJECT / "requirements.txt")])
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)
volume = modal.Volume.from_name("rumik-hf-cache", create_if_missing=True)
app = modal.App("rumik-encoder-first-code", image=image)


# ----------------------------------------------------------------------------- audio helpers (numpy only)
def dbfs(v: float) -> float:
    return float(20.0 * math.log10(max(float(v), 1e-9)))


def seg_stats(x, n: int) -> dict:
    s = x[:n].astype("float64")
    if len(s) == 0:
        return {"rms_dbfs": -180.0, "peak_dbfs": -180.0}
    return {"rms_dbfs": dbfs(math.sqrt(float((s * s).mean()))), "peak_dbfs": dbfs(float(abs(s).max()))}


def make_variants(x, has_signal_thr: float = 10 ** (TRIM_DBFS / 20)) -> dict:
    """Variants of one 24 kHz float32 clip. Keys = variant names, values = float32 arrays."""
    import numpy as np

    out = {"asis": x}
    idx = np.flatnonzero(np.abs(x) > has_signal_thr)
    if len(idx) == 0:  # digital silence: no audible region, only the context-free variants make sense
        out["pad1s"] = np.concatenate([np.zeros(SR, np.float32), x])
        out["pad12f"] = np.concatenate([np.zeros(12 * FRAME, np.float32), x])
        return out
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    out["trim"] = x[i0:]  # starts at the first sample above -30 dBFS
    for off in MID_OFFSETS_S:  # start mid-word / mid-signal, `off` seconds into the audible region
        s = i0 + int(off * SR)
        if i1 - s >= SR // 2:
            out[f"mid_{off}"] = x[s:]
    out["pad1s"] = np.concatenate([np.zeros(SR, np.float32), x])  # 1 s digital silence prepended
    out["pad12f"] = np.concatenate([np.zeros(12 * FRAME, np.float32), x])  # 12 whole frames: frame 12 == asis frame 0
    e = i1 - int(0.9 * SR)  # reversed, starting 0.9 s before the last audible sample => starts loud
    if e - i0 >= SR // 2:
        out["rev_mid"] = np.ascontiguousarray(x[i0:e][::-1])
    out["rev_asis"] = np.ascontiguousarray(x[::-1])
    return out


# ----------------------------------------------------------------------------- remote: encode / decode on GPU
@app.function(gpu=GPU, volumes={HF_CACHE: volume}, timeout=30 * 60)
def encode_clips(clips: list[dict]) -> dict:
    """clips: [{name, group, pcm16 (np.int16 @ 24 kHz mono)}] -> per-variant codes + loudness."""
    import platform
    from time import perf_counter

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import MimiModel

    t0 = perf_counter()
    root = Path(snapshot_download(REPO_ID, revision=REVISION, token=False))
    mimi = MimiModel.from_pretrained(root / "codec").eval().cuda()  # fp32, as the TTS uses it
    torch.cuda.synchronize()
    load_s = perf_counter() - t0

    # Encoder left context at frame 0: sum over causal convs of padding_total * cumulative stride.
    stride, left_ctx = 1, 0
    conv_layers = []
    for name in mimi.encoder._mimiconv1d_layer_names:
        c = mimi.encoder.get_submodule(name)
        pt, st = int(c.padding_total), int(c.stride)
        left_ctx += pt * stride
        conv_layers.append({"layer": name, "kernel": int(c.kernel_size), "stride": st, "pad_left": pt, "pad_mode": c.pad_mode})
        stride *= st
    d = mimi.downsample
    left_ctx += int(d.padding_total) * stride
    conv_layers.append({"layer": "downsample", "kernel": int(d.kernel_size), "stride": int(d.stride),
                        "pad_left": int(d.padding_total), "pad_mode": d.pad_mode})
    stride *= int(d.stride)
    enc_info = {"hop_samples": stride, "left_context_samples": left_ctx, "left_context_ms": left_ctx / SR * 1e3,
                "frame0_real_fraction": FRAME / (left_ctx + FRAME), "conv_layers": conv_layers,
                "sliding_window_frames": int(mimi.config.sliding_window),
                "num_quantizers_total": int(mimi.config.num_quantizers),
                "num_semantic_quantizers": int(mimi.config.num_semantic_quantizers)}

    @torch.inference_mode()
    def run(x: np.ndarray) -> dict:
        xt = torch.from_numpy(x).float().cuda()[None, None]  # [1, 1, T]
        codes = mimi.encode(xt, num_quantizers=NQ).audio_codes  # [1, 8, F]
        full = mimi.decode(codes).audio_values[0, 0].float()
        one = mimi.decode(codes[:, :, :1]).audio_values[0, 0].float()
        nf = int(codes.shape[-1])
        fr = lambda w, i: dbfs(float(w[i * FRAME:(i + 1) * FRAME].pow(2).mean().sqrt())) if (i + 1) * FRAME <= w.shape[-1] else None
        return {
            "n_samples": int(x.shape[0]),
            "n_frames": nf,
            "in_80ms": seg_stats(x, FRAME),
            "in_240ms": seg_stats(x, 3 * FRAME),
            "in_frames_rms_dbfs": [seg_stats(x[i * FRAME:], FRAME)["rms_dbfs"] for i in range(min(4, nf))],
            "codes_first4": codes[0, :, :4].T.cpu().tolist(),  # [frame][codebook]
            "cb0_first16": codes[0, 0, :16].cpu().tolist(),
            "cb0_last4": codes[0, 0, -4:].cpu().tolist(),
            "dec1_n_samples": int(one.shape[-1]),
            "dec1_rms_dbfs": dbfs(float(one.pow(2).mean().sqrt())),
            "dec1_peak_dbfs": dbfs(float(one.abs().max())),
            "decfull_frames_rms_dbfs": [fr(full, i) for i in range(min(4, nf))],
        }

    # warm-up (cudnn autotune etc.)
    run(np.zeros(SR, np.float32))

    results = []
    t1 = perf_counter()
    for c in clips:
        x = (c["pcm16"].astype(np.float32) / 32768.0).clip(-1, 1)
        for vname, v in make_variants(x).items():
            r = run(v)
            r.update({"clip": c["name"], "group": c["group"], "variant": vname})
            results.append(r)
            print(f'{c["name"]:34s} {vname:9s} in80 {r["in_80ms"]["rms_dbfs"]:7.1f} dBFS  f0 cb0={r["codes_first4"][0][0]:5d} '
                  f'f1={r["codes_first4"][1][0] if len(r["codes_first4"]) > 1 else None}  dec1 {r["dec1_rms_dbfs"]:6.1f} dBFS', flush=True)
    torch.cuda.synchronize()
    env = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "transformers": str(transformers.__version__),
           "python": platform.python_version(), "mimi_dtype": str(next(mimi.parameters()).dtype), "repo": REPO_ID,
           "revision": REVISION, "num_quantizers": NQ, "load_s": load_s, "encode_s": perf_counter() - t1,
           "remote_total_s": perf_counter() - t0}
    return {"env": env, "encoder": enc_info, "results": results}


# ----------------------------------------------------------------------------- local: clip loading + token stats
def load_resampled(path: str):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = resample_poly(x.astype("float64"), SR // g, sr // g).astype("float32")
    return np.clip(x, -1, 1)


def synthetic_clips(dur_s: float = 4.0) -> list[tuple[str, "np.ndarray"]]:
    import numpy as np

    n = int(dur_s * SR)
    rng = np.random.default_rng(0)
    t = np.arange(n) / SR
    tone = (10 ** (-6 / 20)) * np.sin(2 * np.pi * 1000 * t)  # 1 kHz at -6 dBFS peak, no fade-in
    white = rng.standard_normal(n)
    white *= (10 ** (-20 / 20)) / np.sqrt((white ** 2).mean())  # -20 dBFS RMS
    clicks = np.zeros(n)
    for s in range(int(0.010 * SR), n, int(0.100 * SR)):  # click every 100 ms, first at 10 ms
        clicks[s:s + 8] = 0.8
    silence = np.zeros(n)
    # pink noise: 1/f spectrum shaping of white noise, mean removed, -20 dBFS RMS
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0.0
    pink = np.fft.irfft(spec, n)
    pink -= pink.mean()
    pink *= (10 ** (-20 / 20)) / np.sqrt((pink ** 2).mean())
    return [("syn_tone_1k_-6dBFS", tone), ("syn_white_-20dBFS", white), ("syn_click_train", clicks),
            ("syn_digital_silence", silence), ("syn_pink_-20dBFS", pink)]


def gather_clips() -> list[dict]:
    import numpy as np

    clips = []

    def add(name, group, x, src=None):
        clips.append({"name": name, "group": group, "src": src, "pcm16": (np.clip(x, -1, 1) * 32767).astype(np.int16)})

    # (a) rumik's own TTS output (Ira voice)
    for p in ["samples/en_short.wav", "samples/en_medium.wav", "samples/hi_short.wav", "samples/hi_medium.wav",
              "samples/hinglish_short.wav", "samples/hinglish_medium.wav", "samples/hi_en_mixed_medium.wav",
              "samples/listening_pack/en_medium_seed1001_optimized.wav", "samples/listening_pack/hi_medium_seed1001_stock.wav"]:
        add("tts_" + Path(p).stem, "tts_rumik", load_resampled(str(PROJECT / p)), p)
    # (b) a few of the 320 audit clips (deterministic picks from the sorted list)
    audit = sorted(glob.glob(str(PROJECT / "results/audit/wavs/*.wav")))
    for i in (7, 77, 150, 230, 300):
        if i < len(audit):
            add("audit_" + Path(audit[i]).stem, "tts_rumik_audit", load_resampled(audit[i]), os.path.relpath(audit[i], PROJECT))
    # (c) synthetic non-speech
    for name, x in synthetic_clips():
        add(name, "synthetic", x.astype("float32"), None)
    # (d) real human speech: LibriSpeech dev-clean utterances kept under ~/Projects/MendSpeech_All_Plans (16 kHz / 22.05 kHz)
    mend = Path.home() / "Projects/MendSpeech_All_Plans/data"
    for p in ["benchmark/1272-128104-0000.wav", "benchmark/1272-128104-0001.wav", "benchmark/1462-170138-0001.wav",
              "benchmark/1462-170138-0002.wav", "benchmark/1919-142785-0000.wav", "benchmark/1919-142785-0003.wav",
              "benchmark/1988-147956-0002.wav", "benchmark/1988-147956-0004.wav", "benchmark/1673-143396-0002.wav",
              "raw/libri1.wav"]:
        fp = mend / p
        if fp.exists():
            add("libri_" + fp.stem, "human_librispeech", load_resampled(str(fp)), "~/Projects/MendSpeech_All_Plans/data/" + p)
    return clips


def frames_of(tokens: list[int]) -> list[list[int]]:
    """Round-robin frame assembly, same rule as RumikOSSForCausalLM.audio_tokens_to_codes."""
    frames, cur = [], []
    for t in tokens:
        if t < FIRST_UNIT_ID or t > FIRST_UNIT_ID + 2048 * Q_TTS - 1:
            cur = []
            continue
        code, q = (t - FIRST_UNIT_ID) // Q_TTS, (t - FIRST_UNIT_ID) % Q_TTS
        if q == len(cur):
            cur.append(code)
        else:
            cur = [code] if q == 0 else []
        if len(cur) == Q_TTS:
            frames.append(cur)
            cur = []
    return frames


def token_position_stats(gen_json: Path, watch=(1049, 127, 958)) -> dict | None:
    if not gen_json.exists():
        return None
    d = json.loads(gen_json.read_text())
    seqs = [(g, k, v["tokens"]) for g in ("stock", "opt") for k, v in d.get(g, {}).items() if "tokens" in v]
    pos = {c: collections.Counter() for c in watch}
    per_frame_cb0 = [collections.Counter() for _ in range(3)]
    per_frame_all = [[collections.Counter() for _ in range(Q_TTS)] for _ in range(3)]
    n_frames = 0
    for _, _, t in seqs:
        fr = frames_of(t)
        n_frames += len(fr)
        for i, f in enumerate(fr):
            if f[0] in pos:
                pos[f[0]]["frame0" if i == 0 else "frame1" if i == 1 else "frame2" if i == 2 else "frame3+"] += 1
            if i < 3:
                per_frame_cb0[i][f[0]] += 1
                for q in range(Q_TTS):
                    per_frame_all[i][q][f[q]] += 1
    return {
        "source": str(gen_json.relative_to(PROJECT)),
        "n_sequences": len(seqs),
        "n_frames_total": n_frames,
        "positions_of_codebook0_codes": {str(c): dict(pos[c]) for c in watch},
        "frame_cb0_top": [dict(cnt.most_common(6)) for cnt in per_frame_cb0],
        "frame_per_codebook_top3": [[{"codebook": q, "distinct": len(c), "top": c.most_common(3)} for q, c in enumerate(fr)]
                                    for fr in per_frame_all],
    }


# ----------------------------------------------------------------------------- summary tables
def summarize(results: list[dict]) -> dict:
    by_var = collections.OrderedDict()
    order = ["asis", "trim", "mid_0.9", "mid_1.3", "mid_2.1", "pad1s", "pad12f", "rev_mid", "rev_asis"]
    for v in order:
        rs = [r for r in results if r["variant"] == v]
        if not rs:
            continue
        f0 = collections.Counter(r["codes_first4"][0][0] for r in rs)
        f1 = collections.Counter(r["codes_first4"][1][0] for r in rs if len(r["codes_first4"]) > 1)
        f2 = collections.Counter(r["codes_first4"][2][0] for r in rs if len(r["codes_first4"]) > 2)
        loud = [r for r in rs if r["in_80ms"]["rms_dbfs"] > LOUD80_DBFS]
        quiet = [r for r in rs if r["in_80ms"]["rms_dbfs"] <= LOUD80_DBFS]
        by_var[v] = {
            "n": len(rs),
            "frame0_cb0_is_1049": sum(1 for r in rs if r["codes_first4"][0][0] == 1049),
            "frame0_cb0_counts": dict(f0.most_common()),
            "frame1_cb0_counts": dict(f1.most_common(6)),
            "frame2_cb0_counts": dict(f2.most_common(6)),
            "n_loud80": len(loud),
            "loud80_frame0_is_1049": sum(1 for r in loud if r["codes_first4"][0][0] == 1049),
            "loud80_frame0_codes": dict(collections.Counter(r["codes_first4"][0][0] for r in loud).most_common()),
            "n_quiet80": len(quiet),
            "quiet80_frame0_is_1049": sum(1 for r in quiet if r["codes_first4"][0][0] == 1049),
            "quiet80_frame0_codes": dict(collections.Counter(r["codes_first4"][0][0] for r in quiet).most_common()),
            "in80_rms_dbfs_median": sorted(r["in_80ms"]["rms_dbfs"] for r in rs)[len(rs) // 2],
            "dec1_rms_dbfs_median": sorted(r["dec1_rms_dbfs"] for r in rs)[len(rs) // 2],
            "dec1_rms_dbfs_min_max": [min(r["dec1_rms_dbfs"] for r in rs), max(r["dec1_rms_dbfs"] for r in rs)],
            "decfull_frame0_rms_dbfs_median": sorted(r["decfull_frames_rms_dbfs"][0] for r in rs)[len(rs) // 2],
        }
    all_loud = [r for r in results if r["in_80ms"]["rms_dbfs"] > LOUD80_DBFS]
    all_quiet = [r for r in results if r["in_80ms"]["rms_dbfs"] <= LOUD80_DBFS]
    return {
        "by_variant": by_var,
        "overall": {
            "n": len(results),
            "frame0_cb0_is_1049": sum(1 for r in results if r["codes_first4"][0][0] == 1049),
            "loud80": {"n": len(all_loud), "frame0_is_1049": sum(1 for r in all_loud if r["codes_first4"][0][0] == 1049),
                       "frame0_codes": dict(collections.Counter(r["codes_first4"][0][0] for r in all_loud).most_common(12))},
            "quiet80": {"n": len(all_quiet), "frame0_is_1049": sum(1 for r in all_quiet if r["codes_first4"][0][0] == 1049),
                        "frame0_codes": dict(collections.Counter(r["codes_first4"][0][0] for r in all_quiet).most_common(12))},
            "any_1049_after_frame0": sum(1 for r in results for c in r["cb0_first16"][1:] if c == 1049),
        },
    }


def md_tables(data: dict) -> str:
    s = data["summary"]
    L = ["| variant | n | frame0 cb0 = 1049 | other frame-0 codes | loud first 80 ms (n, =1049) | quiet first 80 ms (n, =1049) | frame-1 cb0 top | frame-2 cb0 top | median in 80 ms dBFS | median dec(frame0 alone) dBFS |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in s["by_variant"].items():
        other = {k: c for k, c in r["frame0_cb0_counts"].items() if k != 1049}
        L.append(f'| {v} | {r["n"]} | {r["frame0_cb0_is_1049"]} | {other} | {r["n_loud80"]}, {r["loud80_frame0_is_1049"]} | '
                 f'{r["n_quiet80"]}, {r["quiet80_frame0_is_1049"]} | {dict(list(r["frame1_cb0_counts"].items())[:3])} | '
                 f'{dict(list(r["frame2_cb0_counts"].items())[:3])} | {r["in80_rms_dbfs_median"]:.1f} | {r["dec1_rms_dbfs_median"]:.1f} |')
    o = s["overall"]
    L += ["", f'Overall: {o["frame0_cb0_is_1049"]}/{o["n"]} variants have frame-0 cb0 = 1049; loud-onset (RMS of first 80 ms > {LOUD80_DBFS:.0f} dBFS): '
          f'{o["loud80"]["frame0_is_1049"]}/{o["loud80"]["n"]}; quiet-onset: {o["quiet80"]["frame0_is_1049"]}/{o["quiet80"]["n"]}; '
          f'1049 seen in frames 1-15 of any variant: {o["any_1049_after_frame0"]} times.', ""]
    L += ["| clip | variant | in 80 ms RMS/peak dBFS | in 240 ms RMS | frame 0 (8 codebooks) | f1 cb0 | f2 cb0 | f3 cb0 | dec frame0 alone dBFS | full-decode frame0 dBFS |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in data["results"]:
        c = r["codes_first4"]
        g = lambda i: c[i][0] if len(c) > i else "-"
        L.append(f'| {r["clip"]} | {r["variant"]} | {r["in_80ms"]["rms_dbfs"]:.1f} / {r["in_80ms"]["peak_dbfs"]:.1f} | {r["in_240ms"]["rms_dbfs"]:.1f} | '
                 f'{c[0]} | {g(1)} | {g(2)} | {g(3)} | {r["dec1_rms_dbfs"]:.1f} | {r["decfull_frames_rms_dbfs"][0]:.1f} |')
    t = data.get("token_position_stats")
    if t:
        L += ["", f'Generated-token positions ({t["n_sequences"]} sequences, {t["n_frames_total"]} frames, codebook 0):', ""]
        L += ["| code | frame 0 | frame 1 | frame 2 | frame 3+ |", "|---|---|---|---|---|"]
        for code, p in t["positions_of_codebook0_codes"].items():
            L.append(f'| {code} | {p.get("frame0", 0)} | {p.get("frame1", 0)} | {p.get("frame2", 0)} | {p.get("frame3+", 0)} |')
    return "\n".join(L)


@app.local_entrypoint()
def main(out: str = "results/encoder_first_code.json", md: str = "results/encoder_first_code.md"):
    t0 = time.time()
    clips = gather_clips()
    print(f"{len(clips)} clips: " + ", ".join(f'{c["name"]}({len(c["pcm16"]) / SR:.1f}s)' for c in clips))
    remote = encode_clips.remote(clips)
    wall = time.time() - t0
    data = {
        "env": remote["env"], "encoder": remote["encoder"],
        "config": {"gpu": GPU, "sr": SR, "frame": FRAME, "num_quantizers": NQ, "trim_dbfs": TRIM_DBFS, "mid_offsets_s": MID_OFFSETS_S,
                   "loud80_dbfs": LOUD80_DBFS, "wall_clock_s": wall,
                   "est_cost_usd": round(remote["env"]["remote_total_s"] / 3600 * (0.59 + 0.15), 3)},  # T4 $0.59/h + ~cpu/mem
        "clips": [{"name": c["name"], "group": c["group"], "src": c["src"], "seconds": len(c["pcm16"]) / SR} for c in clips],
        "results": remote["results"],
    }
    data["summary"] = summarize(data["results"])
    data["token_position_stats"] = token_position_stats(PROJECT / "results/audit/gen.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tables = md_tables(data)
    Path(md).write_text("# Mimi encoder: is frame-0 code 1049 content-independent?\n\n(auto-generated tables; see verdict below)\n\n" + tables + "\n")
    print(tables)
    e = data["encoder"]
    print(f'\nencoder hop {e["hop_samples"]} samples; frame-0 left context {e["left_context_samples"]} samples '
          f'({e["left_context_ms"]:.0f} ms) => real audio is {e["frame0_real_fraction"] * 100:.0f}% of frame-0 receptive field')
    print(f'wrote {out}, {md}; remote {remote["env"]["remote_total_s"]:.0f}s on {remote["env"]["gpu"]}, '
          f'est cost ${data["config"]["est_cost_usd"]:.3f}, wall {wall:.0f}s')
