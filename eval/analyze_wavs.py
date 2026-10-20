"""Part B: objective analysis of the existing sample WAVs (baseline samples/*.wav vs samples/optimized/*.wav).

    .venv/bin/python eval/analyze_wavs.py            # writes results/audit/partB_wav_analysis.json

Both sets are seed 1000 of each text (baseline: stock generate_audio + HF Mimi, from bench_modal.py;
optimized: streamed output of rumik_fast.StreamingTTS, from bench_fast.py). They are single random
draws whose token sequences diverge after a few dozen tokens, so per-file differences are anecdotal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audio_metrics import (FRAME, SR, boundary_z, clip_metrics, hf_energy_fold, load_wav,  # noqa: E402
                           phase_profile, phase_profile_norm)


def profiles(x):
    """(name, (profile_sum, count), boundary phases) for every boundary detector."""
    out = []
    for per, tag in ((FRAME, "f80"), (FRAME // 2, "f40")):
        out.append((f"{tag}_d1", phase_profile(x, per, 1), (0,)))
        out.append((f"{tag}_d2", phase_profile(x, per, 2), (per - 1, 0)))
        out.append((f"{tag}_d1norm", phase_profile_norm(x, per), (0,)))
        out.append((f"{tag}_hf8k", hf_energy_fold(x, SR, per), (0, -1)))
    return out

ROOT = Path(__file__).resolve().parent.parent
TEXTS = json.loads((ROOT / "texts.json").read_text())


def main():
    rows = []
    pooled = {k: {} for k in ("baseline", "optimized")}
    for t in TEXTS:
        tid = t["id"]
        for kind, path in (("baseline", ROOT / "samples" / f"{tid}.wav"),
                           ("optimized", ROOT / "samples" / "optimized" / f"{tid}.wav")):
            x, sr, info = load_wav(str(path))
            m = clip_metrics(x, sr)
            m.update({"text_id": tid, "lang": t["lang"], "kind": kind, "file": str(path.relative_to(ROOT)),
                      "sample_rate": sr, "channels": info.channels, "subtype": info.subtype})
            rows.append(m)
            for key, (p, c), phases in profiles(x):
                if p is None:
                    continue
                acc = pooled[kind].setdefault(key, [np.zeros(len(p)), 0, phases])
                acc[0] += p
                acc[1] += c

    pooled_z = {}
    for kind, d in pooled.items():
        for key, (p, c, phases) in d.items():
            pooled_z[f"{kind}/{key}"] = boundary_z(p, c, phases) | {"frames": c}

    # detector calibration: add a tiny step at every 80 ms boundary of the baseline clips and re-score
    calib = {}
    base = [load_wav(str(ROOT / "samples" / f'{t["id"]}.wav'))[0] for t in TEXTS]
    for kind in ("step", "click"):
        for amp in (1e-4, 3e-4, 1e-3, 3e-3):
            acc = {}
            for x in base:
                n = np.arange(len(x))
                if kind == "step":   # DC step of size amp at every boundary
                    y = x + amp * np.where((n // FRAME) % 2 == 0, 0.5, -0.5)
                else:                # one-sample impulse of size amp at every boundary
                    y = x.copy()
                    y[::FRAME] += amp
                for key, (p, c), phases in profiles(y):
                    if p is None or not key.startswith("f80"):
                        continue
                    a = acc.setdefault(key, [np.zeros(len(p)), 0, phases])
                    a[0] += p
                    a[1] += c
            calib[f"{kind}_{amp:g}"] = {k: boundary_z(p, c, ph)["z"] for k, (p, c, ph) in acc.items()}

    # paired per-text differences (optimized - baseline)
    diffs = []
    keys = ["duration_s", "peak_dbfs", "n_clip", "rms_dbfs", "active_rms_dbfs", "first_nonsilent_frame",
            "leading_silence_ms", "trailing_silence_ms", "centroid_hz", "flatness", "hf_ratio_6k"]
    by = {(r["text_id"], r["kind"]): r for r in rows}
    for t in TEXTS:
        b, o = by[(t["id"], "baseline")], by[(t["id"], "optimized")]
        diffs.append({"text_id": t["id"], **{k: (o[k] - b[k]) if (o[k] is not None and b[k] is not None) else None for k in keys}})
    mean_diff = {k: float(np.mean([d[k] for d in diffs if d[k] is not None])) for k in keys}

    out = {"note": "seed 1000 single draws per text; differences are anecdotal (different token sequences)",
           "rows": rows, "paired_diff_opt_minus_base": diffs, "mean_paired_diff": mean_diff,
           "pooled_boundary": pooled_z, "boundary_detector_calibration": calib}
    dst = ROOT / "results" / "audit" / "partB_wav_analysis.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    hdr = f'{"text":20s} {"kind":9s} {"dur s":>6s} {"peak dB":>7s} {"clip":>4s} {"RMS dB":>7s} {"actRMS":>7s} {"lead ms":>7s} {"trail ms":>8s} {"f0":>3s} {"centroid":>8s} {"flat":>6s} {"hf6k":>6s} {"bnd z d1":>8s} {"bnd z d2":>8s}'
    print(hdr)
    for r in rows:
        z1 = r["bnd_f80_d1"]["z"] if r["bnd_f80_d1"] else float("nan")
        z2 = r["bnd_f80_d2"]["z"] if r["bnd_f80_d2"] else float("nan")
        print(f'{r["text_id"]:20s} {r["kind"]:9s} {r["duration_s"]:6.2f} {r["peak_dbfs"]:7.2f} {r["n_clip"]:4d} {r["rms_dbfs"]:7.2f} '
              f'{r["active_rms_dbfs"]:7.2f} {r["leading_silence_ms"]:7.0f} {r["trailing_silence_ms"]:8.0f} {r["first_nonsilent_frame"]:3d} '
              f'{r["centroid_hz"]:8.0f} {r["flatness"]:6.3f} {r["hf_ratio_6k"]:6.3f} {z1:8.2f} {z2:8.2f}')
    print("\nmean paired diff (optimized - baseline):", json.dumps({k: round(v, 4) for k, v in mean_diff.items()}))
    print("\npooled boundary statistics (z of the boundary phase vs all other phases; ratio = boundary/other mean |diff|):")
    for k, v in pooled_z.items():
        print(f"  {k:26s} ratio {v['ratio']:.3f}  z {v['z']:6.2f}  pct_rank {v['pct_rank']:.3f}  frames {v['frames']}")
    print("\ndetector calibration (baseline clips + artificial step at each 80 ms boundary):")
    for k, v in calib.items():
        print(f"  {k:12s} " + "  ".join(f"{kk} z {vv:6.2f}" for kk, vv in v.items()))
    print(f"\nwrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
