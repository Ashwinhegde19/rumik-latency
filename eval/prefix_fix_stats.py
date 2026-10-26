#!/usr/bin/env python3
"""Statistics for the prefix experiment: results/prefix_fix/{gen,judge}.json -> report.md + summary.json.

Same normalisation, judges and bootstrap as eval/audit_stats.py (imported). Speaker similarity is the cosine to
the centroid of the k0 clips of the same text (leave-one-out for the k0 clips themselves). Differences versus k0
are paired by (text, seed) and summarised with the audit's stratified bootstrap.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit_stats import TEXTS, asr_scores, signflip_p, strat_boot_mean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABEL = {1049: "1049 (frame-0 transient)", 127: "127 (frame-1 transient)", 1880: "1880 (frame-2 transient)"}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main(d: str = "results/prefix_fix") -> None:
    D = ROOT / d
    gen = json.loads((D / "gen.json").read_text())
    jpath = D / "judge.json"
    jc = json.loads(jpath.read_text())["clips"] if jpath.exists() else {}
    rows = gen["results"]
    conds = [c for c in ("k0", "k1", "k2", "k3", "k4", "k3B") if any(r["cond"] == c for r in rows)]
    for r in rows:
        j = jc.get(f"gen/{r['key']}", {})
        r["asr"] = asr_scores(r["text_id"], j) if j else {}
        r["utmos22"], r["utmosv2"] = j.get("utmos22"), j.get("utmosv2")
        r["_emb"] = {k: np.array(j[k], float) for k in ("ecapa", "wavlm_sv") if k in j}
    for emb in ("ecapa", "wavlm_sv"):
        for tid in TEXTS:
            k0 = [r for r in rows if r["text_id"] == tid and r["cond"] == "k0" and emb in r["_emb"]]
            if len(k0) < 3:
                continue
            S = np.stack([r["_emb"][emb] for r in k0])
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            cen = S.mean(0)
            for r in rows:
                if r["text_id"] != tid or emb not in r["_emb"]:
                    continue
                c = np.delete(S, k0.index(r), axis=0).mean(0) if r["cond"] == "k0" else cen
                r[f"spk_{emb}"] = cos(r["_emb"][emb], c)

    def lang(r):
        return TEXTS[r["text_id"]]["lang"]

    def cer(which):
        def f(r):
            name = {"whisper": ("whisper_en", "whisper_hi"), "mms": ("mms_eng", "mms_hin")}[which][0 if lang(r) == "en" else 1]
            return r["asr"].get(name, {}).get("cer")
        return f

    metrics = [
        ("first_nonsilent_frame", lambda r: r["first_nonsilent_frame"], 2),
        ("ttfa_nonsilent_ms", lambda r: None if r["ttfa_nonsilent_s"] is None else r["ttfa_nonsilent_s"] * 1e3, 0),
        ("ttfa_first_frame_ms", lambda r: None if r["ttfa_s"] is None else r["ttfa_s"] * 1e3, 0),
        ("frames", lambda r: r["frames"], 1),
        ("cer_whisper", cer("whisper"), 3),
        ("cer_mms", cer("mms"), 3),
        ("utmos22", lambda r: r["utmos22"], 3),
        ("utmosv2", lambda r: r["utmosv2"], 3),
        ("spk_ecapa", lambda r: r.get("spk_ecapa"), 4),
        ("spk_wavlm", lambda r: r.get("spk_wavlm_sv"), 4),
    ]

    def vals(c, f, group="ALL"):
        return [f(r) for r in rows if r["cond"] == c and (group == "ALL" or lang(r) == group)
                and f(r) is not None and np.isfinite(f(r))]

    def paired_diff(c, f, group="ALL"):
        by, d = {}, []
        k0 = {(r["text_id"], r["seed"]): r for r in rows if r["cond"] == "k0"}
        for r in rows:
            if r["cond"] != c or (group != "ALL" and lang(r) != group):
                continue
            b = k0.get((r["text_id"], r["seed"]))
            if b is None:
                continue
            x, y = f(b), f(r)
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                continue
            by.setdefault(r["text_id"], []).append(y - x)
            d.append(y - x)
        if not d:
            return None
        lo, hi = strat_boot_mean({k: np.array(v) for k, v in by.items()})
        return {"n": len(d), "diff": float(np.mean(d)), "ci95": [lo, hi], "p": signflip_p(np.array(d)),
                "distinguishable": bool((lo > 0 or hi < 0) and signflip_p(np.array(d)) < 0.05 and len(d) >= 5)}

    L = ["# Prefix experiment: results", "",
         f"GPU {gen['env']['gpu']}; torch {gen['env']['torch']}; model revision {gen['env']['revision'][:8]}; "
         f"silence threshold {gen['env']['silence_dbfs']} dBFS per 80 ms frame. {len(rows)} clips. "
         f"Determinism (k0 loop vs StreamingTTS.run): {gen.get('determinism')}. "
         f"k0 reproduces the audit's optimized tokens: {gen.get('k0_reproduces_audit_tokens')}. Errors: {len(gen.get('errors', []))}.", ""]
    pre = gen["cfg"]["prefixes"]
    L += ["Prefixes (real generated starts; codebook-0 code per frame):"]
    for name in ("A", "B", "A4"):
        if name in pre:
            ids = pre[name]["ids"]
            L.append(f"- {name} from `{pre[name]['source']}`: {[(t - 261008) // 8 for t in ids[::8]]}")
    L += ["", "## Where does the model start speaking?", "",
          "| cond | prefix frames | n | first non-silent generated frame: 0 / 1 / 2 / 3 / 4+ / none | median | first generated code (codebook 0) | ttfa to first audible, p50 ms | stop_predictor |",
          "|---|---:|---:|---|---:|---|---:|---:|"]
    summary = {"conditions": {}, "diff_vs_k0": {}, "by_lang": {}}
    for c in conds:
        rs = [r for r in rows if r["cond"] == c]
        fns = [r["first_nonsilent_frame"] for r in rs]
        cnt = collections.Counter("none" if x is None else ("4+" if x >= 4 else str(x)) for x in fns)
        dist = " / ".join(str(cnt.get(k, 0)) for k in ("0", "1", "2", "3", "4+", "none"))
        med = np.median([x for x in fns if x is not None]) if any(x is not None for x in fns) else None
        cb = collections.Counter(r["first4_cb0"][0] if r["first4_cb0"] else None for r in rs)
        cbs = "; ".join(f"{LABEL.get(k, k)} ×{v}" for k, v in cb.most_common(3))
        tt = vals(c, metrics[1][1])
        sp = sum(1 for r in rs if r["stopped_by"] == "stop_predictor")
        L.append(f"| {c} | {rs[0]['k']} | {len(rs)} | {dist} | {fmt(med, 1)} | {cbs} | {fmt(np.median(tt), 0) if tt else '–'} | {sp}/{len(rs)} |")
        summary["conditions"][c] = {"n": len(rs), "k": rs[0]["k"], "first_nonsilent_frame_counts": dict(cnt),
                                    "first_code_cb0": {str(k): v for k, v in cb.items()},
                                    "ttfa_nonsilent_ms_p50": float(np.median(tt)) if tt else None}
    L += ["", "## Quality of the generated audio (means; latency rows are medians)", "",
          "| metric | " + " | ".join(conds) + " |", "|---|" + "---:|" * len(conds)]
    for name, f, nd in metrics:
        cells = []
        for c in conds:
            v = vals(c, f)
            cells.append(fmt(float(np.median(v) if name.startswith(("ttfa", "first_nonsilent")) else np.mean(v)), nd) if v else "–")
            summary["conditions"][c][name] = {"mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None, "n": len(v)}
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## Paired differences versus k0 (same text and seed; stratified bootstrap 95% CI, sign-flip p)", "",
          "| metric | " + " | ".join(f"{c} − k0" for c in conds if c != "k0") + " |", "|---|" + "---:|" * (len(conds) - 1)]
    for name, f, nd in metrics:
        cells = []
        for c in conds:
            if c == "k0":
                continue
            p = paired_diff(c, f)
            summary["diff_vs_k0"].setdefault(c, {})[name] = p
            cells.append("–" if p is None else f"{p['diff']:+.{nd}f} [{p['ci95'][0]:+.{nd}f}, {p['ci95'][1]:+.{nd}f}] p={p['p']:.3f}"
                         + (" **distinguishable**" if p["distinguishable"] else ""))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## CER by language group (Whisper; mean, k0 vs k3)", "", "| group | n | k0 | k3 | k3 − k0 CI95 |", "|---|---:|---:|---:|---|"]
    for g in ("en", "hinglish", "hi", "hi_en_mixed"):
        v0, v3 = vals("k0", cer("whisper"), g), vals("k3", cer("whisper"), g)
        p = paired_diff("k3", cer("whisper"), g)
        summary["by_lang"][g] = {"k0": float(np.mean(v0)) if v0 else None, "k3": float(np.mean(v3)) if v3 else None, "diff": p}
        L.append(f"| {g} | {len(v3)} | {fmt(np.mean(v0)) if v0 else '–'} | {fmt(np.mean(v3)) if v3 else '–'} | "
                 f"{'–' if p is None else f'[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]'} |")
    L += ["", "## Read", "",
          "Numbers only; the interpretation key was written before the run (notes.md, 24 Oct). "
          "k0 is the optimized path as audited; the prefix frames are never part of the output audio; "
          "ttfa is measured from request start and includes prefix decoding and the longer prefill.", ""]
    (D / "report.md").write_text("\n".join(L) + "\n")
    (D / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
