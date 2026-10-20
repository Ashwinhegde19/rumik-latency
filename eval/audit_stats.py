"""Paired statistics for the quality audit (local, CPU only).

    .venv/bin/python eval/audit_stats.py      # reads results/audit/{gen,judge}.json + results/audit/wavs/
                                              # writes results/audit/summary.json and results/audit/report.txt,
                                              # copies the listening pack to samples/listening_pack/

Every comparison is paired: the same text and seed through the stock path and the optimized path. A pair's
two token sequences share a prefix (same RNG stream) and then diverge, so each pair is two random draws
with a common start. For every metric:
  * diff = optimized - stock per pair, mean over pairs;
  * 95% CI: stratified bootstrap (seeds resampled within each text), 10,000 replicates;
  * p: two-sided sign-flip permutation test on the mean paired difference, 10,000 permutations;
  * null check: the same procedure applied stock-vs-stock on two disjoint seed halves (seed s vs s+8).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audio_metrics import clip_metrics, load_wav  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "results" / "audit"
TEXTS = {t["id"]: t for t in json.loads((ROOT / "texts.json").read_text())}
GROUPS = ["ALL", "en", "hinglish", "hi", "hi_en_mixed"]
RNG = np.random.default_rng(20261019)
N_BOOT = 10_000

# all-Devanagari references for Hindi-speech texts (used with Devanagari ASR output)
DEVA_REF = {
    "hinglish_short": "हाँ भाई, मैं बस पाँच मिनट में आ रहा हूँ।",
    "hinglish_medium": "कल की मीटिंग में सबको हमारा आइडिया बहुत पसंद आया, अब बस फ़ाइनल डेक रेडी करना है।",
    "hinglish_long": "यार, आज ऑफ़िस में इतना काम था कि लंच भी टाइम पे नहीं हो पाया। शाम को ट्रैफ़िक भी बहुत ज़्यादा था, "
                     "इसलिए घर पहुँचने में दो घंटे लग गए। अब बस चाय पीके आराम करना है।",
    "hi_en_mixed_medium": "आज की मीटिंग में सबको हमारा आइडिया पसंद आया, अब कॉफ़ी मेरी तरफ़ से!",
}
for _tid in ("hi_short", "hi_medium", "hi_long"):
    DEVA_REF[_tid] = TEXTS[_tid]["text"]
# Latin words an ASR may leave in a Hindi transcript -> the Devanagari spelling used in DEVA_REF
LATIN2DEVA = {"meeting": "मीटिंग", "idea": "आइडिया", "coffee": "कॉफ़ी", "final": "फ़ाइनल", "deck": "डेक",
              "ready": "रेडी", "office": "ऑफ़िस", "lunch": "लंच", "time": "टाइम", "traffic": "ट्रैफ़िक",
              "minute": "मिनट", "minutes": "मिनट", "ok": "ओके", "okay": "ओके"}
HI_NUM = ["शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह",
          "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस"]
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num2words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num2words(n % 100))
    return str(n)


def norm_latin(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", s)
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return " ".join(s.split())


def norm_deva(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.translate({0x0966 + i: ord("0") + i for i in range(10)})        # Devanagari digits -> ASCII
    s = re.sub(r"\d+", lambda m: " " + (HI_NUM[int(m.group())] if int(m.group()) <= 20 else m.group()) + " ", s)
    s = re.sub(r"[a-z]+", lambda m: LATIN2DEVA.get(m.group(), m.group()), s)
    s = unicodedata.normalize("NFD", s).replace("़", "")          # nukta
    s = s.replace("ँ", "ं")                                     # chandrabindu -> anusvara
    s = s.replace("ऑ", "ओ").replace("ॉ", "ो")         # candra O -> O
    s = s.replace("ॅ", "े").replace("‌", "").replace("‍", "")
    s = unicodedata.normalize("NFC", s)
    s = "".join(" " if (unicodedata.category(c)[0] in "PS" or c in "।॥") else c for c in s)
    return " ".join(s.split())


def edit_distance(a, b) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def cer(ref: str, hyp: str) -> float:
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    return edit_distance(r, h) / max(1, len(r))


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(1, len(r))


def asr_scores(text_id: str, j: dict) -> dict:
    """CER/WER per (recognizer, language) for one clip; normalisation is identical for both paths."""
    t = TEXTS[text_id]
    out = {}
    w, m = j.get("whisper", {}), j.get("mms", {})
    if t["lang"] == "en":
        ref = norm_latin(t["text"])
        for name, hyp in (("whisper_en", w.get("en")), ("mms_eng", m.get("eng"))):
            if hyp is not None:
                h = norm_latin(hyp)
                out[name] = {"cer": cer(ref, h), "wer": wer(ref, h)}
    else:
        ref = norm_deva(DEVA_REF[text_id])
        for name, hyp in (("whisper_hi", w.get("hi")), ("mms_hin", m.get("hin"))):
            if hyp is not None:
                h = norm_deva(hyp)
                out[name] = {"cer": cer(ref, h), "wer": wer(ref, h)}
        if t["lang"] == "hinglish" and w.get("en") is not None:  # romanized check, as the earlier README did
            r2, h2 = norm_latin(t["text"]), norm_latin(w["en"])
            out["whisper_en_roman"] = {"cer": cer(r2, h2), "wer": wer(r2, h2)}
    return out


# ----------------------------------------------------------------------------- statistics
def strat_boot_mean(d_by_text: dict[str, np.ndarray]) -> tuple[float, float]:
    reps = np.zeros(N_BOOT)
    tot = sum(len(v) for v in d_by_text.values())
    for v in d_by_text.values():
        v = np.asarray(v, float)
        idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
        reps += v[idx].sum(1)
    reps /= tot
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def signflip_p(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    s = RNG.choice([-1.0, 1.0], size=(N_BOOT, len(d)))
    return float(((np.abs((s * d).mean(1)) >= obs - 1e-15).sum() + 1) / (N_BOOT + 1))


def paired(rows: list[dict], key, group: str, a="stock", b="optimized", seeds=None) -> dict | None:
    """rows: per (text, seed) dicts with rows[i][a][key], rows[i][b][key]."""
    by = {}
    va, vb = [], []
    for r in rows:
        if group != "ALL" and TEXTS[r["text_id"]]["lang"] != group:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        x, y = key(r[a]), key(r[b])
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            continue
        by.setdefault(r["text_id"], []).append(y - x)
        va.append(x)
        vb.append(y)
    if not va:
        return None
    d = np.array(vb) - np.array(va)
    lo, hi = strat_boot_mean({k: np.array(v) for k, v in by.items()})
    p = signflip_p(d)
    return {"n": len(d), a: float(np.mean(va)), b: float(np.mean(vb)), "diff": float(d.mean()),
            "ci95": [lo, hi], "p": p, "distinguishable": bool((lo > 0 or hi < 0) and p < 0.05 and len(d) >= 5)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=str(AUD / "gen.json"))
    ap.add_argument("--judge", default=str(AUD / "judge.json"))
    ap.add_argument("--wavs", default=str(AUD / "wavs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-pack", action="store_true")
    args = ap.parse_args()
    wav_dir = Path(args.wavs)
    gen = json.loads(Path(args.gen).read_text())
    judge = json.loads(Path(args.judge).read_text()) if Path(args.judge).exists() else {"clips": {}}
    jc = judge["clips"]
    stock, opt = gen["stock"], gen["opt"]
    seeds = sorted({s["seed"] for s in stock.values()})
    half = len(seeds) // 2
    A, B = seeds[:half], seeds[half:2 * half]

    # ---------------------------------------------------------------- per-clip features
    def features(key: str, kind: str) -> dict:
        x, sr, _ = load_wav(str(wav_dir / f"{key}_{kind}.wav"))
        f = clip_metrics(x, sr)
        g = (stock if kind == "stock" else opt)[key]
        f["frames_tok"] = g["frames"]
        f["stopped_by"] = g["stopped_by"]
        j = jc.get(f"gen/{key}_{kind}", {})
        f["asr"] = asr_scores(stock[key]["text_id"], j) if j else {}
        for m in ("utmos22", "utmosv2"):
            f[m] = j.get(m)
        f["_emb"] = {k: np.array(j[k], float) for k in ("ecapa", "wavlm_sv") if k in j}
        return f

    rows = []
    for key in sorted(stock):
        if key not in opt:
            continue
        s = stock[key]
        rows.append({"key": key, "text_id": s["text_id"], "seed": s["seed"],
                     "stock": features(key, "stock"), "optimized": features(key, "optimized"),
                     "common_prefix_tokens": next((i for i, (u, v) in enumerate(zip(s["tokens"], opt[key]["tokens"]))
                                                   if u != v), min(len(s["tokens"]), len(opt[key]["tokens"])))})

    # speaker similarity: cosine to the leave-one-out centroid of the other seeds' stock clips of the same text
    for emb in ("ecapa", "wavlm_sv"):
        for tid in TEXTS:
            rs = [r for r in rows if r["text_id"] == tid and emb in r["stock"]["_emb"]]
            if len(rs) < 3:
                continue
            S = np.stack([r["stock"]["_emb"][emb] for r in rs])
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            for i, r in enumerate(rs):
                c = np.delete(S, i, axis=0).mean(0)
                c /= np.linalg.norm(c)
                for kind in ("stock", "optimized"):
                    e = r[kind]["_emb"].get(emb)
                    if e is not None:
                        r[kind][f"spk_{emb}"] = float(e @ c / np.linalg.norm(e))
    for r in rows:
        for kind in ("stock", "optimized"):
            r[kind].pop("_emb", None)

    # ---------------------------------------------------------------- metric table
    def asr_key(name, m):
        return lambda f: f["asr"].get(name, {}).get(m)

    metrics = {
        "duration_s": lambda f: f["duration_s"],
        "log_duration": lambda f: float(np.log(f["duration_s"])),
        "leading_silence_ms": lambda f: f["leading_silence_ms"],
        "first_nonsilent_frame": lambda f: f["first_nonsilent_frame"],
        "trailing_silence_ms": lambda f: f["trailing_silence_ms"],
        "peak_dbfs": lambda f: f["peak_dbfs"],
        "n_clip": lambda f: f["n_clip"],
        "n_over_0p99": lambda f: f["n_over_0p99"],
        "rms_dbfs": lambda f: f["rms_dbfs"],
        "active_rms_dbfs": lambda f: f["active_rms_dbfs"],
        "centroid_hz": lambda f: f["centroid_hz"],
        "flatness": lambda f: f["flatness"],
        "hf_ratio_6k": lambda f: f["hf_ratio_6k"],
        "boundary_z_d1norm": lambda f: (f["bnd_f80_d1norm"] or {}).get("z"),
        "boundary_z_hf8k": lambda f: (f["bnd_f80_hf"] or {}).get("z"),
        "utmos22": lambda f: f["utmos22"],
        "utmosv2": lambda f: f["utmosv2"],
        "spk_ecapa": lambda f: f.get("spk_ecapa"),
        "spk_wavlm_sv": lambda f: f.get("spk_wavlm_sv"),
        "cer_whisper_en": asr_key("whisper_en", "cer"), "wer_whisper_en": asr_key("whisper_en", "wer"),
        "cer_mms_eng": asr_key("mms_eng", "cer"), "wer_mms_eng": asr_key("mms_eng", "wer"),
        "cer_whisper_hi": asr_key("whisper_hi", "cer"), "wer_whisper_hi": asr_key("whisper_hi", "wer"),
        "cer_mms_hin": asr_key("mms_hin", "cer"), "wer_mms_hin": asr_key("mms_hin", "wer"),
        "cer_whisper_en_roman": asr_key("whisper_en_roman", "cer"),
        "asr_fail_whisper": lambda f: (None if not f["asr"] else
                                       float(max(v["cer"] for k, v in f["asr"].items() if k.startswith("whisper")
                                                 and k != "whisper_en_roman") > 0.3)),
    }
    table = {}
    null = {}
    for mname, fn in metrics.items():
        table[mname] = {g: paired(rows, fn, g) for g in GROUPS}
        # stock-vs-stock null: pair seed s (first half) with seed s + half (second half), same text
        nrows = []
        for r in rows:
            if r["seed"] in A:
                r2 = next((x for x in rows if x["text_id"] == r["text_id"] and x["seed"] == r["seed"] + half), None)
                if r2:
                    nrows.append({"text_id": r["text_id"], "seed": r["seed"], "stock": r["stock"], "stock2": r2["stock"]})
        null[mname] = {g: paired(nrows, fn, g, a="stock", b="stock2") for g in GROUPS}

    # duration ratio as a geometric mean with CI (from log_duration)
    dur_ratio = {g: {k: (float(np.exp(v)) if isinstance(v, float) else [float(np.exp(x)) for x in v])
                     for k, v in table["log_duration"][g].items() if k in ("diff", "ci95")}
                 for g in GROUPS if table["log_duration"][g]}

    # ---------------------------------------------------------------- codec exactness (stream vs HF full decode)
    cc = [v["codec_check"] for v in opt.values()]
    codec = {"n": len(cc), "snr_db_min": float(min(c["snr_db"] for c in cc)),
             "snr_db_median": float(np.median([c["snr_db"] for c in cc])),
             "snr_db_after10s_min": float(min(c["snr_db_after10s"] for c in cc if c["snr_db_after10s"] is not None))
             if any(c["snr_db_after10s"] is not None for c in cc) else None,
             "snr_db_per_second_min": float(min(min(c["snr_db_per_second"]) for c in cc if c["snr_db_per_second"])),
             "max_abs_diff_max": float(max(c["max_abs_diff"] for c in cc)),
             "length_mismatches": int(sum(c["len_full"] != c["len_stream"] for c in cc)),
             "err_energy_boundary_ratio_median": float(np.median([c["err_energy_boundary_ratio"] for c in cc])),
             "n_after10s": int(sum(c["snr_db_after10s"] is not None for c in cc))}

    # ---------------------------------------------------------------- teacher forcing
    tf = gen.get("tf", {})
    tf_sum = {}
    if tf:
        comps = sorted({c for v in tf.values() for c in v["cmp"]})
        mets = ("kl_T0.8", "tvd", "flip", "top1")
        for c in comps:
            entry = {}
            for g in GROUPS:
                seqs = [v for v in tf.values() if c in v["cmp"] and (g == "ALL" or TEXTS[v["text_id"]]["lang"] == g)]
                if not seqs:
                    continue
                e = {"n_seq": len(seqs), "positions": int(sum(v["cmp"][c]["positions"] for v in seqs))}
                for m in mets:
                    per_seq = np.array([np.mean(v["cmp"][c]["per_pos"][m]) for v in seqs])
                    pooled = np.concatenate([v["cmp"][c]["per_pos"][m] for v in seqs])
                    idx = RNG.integers(0, len(per_seq), size=(N_BOOT, len(per_seq)))
                    e[m] = {"mean_pos": float(pooled.mean()), "mean_seq": float(per_seq.mean()),
                            "ci95_seq": [float(np.percentile(per_seq[idx].mean(1), 2.5)),
                                         float(np.percentile(per_seq[idx].mean(1), 97.5))]}
                e["stop_agree"] = float(np.mean([v["cmp"][c]["stop_agree"] for v in seqs if v["cmp"][c]["stop_agree"] is not None]))
                e["stop_prob_absdiff_max"] = float(max(v["cmp"][c]["stop_prob_absdiff_max"] or 0 for v in seqs))
                e["stop_flips"] = int(sum(len(v["cmp"][c]["stop_flip_positions"]) for v in seqs))
                entry[g] = e
            # by KV bucket (decode steps only)
            byb = {}
            for b in (-1, 256, 512, 1024, 2304):  # -1 = prefill (step 0)
                vals = {m: [] for m in mets}
                for v in tf.values():
                    if c not in v["cmp"]:
                        continue
                    bk = np.array(v["bucket"][: v["cmp"][c]["positions"]])
                    for m in mets:
                        vals[m].extend(np.array(v["cmp"][c]["per_pos"][m])[bk == b].tolist())
                if vals["flip"]:
                    byb["prefill" if b == -1 else str(b)] = {"positions": len(vals["flip"]),
                                                              **{m: float(np.mean(x)) for m, x in vals.items()}}
            entry["by_bucket"] = byb
            tf_sum[c] = entry

        # paired per-sequence contrasts (bootstrap over sequences)
        def seq_contrast(c1, c2, m="kl_T0.8"):
            vs = [v for v in tf.values() if c1 in v["cmp"] and c2 in v["cmp"]]
            d = np.array([np.mean(v["cmp"][c1]["per_pos"][m]) - np.mean(v["cmp"][c2]["per_pos"][m]) for v in vs])
            idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
            bm = d[idx].mean(1)
            return {"n_seq": len(d), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)),
                                                                           float(np.percentile(bm, 97.5))],
                    "p": signflip_p(d), "n_seq_c1_larger": int((d > 0).sum())}
        contrasts = {}
        for m in ("kl_T0.8", "tvd", "flip"):
            contrasts[m] = {
                "fast_vs_stock_minus_noise_floor": seq_contrast("stock_inc|fast", "stock_inc|stock_full", m),
                "fp32_dist: fast_minus_stock_inc": seq_contrast("fp32|fast", "fp32|stock_inc", m),
                "fp32_dist: fast_minus_stock_full": seq_contrast("fp32|fast", "fp32|stock_full", m),
                "fp32_dist: fast_minus_fast_nokvb": seq_contrast("fp32|fast", "fp32|fast_nokvb", m),
                "stock_dist: fast_minus_fast_nokvb": seq_contrast("stock_inc|fast", "stock_inc|fast_nokvb", m),
            }
        tf_sum["_contrasts"] = contrasts
        tf_sum["_stop_logit_at_stock_stop"] = {k: v["stop_logit_at_stock_stop"] for k, v in tf.items()}

    # ---------------------------------------------------------------- generation-level facts
    facts = {
        "pairs": len(rows), "seeds": seeds, "texts": sorted({r["text_id"] for r in rows}),
        "identical_token_sequences": int(sum(stock[r["key"]]["tokens"] == opt[r["key"]]["tokens"] for r in rows)),
        "common_prefix_tokens_median": float(np.median([r["common_prefix_tokens"] for r in rows])),
        "common_prefix_tokens_min_max": [int(min(r["common_prefix_tokens"] for r in rows)),
                                         int(max(r["common_prefix_tokens"] for r in rows))],
        "stopped_by": {k: {s: int(sum(r[k]["stopped_by"] == s for r in rows))
                           for s in ("stop_predictor", "audio_end_token", "max_new_tokens")}
                       for k in ("stock", "optimized")},
        "frames_tok_vs_wav_mismatch": int(sum(abs(r[k]["frames_tok"] - r[k]["codec_frames"]) > 1e-6
                                              for r in rows for k in ("stock", "optimized"))),
        "determinism": gen.get("determinism"),
        "errors": gen.get("errors"),
    }
    per_text = {}
    for tid in TEXTS:
        rs = [r for r in rows if r["text_id"] == tid]
        if not rs:
            continue
        per_text[tid] = {k: {"duration_s_mean": float(np.mean([r[k]["duration_s"] for r in rs])),
                             "duration_s_sd": float(np.std([r[k]["duration_s"] for r in rs], ddof=1)),
                             "utmosv2_mean": float(np.mean([r[k]["utmosv2"] for r in rs if r[k]["utmosv2"] is not None]))
                             if any(r[k]["utmosv2"] is not None for r in rs) else None}
                         for k in ("stock", "optimized")}

    summary = {"facts": facts, "table": table, "null_stock_vs_stock": null, "duration_ratio": dur_ratio,
               "codec_stream_vs_full": codec, "teacher_forced": tf_sum, "per_text": per_text,
               "rows": [{k: v for k, v in r.items()} for r in rows]}
    (AUD / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))

    # ---------------------------------------------------------------- report
    L = []
    P = L.append
    P(f"pairs: {facts['pairs']} ({len(facts['texts'])} texts x {len(seeds)} seeds); identical token sequences: "
      f"{facts['identical_token_sequences']}; common prefix median {facts['common_prefix_tokens_median']:.0f} tokens "
      f"(range {facts['common_prefix_tokens_min_max']})")
    P(f"stopped_by: {facts['stopped_by']}")
    P("")
    hdr = f"{'metric':24s} {'group':12s} {'n':>4s} {'stock':>9s} {'optim':>9s} {'diff':>9s} {'95% CI':>21s} {'p':>6s}  sig | null(stock-stock) diff [CI] sig"
    P(hdr)
    for mname in metrics:
        for g in GROUPS:
            e, nl = table[mname][g], null[mname][g]
            if not e:
                continue
            nstr = f"{nl['diff']:+.4f} [{nl['ci95'][0]:+.4f},{nl['ci95'][1]:+.4f}] {'*' if nl['distinguishable'] else ''}" if nl else ""
            P(f"{mname:24s} {g:12s} {e['n']:4d} {e['stock']:9.4f} {e['optimized']:9.4f} {e['diff']:+9.4f} "
              f"[{e['ci95'][0]:+9.4f},{e['ci95'][1]:+9.4f}] {e['p']:6.3f}  {'*' if e['distinguishable'] else ' '}   | {nstr}")
    P("")
    P(f"duration ratio optimized/stock (geometric mean, 95% CI): " +
      "; ".join(f"{g} {v['diff']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]" for g, v in dur_ratio.items()))
    P(f"codec: streaming vs HF full decode of the same tokens: {json.dumps(codec)}")
    P("")
    if tf_sum:
        P("teacher forcing (ALL): comparison -> KL(T=0.8) mean [seq CI] | TVD | matched-noise flip prob | top-1 agree | stop agree/flips")
        for c, e in tf_sum.items():
            if c.startswith("_") or "ALL" not in e:
                continue
            a = e["ALL"]
            P(f"  {c:24s} KL {a['kl_T0.8']['mean_pos']:.2e} [{a['kl_T0.8']['ci95_seq'][0]:.2e},{a['kl_T0.8']['ci95_seq'][1]:.2e}]"
              f"  TVD {a['tvd']['mean_pos']:.4f}  flip {a['flip']['mean_pos']:.4f}  top1 {a['top1']['mean_pos']:.4f}"
              f"  stop {a['stop_agree']:.4f}/{a['stop_flips']}  ({a['positions']} pos)")
        P("  by KV bucket (flip prob / KL):")
        for c in ("stock_inc|stock_full", "stock_inc|fast", "stock_inc|fast_nokvb", "fp32|stock_inc", "fp32|fast", "fp32|fast_nokvb"):
            if c in tf_sum:
                P(f"    {c:24s} " + "  ".join(f"{b}: {v['flip']:.4f}/{v['kl_T0.8']:.2e} (n={v['positions']})"
                                              for b, v in tf_sum[c]["by_bucket"].items()))
        P("  contrasts (per-sequence mean, bootstrap over sequences):")
        for m, cs in tf_sum["_contrasts"].items():
            for name, v in cs.items():
                P(f"    {m:8s} {name:40s} {v['mean_diff']:+.2e} [{v['ci95'][0]:+.2e},{v['ci95'][1]:+.2e}] p={v['p']:.3f} "
                  f"({v['n_seq_c1_larger']}/{v['n_seq']} seqs first > second)")
    fu_path = AUD / "followup.json"
    if fu_path.exists():  # eval/audit_followup.py: Whisper language-ID, codec swap, judge stability
        fu = json.loads(fu_path.read_text())
        lid = fu["lid_logodds_en_vs_hi"]
        by = {}
        for tid in TEXTS:
            if TEXTS[tid]["lang"] == "en":
                continue
            d = [lid[f"{tid}_seed{sd}_optimized"] - lid[f"{tid}_seed{sd}_stock"] for sd in seeds
                 if f"{tid}_seed{sd}_stock" in lid and f"{tid}_seed{sd}_optimized" in lid]
            if d:
                by[tid] = np.array(d)
        lo, hi = strat_boot_mean(by)
        dd = np.concatenate(list(by.values()))
        ref = norm_latin(TEXTS["hinglish_medium"]["text"])
        tr = lambda h: cer(ref, norm_latin(h)) > 0.4  # noqa: E731
        hf = fu["hf_redecode_of_optimized_tokens"]["asr_en"]
        same = sum(tr(hf[k]) == tr(jc[f"gen/{k}_optimized"]["whisper"]["en"]) for k in hf)
        flips = sum(tr(v) != tr(fu["asr_en_clean"][k]) for d in fu["asr_en_noise70"].values() for k, v in d.items())
        ntot = sum(len(d) for d in fu["asr_en_noise70"].values())
        summary["followup"] = {"lid_logodds_diff_all_hindi_speech": {"diff": float(dd.mean()), "ci95": [lo, hi],
                                                                    "p": signflip_p(dd), "n": len(dd)},
                               "codec_swap_same_decision": f"{same}/{len(hf)}", "noise70_decision_flips": f"{flips}/{ntot}"}
        (AUD / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
        P("")
        P(f"follow-up: Whisper language-ID log-odds (en vs hi), optimized - stock, all Hindi-speech pairs: "
          f"{dd.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={signflip_p(dd):.3f} (n={len(dd)}); hinglish_medium forced-English "
          f"decisions unchanged when the optimized tokens are re-decoded with HF Mimi: {same}/{len(hf)}; "
          f"decision flips under -70 dBFS noise: {flips}/{ntot}")
    report = "\n".join(L)
    (AUD / f"report{args.tag}.txt").write_text(report + "\n")
    print(report)
    if args.no_pack:
        return

    # ---------------------------------------------------------------- listening pack
    lp = ROOT / "samples" / "listening_pack"
    lp.mkdir(parents=True, exist_ok=True)
    picks = [("en_medium", seeds[0]), ("en_medium", seeds[1]), ("hinglish_medium", seeds[0]), ("hinglish_medium", seeds[1]),
             ("hi_medium", seeds[0]), ("hi_medium", seeds[1])]
    for tid, sd in picks:
        for kind in ("stock", "optimized"):
            src = wav_dir / f"{tid}_seed{sd}_{kind}.wav"
            if src.exists():
                shutil.copyfile(src, lp / f"{tid}_seed{sd}_{kind}.wav")
    print(f"\nwrote {AUD / f'summary{args.tag}.json'}, {AUD / f'report{args.tag}.txt'}, listening pack in {lp}")


if __name__ == "__main__":
    main()
