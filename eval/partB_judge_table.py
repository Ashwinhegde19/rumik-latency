"""Judge scores for the existing seed-1000 sample WAVs (samples/*.wav vs samples/optimized/*.wav).

    .venv/bin/python eval/partB_judge_table.py     # reads results/audit/judge_partB.json

Single random draws per text: useful as a smoke check of the judges, not as evidence either way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit_stats import TEXTS, asr_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    d = json.loads((ROOT / "results" / "audit" / "judge_partB.json").read_text())
    c = d["clips"]
    rows = []
    print(f'{"text":20s} {"kind":9s} {"UTMOSv2":>7s} {"UTMOS22":>7s}  ASR CER (whisper / mms)          transcript (whisper)')
    for tid, t in TEXTS.items():
        for kind in ("baseline", "optimized"):
            j = c.get(f"partB/{kind}/{tid}")
            if not j:
                continue
            a = asr_scores(tid, j)
            w = a.get("whisper_en" if t["lang"] == "en" else "whisper_hi", {}).get("cer")
            m = a.get("mms_eng" if t["lang"] == "en" else "mms_hin", {}).get("cer")
            hyp = j.get("whisper", {}).get("en" if t["lang"] == "en" else "hi", "")
            rows.append({"text_id": tid, "kind": kind, "utmosv2": j.get("utmosv2"), "utmos22": j.get("utmos22"),
                         "cer_whisper": w, "cer_mms": m, "asr": a})
            print(f'{tid:20s} {kind:9s} {j.get("utmosv2", float("nan")):7.3f} {j.get("utmos22", float("nan")):7.3f}  '
                  f'{w if w is not None else float("nan"):6.3f} / {m if m is not None else float("nan"):6.3f}   {hyp[:70]}')
    for kind in ("baseline", "optimized"):
        rs = [r for r in rows if r["kind"] == kind]
        f = lambda k: float(np.mean([r[k] for r in rs if r[k] is not None]))  # noqa: E731
        print(f"mean {kind:9s} UTMOSv2 {f('utmosv2'):.3f} UTMOS22 {f('utmos22'):.3f} CER whisper {f('cer_whisper'):.3f} "
              f"CER mms {f('cer_mms'):.3f}")
    (ROOT / "results" / "audit" / "partB_judge_table.json").write_text(json.dumps({"rows": rows, "errors": d["errors"]},
                                                                                  indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
