"""Cross-run reproducibility: does this audit's output match earlier runs with the same seed?

    .venv/bin/python eval/cross_run_check.py

* stock audio-token counts vs results/baseline.json (bench_modal.py, non-streaming generate_audio, seeds 1000-1004)
* optimized audio-token counts vs results/optimized.json (bench_fast.py bench part, seeds 1000-1009)
* full token sequences vs the smoke run (fresh container, no previous requests)
Matching counts across separate containers and request histories means the paths are deterministic and the
optimized engine carries no state from one request into the next.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "results" / "audit"


def main():
    gen = json.loads((AUD / "gen.json").read_text())
    base = json.loads((ROOT / "results" / "baseline.json").read_text())
    optb = json.loads((ROOT / "results" / "optimized.json").read_text())
    out = {"stock_vs_baseline_json": [], "opt_vs_optimized_json": [], "vs_smoke": []}
    for r in base["runs"]:
        key = f'{r["text_id"]}_seed{1000 + r["run"]}'
        if r["nonstreaming"] and key in gen["stock"]:
            out["stock_vs_baseline_json"].append(gen["stock"][key]["n_audio_tokens"] == r["nonstreaming"]["audio_tokens"])
    for r in optb["bench"]["runs"]:
        key = f'{r["text_id"]}_seed{1000 + r["run"]}'
        if key in gen["opt"]:
            out["opt_vs_optimized_json"].append(gen["opt"][key]["n_audio_tokens"] == r["streaming"]["audio_tokens"])
    sm = AUD / "smoke_gen.json"
    if sm.exists():
        s = json.loads(sm.read_text())
        for kind in ("stock", "opt"):
            for k, v in s[kind].items():
                if k in gen[kind]:
                    out["vs_smoke"].append({"kind": kind, "key": k, "tokens_equal": v["tokens"] == gen[kind][k]["tokens"]})
    summ = {"stock_counts_match_baseline_json": f'{sum(out["stock_vs_baseline_json"])}/{len(out["stock_vs_baseline_json"])}',
            "opt_counts_match_optimized_json": f'{sum(out["opt_vs_optimized_json"])}/{len(out["opt_vs_optimized_json"])}',
            "tokens_match_smoke_run": f'{sum(x["tokens_equal"] for x in out["vs_smoke"])}/{len(out["vs_smoke"])}'}
    out["summary"] = summ
    (AUD / "cross_run_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
