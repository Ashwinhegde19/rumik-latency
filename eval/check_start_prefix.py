#!/usr/bin/env python3
"""Engine check for StreamingTTS.run(start_prefix=...): with no prefix the tokens must equal the audited k0
clips, and with START_PREFIX they must equal the k3 clips produced by eval/prefix_fix_gen.py (same seeds).

    .venv/bin/modal run eval/check_start_prefix.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "rumik-ai/rumik-oss-1"
REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"
HF_CACHE = "/hf-cache"
GEN_KW = dict(max_new_tokens=2048, min_new_tokens=8, temperature=0.8, top_k=30)
FINAL = dict(use_compile=True, kv_buckets=(256, 512, 1024), attn="f32", compile_prefill=True, prefill_buckets=(32, 64, 128, 256))

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
app = modal.App("rumik-check-start-prefix", image=image)


@app.function(gpu="H100", cpu=4.0, memory=32768, volumes={HF_CACHE: volume}, timeout=1800)
def check(cases: list[dict]) -> dict:
    import sys
    from time import perf_counter

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

    sys.path.insert(0, "/root")
    from rumik_fast import START_PREFIX, FastLM, StreamingMimi, StreamingTTS

    t0 = perf_counter()
    root = snapshot_download(REPO_ID, revision=REVISION, token=False)
    tmap = {t["id"]: t for t in json.loads(Path("/root/texts.json").read_text())}
    prompt = lambda t: f'<text>{t["speaker"]}: <description="{t["description"]}"> {t["text"]}<audio>'  # noqa: E731
    tok = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    mimi = MimiModel.from_pretrained(f"{root}/codec").eval().cuda()
    lm = FastLM(model, **FINAL, **GEN_KW)
    tts = StreamingTTS(lm, tok, smimi=StreamingMimi(mimi))
    for i in range(2):
        tts.run(prompt(tmap["en_medium"]), 10_000 + i)
    out = []
    for c in cases:
        r = tts.run(prompt(tmap[c["text_id"]]), c["seed"], start_prefix=START_PREFIX if c["cond"] == "k3" else None)
        out.append({"key": c["key"], "tokens_equal": r["tokens"] == c["tokens"], "n": (len(r["tokens"]), len(c["tokens"])),
                    "frames": (r["frames"], c["frames"]), "prefix_frames": r.get("prefix_frames"),
                    "first_nonsilent_frame": (r["first_nonsilent_frame"], c["first_nonsilent_frame"]),
                    "ttfa_nonsilent_ms": (round(r["ttfa_nonsilent_s"] * 1e3, 1) if r["ttfa_nonsilent_s"] is not None else None,
                                          round(c["ttfa_nonsilent_s"] * 1e3, 1) if c["ttfa_nonsilent_s"] is not None else None),
                    "ttfa_ms": round(r["ttfa_s"] * 1e3, 1), "audio_seconds": r["audio_seconds"]})
        print(out[-1], flush=True)
    return {"gpu": str(torch.cuda.get_device_name(0)), "cases": out, "wall_s": perf_counter() - t0}


@app.local_entrypoint()
def main(texts: str = "hi_en_mixed_medium,en_short,hinglish_long", seed: int = 1000):
    gen = json.loads((ROOT / "results" / "prefix_fix" / "gen.json").read_text())
    want = {(r["text_id"], r["seed"], r["cond"]): r for r in gen["results"]}
    cases = [dict(want[(t, seed, c)], text_id=t, seed=seed, cond=c) for t in texts.split(",") for c in ("k0", "k3")]
    cases = [{k: v for k, v in c.items() if k in ("key", "text_id", "seed", "cond", "tokens", "frames", "first_nonsilent_frame", "ttfa_nonsilent_s")} for c in cases]
    t = time.time()
    res = check.remote(cases)
    res["wall_clock_s"] = time.time() - t
    (ROOT / "results" / "prefix_fix" / "engine_check.json").write_text(json.dumps(res, indent=1))
    ok = all(c["tokens_equal"] for c in res["cases"])
    print(f"ALL TOKENS EQUAL: {ok}  ({len(res['cases'])} cases, gpu {res['gpu']}, {res['wall_s']:.0f}s)")
