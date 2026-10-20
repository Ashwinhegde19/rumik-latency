"""Follow-up on the one anomaly in the judge results (Whisper forced-English on hinglish_medium translates 16/16
stock clips but only 9/16 optimized clips). Runs on a small GPU; no TTS model.

    .venv/bin/modal run eval/audit_followup.py      # writes results/audit/followup.json

1. Language-ID log-odds log p(<|en|>) - log p(<|hi|>) from Whisper-large-v3 (first decoder step) for every
   Hindi-speech clip of both paths: a continuous "how English does this sound" score, paired by seed.
2. Codec swap: the optimized hinglish_medium token sequences re-decoded with the stock HF MimiModel.decode
   (full clip), then Whisper forced-English again. Same decisions -> the difference lives in the tokens,
   not in the streaming decoder.
3. Judge stability: each hinglish_medium clip (both paths) re-transcribed with -70 dBFS white noise added
   (3 noise seeds). Decisions that flip under inaudible noise mean the binary translate/transcribe outcome is
   a knife-edge, not a robust audio property.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REVISION = "0ed3c98684e14350c910129c0efa179841c41ad2"


def _dl_codec():
    from huggingface_hub import snapshot_download
    snapshot_download("rumik-ai/rumik-oss-1", revision=REVISION, allow_patterns=["codec/*"], local_dir="/models/rumik")


# same layers as eval/audit_judge.py (cached), plus the rumik Mimi codec weights
from audit_judge import image as judge_image  # noqa: E402

image = (judge_image.env({"HF_HUB_OFFLINE": "0"}).run_function(_dl_codec).env({"HF_HUB_OFFLINE": "1"})
         .add_local_python_source("audit_judge"))
app = modal.App("rumik-quality-followup", image=image)


@app.function(gpu="L4", cpu=4.0, memory=24576, timeout=1800)
def followup(clips: list[dict], opt_tokens: dict) -> dict:
    import io

    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import MimiModel, WhisperForConditionalGeneration, WhisperProcessor

    t0 = time.perf_counter()
    proc = WhisperProcessor.from_pretrained("/models/whisper")
    wm = WhisperForConditionalGeneration.from_pretrained("/models/whisper", torch_dtype=torch.float16).cuda().eval()
    tokz = proc.tokenizer
    sot = tokz.convert_tokens_to_ids("<|startoftranscript|>")
    en_id, hi_id = tokz.convert_tokens_to_ids("<|en|>"), tokz.convert_tokens_to_ids("<|hi|>")

    def to16(wav_bytes):
        x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return resample_poly(x.astype(np.float64), 2, 3).astype(np.float32)

    @torch.inference_mode()
    def lid(xs):
        out = []
        for i in range(0, len(xs), 8):
            f = proc(xs[i:i + 8], sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
            d = torch.full((f.shape[0], 1), sot, device="cuda")
            lg = wm(input_features=f, decoder_input_ids=d).logits[:, -1].float().log_softmax(-1)
            out += (lg[:, en_id] - lg[:, hi_id]).tolist()
        return out

    @torch.inference_mode()
    def asr_en(xs):
        out = []
        for i in range(0, len(xs), 8):
            f = proc(xs[i:i + 8], sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
            ids = wm.generate(f, language="en", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=400)
            out += [s.strip() for s in proc.batch_decode(ids, skip_special_tokens=True)]
        return out

    x16 = {c["key"]: to16(c["wav"]) for c in clips}
    keys = list(x16)
    res = {"lid_logodds_en_vs_hi": dict(zip(keys, lid([x16[k] for k in keys])))}

    # codec swap: HF full-clip decode of the optimized hinglish_medium tokens
    mimi = MimiModel.from_pretrained("/models/rumik/codec").eval().cuda()
    first, last, Q, end = 261008, 277391, 8, 277394  # config.json of the pinned revision

    def to_codes(toks):  # == RumikOSSForCausalLM.audio_tokens_to_codes
        frames, frame = [], []
        for t in toks:
            if t == end:
                break
            if not first <= t <= last:
                frame = []
                continue
            code, q = divmod(t - first, Q)
            if q == len(frame):
                frame.append(code)
                if len(frame) == Q:
                    frames.append(frame)
                    frame = []
            else:
                frame = [code] if q == 0 else []
        return torch.tensor(frames, dtype=torch.long).T.unsqueeze(0)

    hf = {}
    snr = {}
    for k, toks in opt_tokens.items():
        with torch.inference_mode():
            w = mimi.decode(to_codes(toks).cuda()).audio_values[0, 0].float().clamp(-1, 1).cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, w, 24000, format="WAV", subtype="PCM_16")  # same quantisation as the clip files
        hf[k] = to16(buf.getvalue())
        a, _ = sf.read(io.BytesIO(buf.getvalue()), dtype="float64")
        b, _ = sf.read(io.BytesIO(next(c["wav"] for c in clips if c["key"] == f"{k}_optimized")), dtype="float64")
        n = min(len(a), len(b))
        snr[k] = float(10 * np.log10(np.sum(a[:n] ** 2) / max(np.sum((a[:n] - b[:n]) ** 2), 1e-30)))
    hk = list(hf)
    res["hf_redecode_of_optimized_tokens"] = {"asr_en": dict(zip(hk, asr_en([hf[k] for k in hk]))),
                                              "lid": dict(zip(hk, lid([hf[k] for k in hk]))),
                                              "snr_vs_streamed_db": snr}

    # judge stability under -70 dBFS white noise
    hm = [k for k in keys if k.startswith("hinglish_medium_")]
    res["asr_en_clean"] = dict(zip(hm, asr_en([x16[k] for k in hm])))
    noisy = {}
    for s in range(3):
        rng = np.random.default_rng(100 + s)
        xs = [x16[k] + (10 ** (-70 / 20)) * rng.standard_normal(len(x16[k])).astype(np.float32) for k in hm]
        noisy[str(s)] = dict(zip(hm, asr_en(xs)))
    res["asr_en_noise70"] = noisy
    res["wall_s"] = time.perf_counter() - t0
    return json.loads(json.dumps(res))


@app.local_entrypoint()
def main(out: str = "results/audit/followup.json"):
    texts = {t["id"]: t for t in json.loads((ROOT / "texts.json").read_text())}
    gen = json.loads((ROOT / "results" / "audit" / "gen.json").read_text())
    clips = []
    for f in sorted((ROOT / "results" / "audit" / "wavs").glob("*.wav")):
        tid = f.stem.rsplit("_seed", 1)[0]
        if texts[tid]["lang"] != "en":
            clips.append({"key": f.stem, "wav": f.read_bytes()})
    opt_tokens = {k: v["tokens"] for k, v in gen["opt"].items() if k.startswith("hinglish_medium_")}
    print(f"{len(clips)} clips, {len(opt_tokens)} token sequences")
    t0 = time.time()
    data = followup.remote(clips, opt_tokens)
    data["wall_clock_remote_call_s"] = time.time() - t0
    Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
