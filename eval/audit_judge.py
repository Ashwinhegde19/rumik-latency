"""Objective judges for the quality audit (runs on a small Modal GPU; no TTS model involved).

    .venv/bin/modal run eval/audit_judge.py --wav-dir "" --out results/audit/judge_partB.json        # existing samples only
    .venv/bin/modal run eval/audit_judge.py --wav-dir results/audit/wavs --out results/audit/judge.json

Per clip it returns raw judge outputs; scoring (CER/WER normalisation, speaker similarity, statistics) is done
locally by eval/audit_stats.py so it can be changed without re-running anything here.

  * ASR: openai/whisper-large-v3 (greedy, forced language: "en" for English texts; "hi" for every Hindi-speech
    text, plus "en" for romanized Hinglish), and facebook/mms-1b-all (CTC; "hin" adapter for Hindi-speech texts,
    "eng" for English). MMS is an independent architecture, so a degradation that one recognizer hides should
    still show up in the other.
  * naturalness: UTMOS22 strong (SpeechMOS port) and UTMOSv2.
  * speaker: SpeechBrain ECAPA-TDNN (VoxCeleb) and microsoft/wavlm-base-plus-sv embeddings.
All judges get the same 16 kHz audio (scipy resample_poly 24k -> 16k) for both paths.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
MODELS = {  # pinned revisions
    "whisper": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "mms": ("facebook/mms-1b-all", "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"),
    "wavlm_sv": ("microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"),
    "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"),
}
UTMOSV2_COMMIT = "cc2700db57bb83ee13dc31ebe1b868c254e15d09"
SPEECHMOS_TAG = "v1.2.0"


def _download():
    import torch
    from huggingface_hub import snapshot_download
    pats = {"whisper": ["*.json", "model.safetensors", "*.txt"],
            "mms": ["*.json", "model.safetensors", "adapter.hin.safetensors", "adapter.eng.safetensors"],
            "wavlm_sv": ["*.json", "pytorch_model.bin"],
            "ecapa": ["*.yaml", "*.ckpt", "*.txt", "*.json"]}
    for k, (repo, rev) in MODELS.items():
        snapshot_download(repo, revision=rev, allow_patterns=pats[k], local_dir=f"/models/{k}")
    torch.hub.load(f"tarepan/SpeechMOS:{SPEECHMOS_TAG}", "utmos22_strong", trust_repo=True)
    import utmosv2
    utmosv2.create_model(pretrained=True, device="cpu")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .uv_pip_install("torch==2.8.0", "torchaudio==2.8.0", "transformers==4.57.6", "accelerate==1.15.0",
                    "huggingface_hub[hf_transfer]==0.36.2", "safetensors", "soundfile==0.14.0", "scipy",
                    "numpy<2.3", "librosa==0.11.0", "speechbrain==1.1.2", "timm", "sentencepiece",
                    f"utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git@{UTMOSV2_COMMIT}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .apt_install("wget")  # utmosv2 fetches its weights with wget
    .run_function(_download)
    .env({"HF_HUB_OFFLINE": "1"})
)
app = modal.App("rumik-quality-judge", image=image)


@app.function(gpu="L4", cpu=4.0, memory=24576, timeout=3000)
def judge(clips: list[dict]) -> dict:
    import io
    import random
    import traceback

    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    t0 = time.perf_counter()
    log = lambda *a: print(f"[{time.perf_counter() - t0:6.1f}s]", *a, flush=True)  # noqa: E731
    errors = []
    wav16 = {}
    for c in clips:
        x, sr = sf.read(io.BytesIO(c["wav"]), dtype="float32")
        assert sr == 24000, sr
        wav16[c["key"]] = resample_poly(x.astype(np.float64), 2, 3).astype(np.float32)
    out = {c["key"]: {"lang": c["lang"], "dur16_s": len(wav16[c["key"]]) / 16000} for c in clips}
    hindi_speech = [c["key"] for c in clips if c["lang"] != "en"]
    english = [c["key"] for c in clips if c["lang"] == "en"]
    hinglish = [c["key"] for c in clips if c["lang"] == "hinglish"]

    # ---------------------------------------------------------------- Whisper large-v3
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained("/models/whisper")
        wm = WhisperForConditionalGeneration.from_pretrained("/models/whisper", torch_dtype=torch.float16).cuda().eval()

        def run_whisper(keys, lang):
            for i in range(0, len(keys), 8):
                ks = keys[i:i + 8]
                feats = proc([wav16[k] for k in ks], sampling_rate=16000, return_tensors="pt").input_features
                with torch.inference_mode():
                    ids = wm.generate(feats.to("cuda", torch.float16), language=lang, task="transcribe",
                                      num_beams=1, do_sample=False, max_new_tokens=400)
                for k, txt in zip(ks, proc.batch_decode(ids, skip_special_tokens=True)):
                    out[k].setdefault("whisper", {})[lang] = txt.strip()

        run_whisper(english, "en")
        run_whisper(hindi_speech, "hi")
        run_whisper(hinglish, "en")
        del wm
        torch.cuda.empty_cache()
        log("whisper done")
    except Exception:
        errors.append({"where": "whisper", "err": traceback.format_exc()[-3000:]})

    # ---------------------------------------------------------------- MMS-1b-all (CTC)
    try:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        mp = AutoProcessor.from_pretrained("/models/mms")
        mm = Wav2Vec2ForCTC.from_pretrained("/models/mms").cuda().eval()
        for lang, keys in (("hin", hindi_speech), ("eng", english)):
            if not keys:
                continue
            mp.tokenizer.set_target_lang(lang)
            mm.load_adapter(lang)
            for k in keys:
                inp = mp(wav16[k], sampling_rate=16000, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    lg = mm(**inp).logits
                out[k].setdefault("mms", {})[lang] = mp.decode(lg.argmax(-1)[0]).strip()
        del mm
        torch.cuda.empty_cache()
        log("mms done")
    except Exception:
        errors.append({"where": "mms", "err": traceback.format_exc()[-3000:]})

    # ---------------------------------------------------------------- UTMOS22 strong
    try:
        u22 = torch.hub.load(f"tarepan/SpeechMOS:{SPEECHMOS_TAG}", "utmos22_strong", trust_repo=True,
                             skip_validation=True).cuda().eval()
        for k in out:
            with torch.inference_mode():
                out[k]["utmos22"] = float(u22(torch.from_numpy(wav16[k])[None].cuda(), 16000).reshape(-1)[0])
        del u22
        log("utmos22 done")
    except Exception:
        errors.append({"where": "utmos22", "err": traceback.format_exc()[-3000:]})

    # ---------------------------------------------------------------- UTMOSv2
    try:
        import utmosv2
        u2 = utmosv2.create_model(pretrained=True, device="cuda")
        d = Path("/tmp/utmosv2_in")
        d.mkdir(parents=True, exist_ok=True)
        names = {}
        for i, k in enumerate(out):
            sf.write(d / f"c{i:04d}.wav", wav16[k], 16000, subtype="FLOAT")
            names[f"c{i:04d}.wav"] = k
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        res = u2.predict(input_dir=str(d), num_repetitions=1, batch_size=16, num_workers=4, verbose=False,
                         device="cuda")
        for r in res:
            out[names[Path(r["file_path"]).name]]["utmosv2"] = float(r["predicted_mos"])
        del u2
        torch.cuda.empty_cache()
        log("utmosv2 done")
    except Exception:
        errors.append({"where": "utmosv2", "err": traceback.format_exc()[-3000:]})

    # ---------------------------------------------------------------- speaker embeddings
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        ec = EncoderClassifier.from_hparams(source="/models/ecapa", savedir="/tmp/ecapa", run_opts={"device": "cuda"},
                                           overrides={"pretrained_path": "/models/ecapa"})
        for k in out:
            with torch.inference_mode():
                e = ec.encode_batch(torch.from_numpy(wav16[k])[None].cuda())
            out[k]["ecapa"] = [round(float(v), 6) for v in e.reshape(-1).tolist()]
        log("ecapa done")
    except Exception:
        errors.append({"where": "ecapa", "err": traceback.format_exc()[-3000:]})
    try:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        fe = AutoFeatureExtractor.from_pretrained("/models/wavlm_sv")
        wl = WavLMForXVector.from_pretrained("/models/wavlm_sv").cuda().eval()
        for k in out:
            inp = fe(wav16[k], sampling_rate=16000, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                e = wl(**inp).embeddings
            out[k]["wavlm_sv"] = [round(float(v), 6) for v in e.reshape(-1).tolist()]
        log("wavlm-sv done")
    except Exception:
        errors.append({"where": "wavlm_sv", "err": traceback.format_exc()[-3000:]})

    import transformers
    env = {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__),
           "transformers": str(transformers.__version__), "models": MODELS, "utmosv2_commit": UTMOSV2_COMMIT,
           "speechmos": SPEECHMOS_TAG, "wall_s": time.perf_counter() - t0}
    return json.loads(json.dumps({"env": env, "clips": out, "errors": errors}))  # plain JSON types only


@app.local_entrypoint()
def main(wav_dir: str = "results/audit/wavs", include_partb: bool = True, out: str = "results/audit/judge.json"):
    texts = {t["id"]: t for t in json.loads((ROOT / "texts.json").read_text())}
    clips = []
    if include_partb:
        for tid, t in texts.items():
            clips.append({"key": f"partB/baseline/{tid}", "lang": t["lang"],
                          "wav": (ROOT / "samples" / f"{tid}.wav").read_bytes()})
            clips.append({"key": f"partB/optimized/{tid}", "lang": t["lang"],
                          "wav": (ROOT / "samples" / "optimized" / f"{tid}.wav").read_bytes()})
    if wav_dir:
        for f in sorted(Path(wav_dir).glob("*.wav")):
            m = re.match(r"(.+?)_seed(\d+)_([A-Za-z0-9]+)\.wav$", f.name)
            if not m:
                continue
            clips.append({"key": f"gen/{f.stem}", "lang": texts[m.group(1)]["lang"], "wav": f.read_bytes()})
    print(f"judging {len(clips)} clips")
    t0 = time.time()
    data = judge.remote(clips)
    data["wall_clock_remote_call_s"] = time.time() - t0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, ensure_ascii=False))
    print(f"wrote {out}; errors: {json.dumps(data['errors'])[:3000]}")
