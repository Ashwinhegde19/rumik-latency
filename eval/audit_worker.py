"""Stock-path worker for eval/audit_gen.py (a subprocess inside the Modal GPU container).

Each worker loads the stock model exactly as bench_modal.py does (bf16, SDPA) plus the fp32 HF Mimi, then
claims jobs from a shared job list and runs the model-card path: generate_audio + MimiModel.decode of the
whole clip. The stock loop is CPU/launch bound (the GPU is mostly idle), so several workers can share one
GPU; each generation is still an unmodified batch-1 generate_audio call.

Forward hooks record the stock incremental logits (audio ids only) and stop-predictor logits during
generation for the teacher-forcing reference jobs. Hooks only read outputs; they do not change the
computation.

    python audit_worker.py --rank 0 --root <snapshot dir> --work /tmp/audit --gen-kw '{...}' [--delay 10]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import time
import traceback
from pathlib import Path


class StockRunner:
    def __init__(self, tok, model, mimi):
        import torch
        self.torch = torch
        self.tok, self.model, self.mimi = tok, model, mimi
        self.aud = model.audio_token_ids(device="cuda")
        self.end_id = int(model.config.audio_end_token_id)
        self.cap_l: list = []
        self.cap_s: list = []
        self.rec = False

        def lm_hook(mod, inp, out):
            if self.rec:
                self.cap_l.append(out[0, -1].index_select(0, self.aud).clone())

        def stop_hook(mod, inp, out):
            self.cap_s.append(out.reshape(-1)[-1].float().clone())

        self.hooks = [model.lm_head.register_forward_hook(lm_hook),
                      model.stop_predictor.register_forward_hook(stop_hook)]

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def run_job(self, job: dict, gen_kw: dict) -> dict:
        import soundfile as sf
        torch = self.torch
        self.cap_l.clear()
        self.cap_s.clear()
        self.rec = bool(job.get("record"))
        try:
            inputs = self.tok(job["prompt"], return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.manual_seed(job["seed"])
            with torch.inference_mode():
                ids = self.model.generate_audio(**inputs, **gen_kw, do_sample=True)
            torch.cuda.synchronize()
            t_gen = time.perf_counter() - t0
            toks = [int(x) for x in ids[0].tolist()[inputs.input_ids.shape[1]:]]
            with torch.inference_mode():
                codes = self.model.audio_tokens_to_codes(toks).to("cuda")
                wav = self.mimi.decode(codes).audio_values[0, 0].float()
            buf = io.BytesIO()
            sf.write(buf, wav.clamp(-1, 1).cpu().numpy(), 24000, format="WAV", subtype="PCM_16")
            n_audio = len([t for t in toks if t != self.end_id])
            last_stop = float(self.cap_s[-1]) if self.cap_s else None
            if n_audio >= gen_kw["max_new_tokens"]:
                stopped_by = "max_new_tokens"
            elif last_stop is not None and last_stop > 0:
                stopped_by = "stop_predictor"
            else:
                stopped_by = "audio_end_token"
            res = {"key": job["key"], "text_id": job["text_id"], "seed": job["seed"], "tokens": toks,
                   "n_audio_tokens": n_audio, "frames": int(codes.shape[-1]), "stopped_by": stopped_by,
                   "last_stop_logit": last_stop, "gen_s": t_gen, "wav": buf.getvalue(),
                   "prompt_tokens": int(inputs.input_ids.shape[1])}
            if self.rec:
                b = io.BytesIO()
                torch.save({"logits": torch.stack(self.cap_l).cpu(),   # [1 + n_fed, 16385] bf16
                            "stop_logits_from_step8": (torch.stack(self.cap_s).cpu() if self.cap_s
                                                       else torch.zeros(0))}, b)
                res["tf_ref"] = b.getvalue()
            return res
        finally:
            self.rec = False


def claim(work: Path, key: str) -> bool:
    try:
        fd = os.open(work / "claims" / key, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def write_result(work: Path, key: str, obj: dict):
    tmp = work / "out" / f".{key}.tmp"
    tmp.write_bytes(pickle.dumps(obj))
    os.replace(tmp, work / "out" / f"{key}.pkl")


def run_claimed_jobs(runner: StockRunner, work: Path, gen_kw: dict, tag: str):
    jobs = json.loads((work / "jobs.json").read_text())
    n = 0
    for job in jobs:
        if not claim(work, job["key"]):
            continue
        try:
            res = runner.run_job(job, gen_kw)
            res["worker"] = tag
        except Exception:
            res = {"key": job["key"], "error": traceback.format_exc()[-3000:], "worker": tag}
        write_result(work, job["key"], res)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gen-kw", required=True)
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    work = Path(a.work)
    gen_kw = json.loads(a.gen_kw)
    time.sleep(a.delay)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel
        tok = AutoTokenizer.from_pretrained(a.root, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(a.root, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
        mimi = MimiModel.from_pretrained(f"{a.root}/codec").eval().cuda()
        runner = StockRunner(tok, model, mimi)
    except Exception:
        (work / "out" / f"fatal_{a.rank}.txt").write_text(traceback.format_exc())
        return
    (work / "out" / f"ready_{a.rank}").write_text(str(time.time()))
    n = run_claimed_jobs(runner, work, gen_kw, f"worker{a.rank}")
    (work / "out" / f"done_{a.rank}").write_text(str(n))


if __name__ == "__main__":
    main()
