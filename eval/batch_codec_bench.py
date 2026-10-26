#!/usr/bin/env python3
"""How long does the streaming Mimi decode take for B listeners per 80 ms frame? Measurement only.

StreamingMimi (rumik_fast.py) with a batch dimension: per-listener conv/transposed-conv state and transformer KV
cache, same weights and maths, CUDA-graph captured. Timed per frame at B = 1..128, plus the batch-1 decoder replayed
B times back to back (one listener at a time). Then combined with results/batch_step.json:
    a batch of B listeners keeps up iff 8 * lm_step_ms(B) + codec_ms(B) <= 80 ms.

Correctness: identical rows give identical PCM; row 0 equals the batch-1 StreamingMimi over the same code frames.

    .venv/bin/modal run eval/batch_codec_bench.py
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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install(requirements=[str(ROOT / "requirements.txt")])
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(ROOT / "rumik_fast.py"), "/root/rumik_fast.py")
)
volume = modal.Volume.from_name("rumik-hf-cache", create_if_missing=False)
app = modal.App("rumik-batch-codec", image=image)


@app.function(gpu="H100", cpu=4.0, memory=32768, volumes={HF_CACHE: volume}, timeout=1800)
def bench(cfg: dict) -> dict:
    import sys
    import traceback
    from time import perf_counter

    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import MimiModel

    sys.path.insert(0, "/root")
    from rumik_fast import StreamingMimi

    t0 = perf_counter()
    log = lambda *a: print(f"[{perf_counter() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    root = snapshot_download(REPO_ID, revision=REVISION, token=False, allow_patterns=["codec/*"])
    mimi = MimiModel.from_pretrained(f"{root}/codec").eval().cuda()
    torch.backends.cuda.matmul.allow_tf32 = True  # as the engine runs (default)
    torch.backends.cudnn.allow_tf32 = True

    class BatchedMimi(StreamingMimi):
        """StreamingMimi with B listeners: every state tensor gets a leading batch dimension B."""

        def __init__(self, mimi, B: int):
            self.B = B
            super().__init__(mimi, use_graphs=False)
            with torch.inference_mode():
                self.states_b = {}
                for st in self.states:
                    self.states_b[id(st)] = torch.zeros((B,) + tuple(st.shape[1:]), device=self.dev, dtype=self.dt)
                remap = lambda st: self.states_b[id(st)] if st is not None else None  # noqa: E731
                self.up_state = remap(self.up_state)
                plan = []
                for kind, mod, st in self.plan:
                    if kind == "res":
                        plan.append((kind, [(k2, m2, remap(s2)) for k2, m2, s2 in mod], None))
                    else:
                        plan.append((kind, mod, remap(st)))
                self.plan = plan
                self.states = list(self.states_b.values())
                self.kc = [torch.zeros((B,) + tuple(k.shape[1:]), device=self.dev, dtype=self.dt) for k in self.kc]
                self.vc = [torch.zeros_like(k) for k in self.kc]
                self.codes_buf = torch.zeros(B, self.n_q, 1, dtype=torch.long, device=self.dev)
                self.pcm_buf = torch.zeros(B, self.spf, device=self.dev, dtype=self.dt)
                self.use_graphs = True
                self._capture()

        def _transformer(self, x):  # x [B, T, C]
            B, T = x.shape[0], x.shape[1]
            pos = self.tpos + self.t_ar[:T]
            cos = self.cos.index_select(0, pos).view(1, 1, T, self.thd)
            sin = self.sin.index_select(0, pos).view(1, 1, T, self.thd)
            mask = (self.tar.view(1, -1) <= pos.view(-1, 1)).view(1, 1, T, self.t_max)
            half = self.thd // 2

            def rope(t):
                rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
                return t * cos + rot * sin

            for i, layer in enumerate(self.tlayers):
                at = layer.self_attn
                h = layer.input_layernorm(x)
                q = at.q_proj(h).view(B, T, self.nh, self.thd).transpose(1, 2)
                k = at.k_proj(h).view(B, T, self.nh, self.thd).transpose(1, 2)
                v = at.v_proj(h).view(B, T, self.nh, self.thd).transpose(1, 2)
                q, k = rope(q), rope(k)
                self.kc[i].index_copy_(2, pos, k)
                self.vc[i].index_copy_(2, pos, v)
                o = F.scaled_dot_product_attention(q, self.kc[i], self.vc[i], attn_mask=mask)
                o = at.o_proj(o.transpose(1, 2).reshape(B, T, -1))
                x = x + layer.self_attn_layer_scale(o)
                h = layer.post_attention_layernorm(x)
                x = x + layer.mlp_layer_scale(layer.mlp(h))
            self.tpos.add_(T)
            return x

        def _frame(self):
            x = self._rvq(self.codes_buf)
            x = self._convt(x, self.up.conv, self.up_state)
            x = self._transformer(x.transpose(1, 2)).transpose(1, 2)
            for kind, mod, st in self.plan:
                if kind == "conv":
                    x = self._conv(x, mod, st)
                elif kind == "convt":
                    x = self._convt(x, mod, st)
                elif kind == "act":
                    x = mod(x)
                else:
                    r = x
                    for k2, m2, s2 in mod:
                        x = self._conv(x, m2, s2) if k2 == "conv" else m2(x)
                    x = r + x
            self.pcm_buf.copy_(x.reshape(self.B, -1))

    def time_graph(replay, n):
        for _ in range(5):
            replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    checks, results, errors = {}, [], []
    with torch.inference_mode():
        one = StreamingMimi(mimi)
        g = torch.Generator(device="cuda").manual_seed(0)
        frames = torch.randint(0, 2048, (12, 8), device="cuda", generator=g)
        # ---- correctness
        try:
            one.reset()
            ref = torch.stack([one.decode_frame(frames[t].view(1, 8, 1)).clone() for t in range(12)])
            b4 = BatchedMimi(mimi, 4)
            b4.reset()
            outs = []
            for t in range(12):
                b4.codes_buf.copy_(frames[t].view(1, 8, 1).expand(4, 8, 1))
                b4.graph.replay()
                outs.append(b4.pcm_buf.clone())
            got = torch.stack(outs)  # [12, 4, spf]
            checks["identical_rows_max_abs_diff"] = float((got - got[:, :1]).abs().max())
            checks["row0_vs_batch1_max_abs_diff"] = float((got[:, 0] - ref).abs().max())
            checks["row0_vs_batch1_snr_db"] = float(10 * torch.log10(ref.pow(2).mean() / (got[:, 0] - ref).pow(2).mean().clamp_min(1e-20)))
            log("checks", checks)
            del b4
        except Exception:
            errors.append({"where": "checks", "err": traceback.format_exc()[-3000:]})
        # ---- batch-1 decoder, one listener at a time
        one.reset()
        one.codes_buf.copy_(frames[0].view(1, 8, 1))
        t1 = time_graph(one.graph.replay, cfg["iters"])
        log(f"batch-1 decode per frame: {t1:.3f} ms")
        # ---- batched decoder
        for B in cfg["batches"]:
            try:
                bm = BatchedMimi(mimi, B)
                bm.reset()
                bm.codes_buf.copy_(frames[0].view(1, 8, 1).expand(B, 8, 1))
                ms = time_graph(bm.graph.replay, cfg["iters"])
                results.append({"B": B, "batched_ms": ms, "serial_ms": B * t1})
                log(f"B={B:4d} batched {ms:7.3f} ms per frame | one-at-a-time {B * t1:7.2f} ms")
                del bm
                torch.cuda.empty_cache()
            except Exception:
                errors.append({"where": f"B={B}", "err": traceback.format_exc()[-3000:]})
    return json.loads(json.dumps({"env": {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__),
                                          "mimi_dtype": str(next(mimi.parameters()).dtype), "tf32": True},
                                  "cfg": cfg, "checks": checks, "batch1_frame_ms": t1, "results": results,
                                  "errors": errors, "wall_s": perf_counter() - t0}, default=str))


@app.local_entrypoint()
def main():
    cfg = {"batches": [1, 2, 4, 8, 16, 32, 64, 128], "iters": 200}
    t = time.time()
    data = bench.remote(cfg)
    data["wall_clock_remote_call_s"] = time.time() - t
    lm = json.loads((ROOT / "results" / "batch_step.json").read_text())["results"]
    step = {(r["B"], r["Lb"]): r["step_ms"] for r in lm}
    combined = []
    for r in data["results"]:
        B = r["B"]
        for Lb in (512, 1024):
            if (B, Lb) not in step:
                continue
            lm8 = 8 * step[(B, Lb)]
            row = {"B": B, "Lb": Lb, "lm_8_steps_ms": lm8, "codec_batched_ms": r["batched_ms"], "codec_serial_ms": r["serial_ms"],
                   "total_batched_ms": lm8 + r["batched_ms"], "total_serial_ms": lm8 + r["serial_ms"]}
            row["keeps_up_batched"] = row["total_batched_ms"] <= 80.0
            row["keeps_up_serial"] = row["total_serial_ms"] <= 80.0
            combined.append(row)
    data["combined_per_80ms_frame"] = combined
    out = ROOT / "results" / "batch_codec.json"
    out.write_text(json.dumps(data, indent=1))
    print("checks:", data["checks"], "| batch-1 frame:", round(data["batch1_frame_ms"], 3), "ms | errors:", len(data["errors"]))
    print(f"{'B':>4} {'cache':>5} {'LM 8 steps':>10} {'codec batch':>11} {'codec 1-by-1':>12} {'total batch':>11} {'total 1-by-1':>12}  fits(batch/1-by-1)")
    for r in combined:
        print(f"{r['B']:4d} {r['Lb']:5d} {r['lm_8_steps_ms']:10.1f} {r['codec_batched_ms']:11.2f} {r['codec_serial_ms']:12.1f} "
              f"{r['total_batched_ms']:11.1f} {r['total_serial_ms']:12.1f}  {r['keeps_up_batched']}/{r['keeps_up_serial']}")
    for e in data["errors"]:
        print(e["where"], e["err"][-1200:])
