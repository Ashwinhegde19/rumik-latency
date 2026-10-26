#!/usr/bin/env python3
"""How long does one decode step take when B listeners share it? Measurement only.

FastLM's decode step (same weights, same Cohere2 maths, fp32-reduction attention, same top-k/exponential
sampling) rewritten with a batch dimension, torch.compile'd and CUDA-graph captured per (B, KV bucket).
Listeners sit at different positions inside the bucket. Timed by replaying the graph back to back.
No scheduler, no codec: this is the language model's share of the per-listener budget only.

Correctness: (a) B identical rows give identical logits; (b) row 0 at B=1 matches FastLM's own batch-1 step
on the same cache contents and token.

    .venv/bin/modal run eval/batch_step_bench.py [--smoke]
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
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "TORCHINDUCTOR_CACHE_DIR": f"{HF_CACHE}/inductor-cache", "TRITON_CACHE_DIR": f"{HF_CACHE}/triton-cache"})
    .add_local_file(str(ROOT / "rumik_fast.py"), "/root/rumik_fast.py")
)
volume = modal.Volume.from_name("rumik-hf-cache", create_if_missing=False)
app = modal.App("rumik-batch-step", image=image)


@app.function(gpu="H100", cpu=4.0, memory=65536, volumes={HF_CACHE: volume}, timeout=3600)
def bench(cfg: dict) -> dict:
    import sys
    import traceback
    from time import perf_counter

    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, "/root")
    from rumik_fast import FastLM, _cohere_ln, _rope_interleaved

    t0 = perf_counter()
    log = lambda *a: print(f"[{perf_counter() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    root = snapshot_download(REPO_ID, revision=REVISION, token=False)
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
    lm = FastLM(model, use_compile=False, use_graphs=False, kv_buckets=(256, 512, 1024), attn="f32")
    dev, dt = lm.dev, lm.dt
    H, KV, HD, G = lm.n_heads, lm.n_kv, lm.hd, lm.n_heads // lm.n_kv
    torch._dynamo.config.cache_size_limit = 256
    torch.backends.cuda.matmul.allow_tf32 = False
    log("weights ready")

    class Batched:
        def __init__(self, B: int):
            self.B = B
            self.kc = [torch.zeros(B, KV, lm.max_seq, HD, device=dev, dtype=dt) for _ in range(lm.n_layers)]
            self.vc = [torch.zeros(B, KV, lm.max_seq, HD, device=dev, dtype=dt) for _ in range(lm.n_layers)]
            self.tok = torch.zeros(B, dtype=torch.long, device=dev)
            self.pos = torch.zeros(B, dtype=torch.long, device=dev)
            self.rows = torch.arange(B, device=dev)
            self.rng = torch.empty(B, lm.vocab, device=dev, dtype=torch.float32)
            self.logits = torch.zeros(B, lm.n_audio, device=dev, dtype=dt)
            self.out = torch.zeros(B, dtype=torch.long, device=dev)

        def forward(self, tok, pos, Lb: int):
            B = self.B
            x = F.embedding(tok, lm.embed).view(B, 1, lm.hidden)
            cos = lm.cos.index_select(0, pos).view(B, 1, 1, HD)
            sin = lm.sin.index_select(0, pos).view(B, 1, 1, HD)
            mask = (lm.ar[:Lb].view(1, Lb) <= pos.view(B, 1)).view(B, 1, 1, Lb)

            def attn(i, q, k, v):
                self.kc[i][self.rows, :, pos] = k[:, :, 0]
                self.vc[i][self.rows, :, pos] = v[:, :, 0]
                kk, vv = self.kc[i][:, :, :Lb], self.vc[i][:, :, :Lb]
                qg = q.reshape(B, KV, G, HD)
                sc = (qg.float().unsqueeze(3) * kk.unsqueeze(2).float()).sum(-1) * lm.scale
                pr = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1)
                o = (pr.unsqueeze(-1) * vv.unsqueeze(2).float()).sum(-2).to(qg.dtype)
                return o.reshape(B, 1, H * HD)

            for i in range(lm.n_layers):
                h = _cohere_ln(x, lm.ln_w[i], lm.eps)
                qkv = F.linear(h, lm.w_qkv[i])
                nq, nk = H * HD, KV * HD
                q = qkv[..., :nq].view(B, 1, H, HD).transpose(1, 2)
                k = qkv[..., nq:nq + nk].view(B, 1, KV, HD).transpose(1, 2)
                v = qkv[..., nq + nk:].view(B, 1, KV, HD).transpose(1, 2)
                if lm.sliding[i]:
                    q = _rope_interleaved(q, cos, sin)
                    k = _rope_interleaved(k, cos, sin)
                o = attn(i, q, k, v)
                a = F.linear(o, lm.w_o[i])
                gu = F.linear(h, lm.w_gu[i])
                g, u = gu[..., :lm.inter], gu[..., lm.inter:]
                x = x + a + F.linear(F.silu(g) * u, lm.w_down[i])
            hn = _cohere_ln(x, lm.norm_w, lm.eps)
            logits = F.linear(hn, lm.w_head).view(B, -1)
            if lm.logit_scale != 1.0:
                logits = logits * lm.logit_scale
            return logits, hn

        def sample(self, logits):
            s = logits / max(lm.temperature, 1e-5)
            cutoff = torch.topk(s, lm.top_k, dim=-1).values[:, -1:]
            s = s.masked_fill(s < cutoff, torch.finfo(s.dtype).min)
            p = torch.softmax(s.float(), dim=-1)
            q = self.rng.exponential_(1).index_select(1, lm.audio_ids)
            return lm.audio_ids.index_select(0, torch.argmax(p / q, dim=-1))

    results, errors, checks = [], [], {}
    fwd_cache = {}

    def compiled(bt: Batched, Lb: int):
        key = (bt.B, Lb)
        if key not in fwd_cache:
            def step(tok, pos):
                logits, hn = bt.forward(tok, pos, Lb)
                return logits, bt.sample(logits)
            fwd_cache[key] = torch.compile(step, fullgraph=True, dynamic=False)
        return fwd_cache[key]

    # ---------------- correctness
    try:
      with torch.inference_mode():
          g = torch.Generator(device=dev).manual_seed(0)
          P0 = 300
          bt1 = Batched(1)
          for i in range(lm.n_layers):  # same random cache contents in FastLM and the batched copy
              kv = torch.randn(1, KV, lm.max_seq, HD, device=dev, generator=g).to(dt) * 0.5
              vv = torch.randn(1, KV, lm.max_seq, HD, device=dev, generator=g).to(dt) * 0.5
              lm.kc[i].copy_(kv); lm.vc[i].copy_(vv); bt1.kc[i].copy_(kv); bt1.vc[i].copy_(vv)
          tok = lm.audio_ids[1234:1235].clone()
          lm.tok_buf.copy_(tok); lm.pos_buf.fill_(P0)
          ref, _ = lm._decode_forward(lm.tok_buf, lm.pos_buf, 512)
          bt1.tok.copy_(tok); bt1.pos.fill_(P0)
          got, _ = bt1.forward(bt1.tok, bt1.pos, 512)
          checks["row0_vs_fastlm_max_abs_diff"] = float((got[0].float() - ref.float()).abs().max())
          checks["row0_vs_fastlm_top1_equal"] = bool(got[0].argmax() == ref.argmax())
          bt4 = Batched(4)
          for i in range(lm.n_layers):
              for b in range(4):
                  bt4.kc[i][b].copy_(bt1.kc[i][0]); bt4.vc[i][b].copy_(bt1.vc[i][0])
          bt4.tok.fill_(int(tok)); bt4.pos.fill_(P0)
          got4, _ = bt4.forward(bt4.tok, bt4.pos, 512)
          checks["identical_rows_max_abs_diff"] = float((got4 - got4[0:1]).abs().max())
          checks["b4_row0_vs_b1_max_abs_diff"] = float((got4[0].float() - got[0].float()).abs().max())
          log("checks", checks)
          del bt1, bt4
          torch.cuda.empty_cache()
    except Exception:
        errors.append({"where": "checks", "err": traceback.format_exc()[-3000:]})

    # ---------------- timing
    stream = torch.cuda.Stream()
    torch.inference_mode().__enter__()
    for B in cfg["batches"]:
        try:
            bt = Batched(B)
            for Lb in cfg["buckets"]:
                # listeners spread over the bucket: positions between Lb/2 and Lb-2
                bt.pos.copy_(torch.linspace(Lb // 2, Lb - 2, B, device=dev).long())
                bt.tok.copy_(lm.audio_ids.index_select(0, torch.arange(B, device=dev) % 2048))
                fn = compiled(bt, Lb)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        fn(bt.tok, bt.pos)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    lo, nt = fn(bt.tok, bt.pos)
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                n = cfg["iters"]
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(n):
                    graph.replay()
                e1.record()
                torch.cuda.synchronize()
                ms = e0.elapsed_time(e1) / n
                r = {"B": B, "Lb": Lb, "step_ms": ms, "tok_per_s_per_listener": 1000.0 / ms,
                     "tok_per_s_total": B * 1000.0 / ms, "listener_realtime_x": (1000.0 / ms) / 100.0,
                     "mem_alloc_gb": torch.cuda.max_memory_allocated() / 1e9}
                results.append(r)
                log(f"B={B:4d} Lb={Lb:5d} step {ms:6.2f} ms -> {1000/ms:6.1f} tok/s per listener ({(1000/ms)/100:4.2f}x real time), "
                    f"{B*1000/ms:8.0f} tok/s total, mem {r['mem_alloc_gb']:.1f} GB")
                del graph
            del bt
            fwd_cache.clear()
            torch.cuda.empty_cache()
        except Exception:
            errors.append({"where": f"B={B}", "err": traceback.format_exc()[-3000:]})
            log(f"B={B} failed")
    env = {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__), "revision": REVISION,
           "tf32_matmul": False, "attn": "f32-reduction, as FastLM final", "sampling": "top-k 30 + exponential, as FastLM"}
    return json.loads(json.dumps({"env": env, "cfg": cfg, "checks": checks, "results": results, "errors": errors,
                                  "wall_s": perf_counter() - t0}, default=str))


@app.local_entrypoint()
def main(smoke: bool = False):
    cfg = {"batches": [1, 2, 4, 8, 16, 32, 64, 128], "buckets": [512, 1024], "iters": 200}
    if smoke:
        cfg = {"batches": [1, 8], "buckets": [512], "iters": 50}
    t = time.time()
    data = bench.remote(cfg)
    data["wall_clock_remote_call_s"] = time.time() - t
    out = ROOT / "results" / ("batch_step_smoke.json" if smoke else "batch_step.json")
    out.write_text(json.dumps(data, indent=1))
    print("checks:", data["checks"], "| errors:", len(data["errors"]))
    for r in data["results"]:
        print(f"B={r['B']:4d} Lb={r['Lb']:5d} step {r['step_ms']:6.2f} ms  per-listener {r['tok_per_s_per_listener']:6.1f} tok/s  "
              f"({r['listener_realtime_x']:.2f}x)  total {r['tok_per_s_total']:8.0f} tok/s")
    for e in data["errors"]:
        print(e["where"], e["err"][-800:])
    print("wrote", out)
