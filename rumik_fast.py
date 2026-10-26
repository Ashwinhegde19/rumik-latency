"""Low-latency streaming inference for rumik-ai/rumik-oss-1.

Three pieces, all built from the weights of the stock HF modules (no retraining,
no weight changes):

* ``FastLM``: a lean re-implementation of the Cohere2 decoder forward for
  batch-1 streaming. Static KV cache, single-token decode step with on-GPU
  sampling + stop check, logits only for the 16,385 audio ids (16,384 codec
  tokens + ``</audio>``; the stock loop masks every other id anyway), optional
  ``torch.compile`` of the forward, and CUDA-graph capture of whole steps.
  One captured step = forward of the previous token + sampling of the next, so
  the first 8 tokens need prefill + 7 steps.
  Sampling reproduces ``RumikOSSForCausalLM._constrained_sample`` exactly,
  including its random-number consumption (``torch.multinomial`` with one
  sample draws ``Exp(1)`` noise over the full vocabulary and takes
  ``argmax(p / q)``), so a fixed seed gives the same tokens as the stock loop
  up to bf16 kernel differences.

* ``StreamingMimi``: frame-by-frame Mimi decoder with explicit streaming state
  (causal-conv history, transposed-conv overlap, decoder-transformer KV cache).
  HF Mimi is fully causal (and its decoder transformer uses plain causal
  attention), so streaming output equals ``MimiModel.decode`` of the whole clip
  up to float rounding. One frame (8 codes) in, 1920 samples out; the
  per-frame step is CUDA-graph captured.

* ``StreamingTTS``: the request driver. The LM runs ahead on the default
  stream in chunks of 8 tokens (one codec frame); the host checks each chunk
  once (stop / ``</audio>``), assembles frames and launches the Mimi step on a
  second stream, so codec decode overlaps LM decode.
"""
from __future__ import annotations

from time import perf_counter

import torch
import torch._dynamo
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers matching the HF Cohere2 numerics
# ---------------------------------------------------------------------------
def _cohere_ln(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    xf = (xf - mean) * torch.rsqrt(var + eps)
    return (w.float() * xf).to(x.dtype)


def _rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # GPT-J style rotation, computed in fp32 like Cohere2's apply_rotary_pos_emb
    xf = x.float()
    rot = torch.stack((-xf[..., 1::2], xf[..., ::2]), dim=-1).flatten(-2)
    return (xf * cos + rot * sin).to(x.dtype)


# The model's own typical start: three near-silent frames (codebook-0 codes 1049, 127, 1880), taken verbatim from a
# real generation (quality-audit run hinglish_medium_seed1009). Passed as StreamingTTS.run(..., start_prefix=START_PREFIX)
# they are processed in prefill, so the first generated frame is speech: first audible audio 110 -> 34 ms p50 on an
# H100 with output quality unchanged (results/prefix_fix/report.md). Exactly three frames: four reads as a pause and
# the model stays silent longer. Only valid for this model revision and codec.
START_PREFIX = [269400, 272769, 263994, 264915, 264060, 277093, 262478, 267047,
                262024, 272769, 264666, 263331, 264860, 265453, 268822, 274335,
                276048, 261737, 273482, 263331, 271692, 264837, 267614, 274335]


class FastLM:
    @torch.inference_mode()
    def __init__(
        self,
        model,
        max_seq: int = 2304,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_k: int = 30,
        min_new_tokens: int = 8,
        use_compile: bool = False,
        use_graphs: bool = True,
        prefill_buckets: tuple[int, ...] = (32, 64, 96, 128, 192, 256),
        kv_buckets: tuple[int, ...] | None = None,
        attn: str = "sdpa",
        compile_mode: str | None = None,
        compile_prefill: bool = False,
    ):
        cfg = model.config
        dev = next(model.parameters()).device
        dt = torch.bfloat16
        self.dev, self.dt = dev, dt
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.hidden = cfg.hidden_size
        self.eps = cfg.layer_norm_eps
        # Cohere2 applies RoPE only on sliding-window layers; global layers are NoPE.
        self.sliding = [t == "sliding_attention" for t in cfg.layer_types]
        # The static cache never exceeds the sliding window, so sliding layers see
        # exactly the same keys as full layers.
        self.window = int(cfg.sliding_window)
        assert max_seq <= self.window, "static cache longer than the sliding window is not handled"
        self.max_seq = max_seq
        self.max_new = max_new_tokens
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.min_new = int(min_new_tokens)
        self.use_graphs = use_graphs
        self.prefill_buckets = tuple(b for b in prefill_buckets if b <= max_seq)
        # decode attention only reads the first Lb cache slots; one captured step per bucket,
        # chosen on the host from the (host-known) position. Masking keeps it exact.
        self.kv_buckets = tuple(sorted(set(kv_buckets or ()) | {max_seq}))
        assert self.kv_buckets[-1] == max_seq
        assert attn in ("sdpa", "manual", "f32")
        self.attn_impl = attn
        self.scale = self.hd ** -0.5
        self.cur_pos = 0
        self.vocab = int(cfg.vocab_size)
        self.first, self.last = int(cfg.first_unit_id), int(cfg.last_unit_id)
        self.end_id = int(cfg.audio_end_token_id)
        self.Q = int(cfg.num_quantizers)

        m = model.model
        self.embed = m.embed_tokens.weight.detach()
        L = m.layers
        self.ln_w = [l.input_layernorm.weight.detach() for l in L]
        self.w_qkv = [torch.cat([l.self_attn.q_proj.weight, l.self_attn.k_proj.weight,
                                 l.self_attn.v_proj.weight], 0).contiguous() for l in L]
        self.w_o = [l.self_attn.o_proj.weight.detach() for l in L]
        self.w_gu = [torch.cat([l.mlp.gate_proj.weight, l.mlp.up_proj.weight], 0).contiguous() for l in L]
        self.w_down = [l.mlp.down_proj.weight.detach() for l in L]
        self.norm_w = m.norm.weight.detach()
        self.inter = L[0].mlp.intermediate_size
        # audio ids (ascending; </audio> is the largest id, so order matches the full vocab)
        self.audio_ids = model.audio_token_ids(device=dev)
        assert bool((self.audio_ids[1:] > self.audio_ids[:-1]).all()) and int(self.audio_ids[-1]) == self.end_id
        self.n_audio = self.audio_ids.numel()
        self.w_head = model.lm_head.weight.detach().index_select(0, self.audio_ids).contiguous()
        self.logit_scale = float(model.logit_scale)
        self.stop_head = model.stop_predictor

        # rope tables (bf16, as Cohere2RotaryEmbedding returns them)
        rot = m.rotary_emb
        pos = torch.arange(max_seq, device=dev, dtype=torch.float32)
        freqs = torch.outer(pos, rot.inv_freq.float().to(dev))
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.cos = (emb.cos() * rot.attention_scaling).to(dt)
        self.sin = (emb.sin() * rot.attention_scaling).to(dt)

        # static state
        self.kc = [torch.zeros(1, self.n_kv, max_seq, self.hd, device=dev, dtype=dt) for _ in L]
        self.vc = [torch.zeros(1, self.n_kv, max_seq, self.hd, device=dev, dtype=dt) for _ in L]
        self.ar = torch.arange(max_seq, device=dev)
        self.tok_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.step_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.logits_buf = torch.zeros(self.n_audio, device=dev, dtype=dt)
        self.hidden_buf = torch.zeros(1, 1, self.hidden, device=dev, dtype=dt)
        self.out_rec = torch.zeros(max_new_tokens + 16, dtype=torch.long, device=dev)
        self.rng_buf = torch.empty(1, self.vocab, device=dev, dtype=torch.float32)
        self.end_col = torch.zeros(self.n_audio, dtype=torch.bool, device=dev)
        self.end_col[-1] = True
        self.neg = torch.tensor(torch.finfo(dt).min, device=dev, dtype=dt)
        self.min_new_t = torch.tensor(self.min_new, device=dev)
        self.pre_ids = torch.zeros(max(self.prefill_buckets), dtype=torch.long, device=dev)
        self.pre_last = torch.zeros(1, dtype=torch.long, device=dev)
        self.pin_ids = torch.zeros(max(self.prefill_buckets), dtype=torch.long).pin_memory()
        self.pin_rec = torch.zeros(max_new_tokens + 16, dtype=torch.long).pin_memory()

        self._fwd_decode = self._decode_forward
        self._fwd_prefill = self._prefill_forward
        if use_compile:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._fwd_decode = torch.compile(self._decode_forward, fullgraph=True, dynamic=False, mode=compile_mode)
            if compile_prefill:
                self._fwd_prefill = torch.compile(self._prefill_forward, fullgraph=True, dynamic=False, mode=compile_mode)
        self.graphs: dict = {}
        if use_graphs:
            self._capture()

    # -------------------------------------------------------------- forward
    def _layer(self, i: int, x, cos, sin, attn):
        h = _cohere_ln(x, self.ln_w[i], self.eps)
        T = x.shape[1]
        qkv = F.linear(h, self.w_qkv[i])
        nq, nk = self.n_heads * self.hd, self.n_kv * self.hd
        q = qkv[..., :nq].view(1, T, self.n_heads, self.hd).transpose(1, 2)
        k = qkv[..., nq:nq + nk].view(1, T, self.n_kv, self.hd).transpose(1, 2)
        v = qkv[..., nq + nk:].view(1, T, self.n_kv, self.hd).transpose(1, 2)
        if self.sliding[i]:
            q = _rope_interleaved(q, cos, sin)
            k = _rope_interleaved(k, cos, sin)
        o = attn(i, q, k, v)  # [1, T, H*D]
        a = F.linear(o, self.w_o[i])
        gu = F.linear(h, self.w_gu[i])
        g, u = gu[..., :self.inter], gu[..., self.inter:]
        mm = F.linear(F.silu(g) * u, self.w_down[i])
        return x + a + mm

    def _head(self, x):
        hn = _cohere_ln(x, self.norm_w, self.eps)
        logits = F.linear(hn, self.w_head)
        if self.logit_scale != 1.0:
            logits = logits * self.logit_scale
        return logits.view(-1), hn

    def _decode_forward(self, tok, pos, Lb: int):
        """One token at position `pos` (both 1-element device tensors); attends to cache[:Lb] (pos < Lb)."""
        x = F.embedding(tok, self.embed).view(1, 1, self.hidden)
        cos = self.cos.index_select(0, pos).view(1, 1, 1, self.hd)
        sin = self.sin.index_select(0, pos).view(1, 1, 1, self.hd)
        mask = (self.ar[:Lb] <= pos).view(1, 1, 1, Lb)
        G = self.n_heads // self.n_kv

        def attn(i, q, k, v):
            self.kc[i][:, :, pos] = k
            self.vc[i][:, :, pos] = v
            kk, vv = self.kc[i][:, :, :Lb], self.vc[i][:, :, :Lb]
            # GQA without repeat_kv: the G query heads of one KV head become G query rows
            qg = q.reshape(1, self.n_kv, G, self.hd)
            if self.attn_impl == "sdpa":
                o = F.scaled_dot_product_attention(qg, kk, vv, attn_mask=mask)
            elif self.attn_impl == "manual":  # as HF eager attention: bf16 matmuls, fp32 softmax
                sc = torch.matmul(qg, kk.transpose(-1, -2)) * self.scale
                sc = sc.float().masked_fill(~mask, float("-inf"))
                o = torch.matmul(torch.softmax(sc, -1).to(qg.dtype), vv)
            else:  # fp32 scores / softmax / accumulation (as SDPA kernels); written as reductions so
                # inductor emits two fused kernels without materialising fp32 copies of the cache
                sc = (qg.float().unsqueeze(3) * kk.unsqueeze(2).float()).sum(-1) * self.scale
                pr = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1)
                o = (pr.unsqueeze(-1) * vv.unsqueeze(2).float()).sum(-2).to(qg.dtype)
            return o.reshape(1, 1, self.n_heads * self.hd)

        for i in range(self.n_layers):
            x = self._layer(i, x, cos, sin, attn)
        return self._head(x)

    def _prefill_forward(self, ids, last):
        """`ids` [P] (prompt + right padding), returns logits/hidden at index `last`."""
        P = ids.shape[0]
        x = F.embedding(ids, self.embed).view(1, P, self.hidden)
        cos = self.cos[:P].view(1, 1, P, self.hd)
        sin = self.sin[:P].view(1, 1, P, self.hd)
        G = self.n_heads // self.n_kv

        def attn(i, q, k, v):
            self.kc[i][:, :, :P] = k
            self.vc[i][:, :, :P] = v
            kk = k.repeat_interleave(G, dim=1)
            vv = v.repeat_interleave(G, dim=1)
            o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            return o.transpose(1, 2).reshape(1, P, self.n_heads * self.hd)

        for i in range(self.n_layers):
            x = self._layer(i, x, cos, sin, attn)
        return self._head(x.index_select(1, last))

    # -------------------------------------------------------------- steps
    def _sample(self):
        """== one iteration of RumikOSSForCausalLM.generate_audio, restricted to audio ids.

        Reads logits/hidden of the last forward, writes the sampled id (or -1 if the
        stop head fired) to out_rec[step] and to tok_buf, increments step.
        """
        s = torch.where(self.end_col & (self.step_buf < self.min_new_t), self.neg, self.logits_buf)
        s = s / max(self.temperature, 1e-5)
        cutoff = torch.topk(s, self.top_k).values[-1:]
        s = s.masked_fill(s < cutoff, torch.finfo(s.dtype).min)
        p = torch.softmax(s.float(), dim=-1)
        q = self.rng_buf.exponential_(1)  # same draw as torch.multinomial(p_full, 1)
        idx = torch.argmax(p / q.view(-1).index_select(0, self.audio_ids)).view(1)
        tok = self.audio_ids.index_select(0, idx)
        stop = torch.sigmoid(self.stop_head(self.hidden_buf).view(1)) > 0.5
        stop = stop & (self.step_buf >= self.min_new_t)
        self.out_rec.index_copy_(0, self.step_buf, torch.where(stop, -1, tok))
        self.tok_buf.copy_(tok)
        self.step_buf.add_(1)

    def _prefill_step(self, P: int):
        logits, hn = self._fwd_prefill(self.pre_ids[:P], self.pre_last)
        self.logits_buf.copy_(logits)
        self.hidden_buf.copy_(hn)
        self._sample()

    def _fwd_step(self, Lb: int):
        logits, hn = self._fwd_decode(self.tok_buf, self.pos_buf, Lb)
        self.logits_buf.copy_(logits)
        self.hidden_buf.copy_(hn)
        self.pos_buf.add_(1)

    def _gen_step(self, Lb: int):
        self._fwd_step(Lb)
        self._sample()

    def _bucket(self) -> int:
        return next(b for b in self.kv_buckets if b > self.cur_pos)

    def _capture(self):
        pool = torch.cuda.graph_pool_handle()
        # Dedicated capture stream: cuBLAS workspaces are keyed by stream, and graphs captured on
        # PyTorch's shared default capture stream would bake in the same workspace as the Mimi
        # graph, which replays concurrently on another stream.
        s = self.cap_stream = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        rng_state = torch.cuda.get_rng_state()
        with torch.cuda.stream(s):
            for Lb in self.kv_buckets:
                self.pos_buf.fill_(1)
                self.step_buf.zero_()
                for _ in range(3):
                    self._gen_step(Lb)
                    self._fwd_step(Lb)
            for P in self.prefill_buckets:
                self._prefill_step(P)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for Lb in self.kv_buckets:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=s):
                self._gen_step(Lb)
            self.graphs[("step", Lb)] = g
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=s):
                self._fwd_step(Lb)
            self.graphs[("fwd", Lb)] = g
        for P in self.prefill_buckets:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=s):
                self._prefill_step(P)
            self.graphs[("prefill", P)] = g
        torch.cuda.synchronize()
        torch.cuda.set_rng_state(rng_state)

    # -------------------------------------------------------------- public API
    @torch.inference_mode()
    def prefill(self, ids: list[int]):
        """Prompt forward + first sample (token 0). Async, no host sync."""
        T = len(ids)
        assert T + self.max_new + 1 <= self.max_seq, "prompt too long for the static cache"
        self.step_buf.zero_()
        if self.use_graphs:
            P = next(b for b in self.prefill_buckets if b >= T)
            self.pin_ids[:T] = torch.tensor(ids)
            self.pin_ids[T:P] = 0
            self.pre_ids[:P].copy_(self.pin_ids[:P], non_blocking=True)
            self.pre_last.fill_(T - 1)
            self.graphs[("prefill", P)].replay()
        else:
            self.pre_ids[:T].copy_(torch.tensor(ids), non_blocking=True)
            self.pre_last.fill_(T - 1)
            self._prefill_step(T)
        self.pos_buf.fill_(T)
        self.n_prompt = T
        self.cur_pos = T

    @torch.inference_mode()
    def step(self, n: int = 1):
        """n x (forward of the last sampled token + sample the next). Async."""
        for _ in range(n):
            Lb = self._bucket()
            if self.use_graphs:
                self.graphs[("step", Lb)].replay()
            else:
                self._gen_step(Lb)
            self.cur_pos += 1

    @torch.inference_mode()
    def forward_token(self, tok: int):
        """Teacher forcing: feed `tok` at the current position (logits_buf/hidden_buf updated)."""
        self.tok_buf.fill_(tok)
        Lb = self._bucket()
        if self.use_graphs:
            self.graphs[("fwd", Lb)].replay()
        else:
            self._fwd_step(Lb)
        self.cur_pos += 1

    def copy_out(self, a: int, b: int):
        """Enqueue device->pinned copy of records [a, b)."""
        self.pin_rec[a:b].copy_(self.out_rec[a:b], non_blocking=True)

    @torch.inference_mode()
    def stop_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.stop_head(self.hidden_buf).float()).view(-1)[0]


class FrameAssembler:
    """Incremental version of RumikOSSForCausalLM.audio_tokens_to_codes (without </audio>)."""

    def __init__(self, first: int, last: int, Q: int):
        self.first, self.last, self.Q = first, last, Q
        self.frame: list[int] = []

    def push(self, tid: int):
        """Returns a complete frame (list of Q codes) or None."""
        if not self.first <= tid <= self.last:
            self.frame = []
            return None
        code, q = divmod(tid - self.first, self.Q)
        if q == len(self.frame):
            self.frame.append(code)
            if len(self.frame) == self.Q:
                f, self.frame = self.frame, []
                return f
        else:
            self.frame = [code] if q == 0 else []
        return None


# ---------------------------------------------------------------------------
# streaming Mimi decoder
# ---------------------------------------------------------------------------
class StreamingMimi:
    @torch.inference_mode()
    def __init__(self, mimi, n_q: int = 8, max_frames: int = 512, use_graphs: bool = True):
        self.mimi = mimi
        p = next(mimi.parameters())
        self.dev, self.dt = p.device, p.dtype
        cfg = mimi.config
        self.n_q = n_q
        self.spf = int(cfg.sampling_rate / cfg.frame_rate)  # 1920
        rvq = mimi.quantizer
        self.sem = rvq.semantic_residual_vector_quantizer
        self.aco = rvq.acoustic_residual_vector_quantizer
        self.n_sem = rvq.num_semantic_quantizers
        for l in list(self.sem.layers) + list(self.aco.layers):
            l.codebook.embed  # materialise the cached table before capture
        self.up = mimi.upsample  # MimiConvTranspose1d (depthwise, stride 2)
        self.tlayers = mimi.decoder_transformer.layers
        a0 = self.tlayers[0].self_attn
        assert a0.num_heads == a0.num_key_value_heads
        self.nh, self.thd = a0.num_heads, a0.head_dim
        self.max_frames = max_frames
        self.t_max = 2 * max_frames
        rot = a0.rotary_emb
        pos = torch.arange(self.t_max, device=self.dev, dtype=torch.float32)
        freqs = torch.outer(pos, rot.inv_freq.float().to(self.dev))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = (emb.cos() * rot.attention_scaling).to(self.dt)
        self.sin = (emb.sin() * rot.attention_scaling).to(self.dt)
        self.kc = [torch.zeros(1, self.nh, self.t_max, self.thd, device=self.dev, dtype=self.dt) for _ in self.tlayers]
        self.vc = [torch.zeros_like(k) for k in self.kc]
        self.tar = torch.arange(self.t_max, device=self.dev)
        self.tpos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.t_ar = torch.arange(2, device=self.dev)
        self.ups = 2  # transformer steps per codec frame

        # SEANet decoder plan with per-layer streaming state
        from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d, MimiResnetBlock
        self.plan = []
        self.states: list[torch.Tensor] = []
        T = self.ups
        C = cfg.hidden_size

        def conv_state(mod, Cin):
            c = mod.conv
            assert mod.causal and c.stride[0] == 1 and mod.pad_mode == "constant" and c.padding[0] == 0
            pad = (c.kernel_size[0] - 1) * c.dilation[0]
            st = torch.zeros(1, Cin, pad, device=self.dev, dtype=self.dt) if pad > 0 else None
            if st is not None:
                self.states.append(st)
            return st

        def convt_state(mod, Cout):
            c = mod.conv
            k, s = c.kernel_size[0], c.stride[0]
            assert mod.padding_left == 0 and mod.padding_right == k - s
            st = torch.zeros(1, Cout, k - s, device=self.dev, dtype=self.dt)
            self.states.append(st)
            return st

        self.up_state = convt_state(self.up, C)
        for layer in mimi.decoder.layers:
            if isinstance(layer, MimiConv1d):
                self.plan.append(("conv", layer.conv, conv_state(layer, layer.conv.in_channels)))
            elif isinstance(layer, MimiConvTranspose1d):
                self.plan.append(("convt", layer.conv, convt_state(layer, layer.conv.out_channels)))
                T = T * layer.conv.stride[0]
            elif isinstance(layer, MimiResnetBlock):
                assert not isinstance(layer.shortcut, MimiConv1d)
                sub = []
                for b in layer.block:
                    if isinstance(b, MimiConv1d):
                        sub.append(("conv", b.conv, conv_state(b, b.conv.in_channels)))
                    else:
                        sub.append(("act", b, None))
                self.plan.append(("res", sub, None))
            else:
                self.plan.append(("act", layer, None))
        assert T == self.spf, (T, self.spf)

        self.codes_buf = torch.zeros(1, n_q, 1, dtype=torch.long, device=self.dev)
        self.pcm_buf = torch.zeros(self.spf, device=self.dev, dtype=self.dt)
        self.use_graphs = use_graphs
        self.graph = None
        self.reset_graph = None
        if use_graphs:
            self._capture()

    # ----------------------------------------------------------------
    @staticmethod
    def _conv(x, conv, st):
        if st is not None:
            xin = torch.cat((st, x), dim=-1)
            st.copy_(xin[..., -st.shape[-1]:])
        else:
            xin = x
        return F.conv1d(xin, conv.weight, conv.bias, stride=1, dilation=conv.dilation, groups=conv.groups)

    @staticmethod
    def _convt(x, conv, st):
        y = F.conv_transpose1d(x, conv.weight, None, stride=conv.stride, groups=conv.groups)
        n = x.shape[-1] * conv.stride[0]
        out = y[..., :n].clone()
        out[..., :st.shape[-1]] += st
        st.copy_(y[..., n:])
        if conv.bias is not None:
            out = out + conv.bias.view(1, -1, 1)
        return out

    def _rvq(self, codes):
        def dec(rvq, cs):
            out = None
            for i in range(cs.shape[1]):
                e = F.embedding(cs[:, i], rvq.layers[i].codebook.embed).permute(0, 2, 1)
                out = e if out is None else out + e
            return F.conv1d(out, rvq.output_proj.weight) if rvq.output_proj is not None else out
        x = dec(self.sem, codes[:, :self.n_sem])
        return x + dec(self.aco, codes[:, self.n_sem:])

    def _transformer(self, x):  # x [1, T, C]
        T = x.shape[1]
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
            q = at.q_proj(h).view(1, T, self.nh, self.thd).transpose(1, 2)
            k = at.k_proj(h).view(1, T, self.nh, self.thd).transpose(1, 2)
            v = at.v_proj(h).view(1, T, self.nh, self.thd).transpose(1, 2)
            q, k = rope(q), rope(k)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            o = F.scaled_dot_product_attention(q, self.kc[i], self.vc[i], attn_mask=mask)
            o = at.o_proj(o.transpose(1, 2).reshape(1, T, -1))
            x = x + layer.self_attn_layer_scale(o)
            h = layer.post_attention_layernorm(x)
            x = x + layer.mlp_layer_scale(layer.mlp(h))
        self.tpos.add_(T)
        return x

    def _frame(self):
        x = self._rvq(self.codes_buf)                    # [1, C, 1]
        x = self._convt(x, self.up.conv, self.up_state)  # [1, C, 2]
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
        self.pcm_buf.copy_(x.view(-1))

    def _reset(self):
        for st in self.states:
            st.zero_()
        self.tpos.zero_()

    def _capture(self):
        s = self.cap_stream = torch.cuda.Stream()  # own capture stream -> own cuBLAS workspace
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._frame()
            self._reset()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self._frame()
        self.reset_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.reset_graph, stream=s):
            self._reset()
        torch.cuda.synchronize()

    # ----------------------------------------------------------------
    @torch.inference_mode()
    def reset(self):
        if self.use_graphs:
            self.reset_graph.replay()
        else:
            self._reset()

    @torch.inference_mode()
    def decode_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [1, n_q, 1] long (device or pinned host). Returns pcm_buf (reused next call)."""
        self.codes_buf.copy_(codes, non_blocking=True)
        if self.use_graphs:
            self.graph.replay()
        else:
            self._frame()
        return self.pcm_buf

    @torch.inference_mode()
    def decode_all(self, codes: torch.Tensor) -> torch.Tensor:
        """Frame-by-frame decode of [1, n_q, T] codes from a fresh state (for checks)."""
        self.reset()
        out = []
        for t in range(codes.shape[-1]):
            out.append(self.decode_frame(codes[..., t:t + 1]).clone())
        return torch.cat(out)


# ---------------------------------------------------------------------------
# request driver
# ---------------------------------------------------------------------------
class StreamingTTS:
    """Streaming synthesis: LM runs ahead in 8-token chunks, Mimi decodes each frame on a side stream.

    mimi_mode:
      "stream"    StreamingMimi for every frame (the optimized path)
      "hf_first"  HF MimiModel.decode of the first frame only (as the baseline loop), no further decode
    """

    def __init__(self, lm: FastLM, tok, smimi: StreamingMimi | None = None, hf_mimi=None,
                 lookahead: int = 2, silence_dbfs: float = -40.0):
        self.lm, self.tok, self.smimi, self.hf_mimi = lm, tok, smimi, hf_mimi
        self.Q = lm.Q
        self.lookahead = lookahead
        self.silence_dbfs = silence_dbfs
        self.mstream = torch.cuda.Stream()
        self.spf = 1920
        mf = lm.max_new // lm.Q + 2 + 8  # + room for start-prefix frames
        self.pin_codes = torch.zeros(mf, 1, lm.Q, 1, dtype=torch.long).pin_memory()
        self.pin_pcm = torch.zeros(mf, self.spf, dtype=torch.float32).pin_memory()

    @torch.inference_mode()
    def run(self, prompt: str, seed: int, mimi_mode: str = "stream", keep_audio: bool = False,
            start_prefix: list[int] | None = None) -> dict:
        """start_prefix: audio token ids (whole frames) placed after the prompt in prefill, e.g. START_PREFIX.
        They are decoded to warm the Mimi state but are not part of the output; `frames`/`tokens` exclude them."""
        lm, Q = self.lm, self.Q
        pre = list(start_prefix or [])
        n_pre = len(pre) // Q
        assert len(pre) == n_pre * Q, "start_prefix must be whole frames (Q token ids per frame)"
        torch.cuda.synchronize()
        t0 = perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        ids = self.tok(prompt).input_ids + pre
        torch.manual_seed(seed)
        if mimi_mode == "stream":
            with torch.cuda.stream(self.mstream):
                self.smimi.reset()
                for j in range(n_pre):  # prefix frames warm the decoder state; their audio is never output
                    self.pin_codes[j].view(-1).copy_(torch.tensor([(t - lm.first) // Q for t in pre[j * Q:(j + 1) * Q]]))
                    self.pin_pcm[j].copy_(self.smimi.decode_frame(self.pin_codes[j]), non_blocking=True)
        lm.prefill(ids)
        ev_pre = torch.cuda.Event(enable_timing=True)
        ev_pre.record()
        n_sampled = 1
        chunk_ev: list = []
        max_new = lm.max_new

        def enqueue_chunk(c):
            nonlocal n_sampled
            a, b = Q * c, min(Q * (c + 1), max_new)
            lm.step(b - n_sampled)
            n_sampled = b
            lm.copy_out(a, b)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            chunk_ev.append(e)

        n_chunks_max = (max_new + Q - 1) // Q
        next_chunk = 0
        while next_chunk <= self.lookahead and next_chunk < n_chunks_max:
            enqueue_chunk(next_chunk)
            next_chunk += 1

        asm = FrameAssembler(lm.first, lm.last, Q)
        tokens: list[int] = []
        stopped_by = "max_new_tokens"
        frame_ev: list = []       # (start, end) events on the mimi stream per decoded frame
        frame_chunk: list[int] = []
        t_first_audio = None
        first_chunk_decode = None
        n_frames = n_pre
        c = 0
        done = False
        while not done and c < len(chunk_ev):
            chunk_ev[c].synchronize()
            recs = lm.pin_rec[Q * c: min(Q * (c + 1), max_new)].tolist()
            if next_chunk < n_chunks_max:
                enqueue_chunk(next_chunk)
                next_chunk += 1
            new_frames = []
            for r in recs:
                if r == -1:
                    stopped_by, done = "stop_predictor", True
                    break
                tokens.append(r)
                if r == lm.end_id:
                    stopped_by, done = "audio_end_token", True
                    break
                f = asm.push(r)
                if f is not None:
                    new_frames.append(f)
            for f in new_frames:
                k = n_frames
                if mimi_mode == "stream":
                    self.pin_codes[k].view(-1).copy_(torch.tensor(f))
                    with torch.cuda.stream(self.mstream):
                        es, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        es.record()
                        pcm = self.smimi.decode_frame(self.pin_codes[k])
                        ee.record()
                        self.pin_pcm[k].copy_(pcm, non_blocking=True)
                    frame_ev.append((es, ee))
                    frame_chunk.append(c)
                    if k == n_pre:  # first generated frame
                        ee.synchronize()
                        t_first_audio = perf_counter() - t0
                        first_chunk_decode = es.elapsed_time(ee) / 1e3
                elif mimi_mode == "hf_first" and k == n_pre:
                    with torch.cuda.stream(self.mstream):
                        td = perf_counter()
                        codes = torch.tensor(f, device=lm.dev).view(1, Q, 1)
                        self.hf_mimi.decode(codes).audio_values
                        torch.cuda.current_stream().synchronize()
                        t_first_audio = perf_counter() - t0
                        first_chunk_decode = perf_counter() - td
                n_frames += 1
            c += 1
        t_last_tokens = perf_counter() - t0
        torch.cuda.synchronize()  # includes run-ahead LM steps beyond the stop and the last Mimi frame
        t_end = perf_counter() - t0

        ms = lambda e: ev0.elapsed_time(e) / 1e3  # noqa: E731  (GPU timeline, s since request start)
        chunk_t = [ms(e) for e in chunk_ev[:c]]
        res = {
            "seed": seed,
            "prompt_tokens": len(ids) - len(pre),
            "prefix_frames": n_pre,
            "audio_tokens": len([t for t in tokens if t != lm.end_id]),
            "frames": n_frames - n_pre,
            "audio_seconds": (n_frames - n_pre) * 0.08,
            "stopped_by": stopped_by,
            "prefill_s": ms(ev_pre),
            "first_frame_tokens_s": chunk_t[0] if chunk_t else None,
            "ttfa_s": t_first_audio,
            "first_chunk_decode_s": first_chunk_decode,
            "gen_total_s": t_last_tokens,
            "total_s": t_end,
            "tokens": tokens,
        }
        if len(chunk_t) > 3:
            # steady-state LM rate: complete chunks after the first (GPU timeline)
            res["decode_tok_per_s"] = Q * (len(chunk_t) - 2) / (chunk_t[-2] - chunk_t[0])
            res["per_token_ms"] = 1e3 / res["decode_tok_per_s"]
        if mimi_mode == "stream" and frame_ev:
            ready = [ms(ee) for _, ee in frame_ev]
            gpu_ms = [es.elapsed_time(ee) for es, ee in frame_ev]
            pcm = self.pin_pcm[n_pre:n_frames]
            dbfs = (20 * torch.log10(pcm.pow(2).mean(-1).sqrt().clamp_min(1e-9))).tolist()
            k = next((i for i, d in enumerate(dbfs) if d > self.silence_dbfs), None)
            # playback starts when frame 0 is ready; frame i is due at ready[0] + 0.08 i
            lateness = [ready[i] - (ready[0] + 0.08 * i) for i in range(n_frames - n_pre)]
            res.update({
                "frame_ready_s": ready,
                "frame_dbfs": dbfs,
                "first_nonsilent_frame": k,
                "ttfa_nonsilent_s": ready[k] if k is not None else None,
                "mimi_frame_ms_p50": float(sorted(gpu_ms)[len(gpu_ms) // 2]),
                "mimi_frame_ms_max": float(max(gpu_ms)),
                "max_lateness_s": max(lateness),
                "stall_free": max(lateness) <= 1e-3,
                "rtfx": res["audio_seconds"] / t_end,
                "rtfx_last_audio": res["audio_seconds"] / ready[-1],
            })
            if keep_audio:
                res["_pcm"] = pcm.reshape(-1).clone()
        return res
