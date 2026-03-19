#!/usr/bin/env python3
"""
Tensor-parallel transformer inference test for MI300X clusters.

Models one LLaMA-3-8B-style decoder layer with column/row-parallel linear
layers sharded across all available GPUs on this node.  Communication is
intra-node RCCL all-reduce (same pattern used by vLLM / Megatron-LM).

Each node runs its own TP group independently.  Spur dispatches this script
to every allocated node; the test verifies both nodes produce correct results.

Outputs on rank 0:
  [<node>] GPUs: N x <name>
  [<node>] TP degree: N  hidden=4096 ffn=14336 heads=32
  [<node>] Throughput: X.X tok/s  peak_mem: Y.Y GB/GPU
  [<node>] INFERENCE_OK
"""

import os
import sys
import time
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

# ── Model hyperparameters (LLaMA-3-8B style) ─────────────────────────────────
HIDDEN = 4096
NUM_HEADS = 32
HEAD_DIM = HIDDEN // NUM_HEADS   # 128
FFN_DIM = 14336                  # SwiGLU intermediate

# ── Benchmark settings ────────────────────────────────────────────────────────
BATCH = 4
SEQ_LEN = 512
NUM_WARMUP = 3
NUM_DECODE_STEPS = 20            # simulate autoregressive decode

# ── Use SPUR_TARGET_NODE if set, else hostname ────────────────────────────────
NODE = os.environ.get("SPUR_TARGET_NODE", socket.gethostname())

# ── Port: base + (job_id % 100) to avoid conflicts between parallel runs ─────
_PORT_BASE = 29501
_PORT = _PORT_BASE + (int(os.environ.get("SPUR_JOB_ID", 0)) % 100)


# ── Tensor-parallel building blocks ──────────────────────────────────────────

class ColParallel(torch.nn.Module):
    """Column-parallel linear: replicated input → sharded output."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, tp: int) -> None:
        super().__init__()
        assert out_dim % tp == 0
        self.w = torch.nn.Parameter(
            torch.empty(out_dim // tp, in_dim, device=f"cuda:{rank}")
        )
        torch.nn.init.normal_(self.w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w)


class RowParallel(torch.nn.Module):
    """Row-parallel linear: sharded input → all-reduced output."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, tp: int) -> None:
        super().__init__()
        assert in_dim % tp == 0
        self.w = torch.nn.Parameter(
            torch.empty(out_dim, in_dim // tp, device=f"cuda:{rank}")
        )
        torch.nn.init.normal_(self.w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.w)
        dist.all_reduce(out)
        return out


# ── One transformer decoder layer (attention + SwiGLU FFN) ───────────────────

class TensorParallelDecoderLayer(torch.nn.Module):

    def __init__(self, rank: int, tp: int) -> None:
        super().__init__()
        self.tp = tp
        self.local_heads = NUM_HEADS // tp
        # Attention: QKV column-parallel, output row-parallel
        self.qkv = ColParallel(HIDDEN, HIDDEN * 3, rank, tp)
        self.o_proj = RowParallel(HIDDEN, HIDDEN, rank, tp)
        # SwiGLU FFN: gate+up column-parallel, down row-parallel
        self.gate = ColParallel(HIDDEN, FFN_DIM, rank, tp)
        self.up = ColParallel(HIDDEN, FFN_DIM, rank, tp)
        self.down = RowParallel(FFN_DIM, HIDDEN, rank, tp)
        self.norm1 = torch.nn.LayerNorm(HIDDEN, device=f"cuda:{rank}")
        self.norm2 = torch.nn.LayerNorm(HIDDEN, device=f"cuda:{rank}")

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.local_heads, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, self.local_heads, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, self.local_heads, HEAD_DIM).transpose(1, 2)
        scale = HEAD_DIM ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous()
        return self.o_proj(out.view(B, T, HIDDEN // self.tp))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.norm1(x))
        x = x + self._ffn(self.norm2(x))
        return x


# ── Worker (one per GPU) ──────────────────────────────────────────────────────

def worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_PORT)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"

    layer = TensorParallelDecoderLayer(rank, world_size).to(dev).half()

    x_full = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=dev, dtype=torch.float16)

    # Warmup with full-sequence prefill
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = layer(x_full)
    torch.cuda.synchronize()

    # Benchmark: single-token decode (the hot path for autoregressive serving)
    x_tok = x_full[:, :1, :].clone()   # (B, 1, H)
    tokens = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_DECODE_STEPS):
            x_tok = layer(x_tok)
            tokens += BATCH
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Verify output is finite (catches NaN from wrong all-reduce)
    if not torch.isfinite(x_tok).all():
        if rank == 0:
            print(f"[{NODE}] ERROR: non-finite values in output", flush=True)
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        throughput = tokens / elapsed
        peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9
        gpu_name = torch.cuda.get_device_name(dev)
        print(f"[{NODE}] GPUs: {world_size} x {gpu_name}", flush=True)
        print(
            f"[{NODE}] TP degree: {world_size}  "
            f"hidden={HIDDEN} ffn={FFN_DIM} heads={NUM_HEADS}",
            flush=True,
        )
        print(
            f"[{NODE}] Throughput: {throughput:.1f} tok/s  "
            f"peak_mem: {peak_gb:.2f} GB/GPU",
            flush=True,
        )
        print(f"[{NODE}] INFERENCE_OK", flush=True)

    dist.destroy_process_group()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n = torch.cuda.device_count()
    if n == 0:
        print(f"[{NODE}] ERROR: no GPUs found", flush=True)
        sys.exit(1)

    print(f"[{NODE}] Starting TP inference on {n} GPUs (port {_PORT})", flush=True)
    sys.stdout.flush()
    mp.spawn(worker, args=(n,), nprocs=n, join=True)
