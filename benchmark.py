# !/usr/bin/env python
# How to run: python benchmark.py --device cuda --dtype bf16 --trials 1000
# To fix  No module named 'torch' error:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

"""
Machine-learning-focused GPU micro-benchmark + one-number score.
▪ Tensor Core GEMM & Conv2D (bf16 / fp16)
▪ FP32 GEMM
▪ DRAM & cache-window bandwidth
"""

import argparse, math, time, contextlib, numpy as np, torch
from torch import cuda, mps
try:                               # Intel discrete GPUs
    import intel_extension_for_pytorch as ipex
    from torch import xpu
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# ----------------------------- helpers ------------------------------------- #
# --------------------------------------------------------------------------- #
def synchronize(dev):
    if dev.type == "cuda":   cuda.synchronize()
    elif dev.type == "mps":  mps.synchronize()
    elif dev.type == "xpu":  xpu.synchronize()

@contextlib.contextmanager
def timed(dev):
    """high-res timer wrapped with device sync"""
    synchronize(dev)
    t0 = time.perf_counter()
    yield
    synchronize(dev)
    t1 = time.perf_counter()
    yield_time[0] = t1 - t0

def median(timings):                 # robust to occasional spikes
    return float(np.median(np.array(timings)))

def tflops(n, secs):                 # 2·N³ / t / 1e12
    return 2 * n**3 / secs / 1e12

# --------------------------------------------------------------------------- #
# ----------------------------- benchmarks ---------------------------------- #
# --------------------------------------------------------------------------- #
def gemm_tc(dev, dtype, size, trials):
    """Tensor-Core (bf16/fp16) GEMM throughput"""
    a = torch.randn(size, size, device=dev, dtype=dtype) * 0.1
    times = []
    for _ in range(trials):
        with torch.autocast('cuda', dtype=dtype):
            with timed(dev):
                torch.matmul(a, a)
        times.append(yield_time[0])
    return tflops(size, median(times))

def gemm_fp32(dev, size, trials):
    """FP32 GEMM"""
    a = torch.randn(size, size, device=dev, dtype=torch.float32) * 0.1
    times = []
    for _ in range(trials):
        with timed(dev):
            torch.matmul(a, a)
        times.append(yield_time[0])
    return tflops(size, median(times))

def conv2d_tc(dev, dtype, trials):
    """Tensor-Core Conv2D: 224×224, in=3→out=64, kernel 7×7"""
    input  = torch.randn(  1,   3, 224, 224, device=dev, dtype=dtype)
    weight = torch.randn( 64,   3,   7,   7, device=dev, dtype=dtype)
    times  = []
    for _ in range(trials):
        with torch.autocast('cuda', dtype=dtype):
            with timed(dev):
                torch.nn.functional.conv2d(input, weight, stride=2, padding=3)
        times.append(yield_time[0])
    # conv FLOPs ≈ 2 * out_H * out_W * in_C * kH * kW * out_C
    out_h = out_w = 112
    flops = 2 * out_h * out_w * 3 * 7 * 7 * 64
    return flops / median(times) / 1e12

def bandwidth(dev, trials):
    """peak DRAM + peak L2/L1 ('cache window') bandwidth"""
    sizes = 2 ** np.arange(20, 28, 0.5)   # 1 MB .. 256 MB
    peak_dram = peak_cache = 0.0

    for sz in sizes:
        sz = int(sz)
        a = torch.empty(sz, device=dev, dtype=torch.float32)
        b = torch.empty(sz, device=dev, dtype=torch.float32)
        times = []
        for _ in range(trials):
            with timed(dev):
                a.copy_(b)
            times.append(yield_time[0])

        elap = median(times)
        if elap == 0: elap = 1e-9       # guard µs-quantisation
        bytes_copied = 2 * a.element_size() * a.nelement()
        gbps = bytes_copied / elap / 1e9

        if sz <= 16 * 1024 * 1024:      # ≤16 MB ≈ fits L2 on most GPUs
            peak_cache = max(peak_cache, gbps)
        peak_dram  = max(peak_dram, gbps)

    return peak_dram, peak_cache

# --------------------------------------------------------------------------- #
# ----------------------------- main driver --------------------------------- #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype",  default="bf16",
                    help="bf16 | fp16 (needs tensor-core capable device)")
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    dev   = torch.device(args.device)
    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev.index or 0)
    else:
        gpu_name = str(dev)
    print(f"\n▶  Device : {gpu_name}")

    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16)[args.dtype]
    N     = 4096                        # GEMM dimension (big enough to saturate)

    # make sure tensor cores are actually used
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # :contentReference[oaicite:0]{index=0}

    yield_time = [0.0]                  # mutable container for the ctx-manager

    tc_gemm  = gemm_tc (dev, dtype, N,        args.trials)
    tc_conv  = conv2d_tc(dev, dtype,         args.trials)
    fp32_gemm= gemm_fp32(dev,        N,      args.trials)
    dram_bw, cache_bw = bandwidth(dev,       args.trials)

    # Tensor-core metric = geometric mean of GEMM + Conv2D
    tc_metric = math.sqrt(tc_gemm * tc_conv)

    # -------------------  pretty print  ------------------------------------ #
    print(f"\nTensor-Core GEMM ({args.dtype}) : {tc_gemm:7.2f} TFLOPS")
    print(f"Tensor-Core Conv2D ({args.dtype}): {tc_conv:7.2f} TFLOPS")
    print(f"FP32 GEMM                 : {fp32_gemm:7.2f} TFLOPS")
    print(f"Peak DRAM bandwidth       : {dram_bw:7.0f} GB/s")
    print(f"Peak 'cache window' BW    : {cache_bw:7.0f} GB/s")

    # -------------------  final ML-GPU score  ------------------------------ #
    w_tc, w_bw, w_cache, w_fp32 = 0.4, 0.3, 0.2, 0.1
    ml_score = math.exp(
          w_tc   * math.log(tc_metric)
        + w_bw   * math.log(dram_bw)
        + w_cache* math.log(cache_bw)
        + w_fp32 * math.log(fp32_gemm)
    )
    print("\nML-GPU SCORE "
          f"(weights: TC {w_tc} | Mem {w_bw} | Cache {w_cache} | FP32 {w_fp32})")
    print(f"⇒  {ml_score:0.1f}")
