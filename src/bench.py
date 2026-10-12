import logging
import time

import torch

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from . import config as C

# Число потоков CPU на момент импорта (по умолчанию — число физических ядер).
# Служебный поток DataLoader вызывает torch.set_num_threads(1), поэтому
# для замеров на CPU число потоков задаётся явно.
CPU_THREADS = torch.get_num_threads()


def count_gflops(model, in_channels=4, size=C.IMG_SIZE):
    model = model.eval().cpu()
    x = torch.randn(1, in_channels, size, size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def latency_ms(model, device, batch_size=1, iters=50, warmup=10, bf16=True, in_channels=4, size=C.IMG_SIZE):
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    try:
        return _latency_ms(model, device, batch_size, iters, warmup, bf16, in_channels, size)
    finally:
        torch.set_num_threads(prev_threads)


@torch.no_grad()
def _latency_ms(model, device, batch_size, iters, warmup, bf16, in_channels, size):
    model = model.eval().to(device)
    x = torch.randn(batch_size, in_channels, size, size, device=device)
    use_amp = bf16 and device == "cuda"
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model(x)
    sync()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model(x)
        sync()
        times.append((time.perf_counter() - t) * 1000)
    times = torch.tensor(times)
    return float(times.median()), float(times.std())
