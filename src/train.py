import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .data import AgriVisionDataset, AugmentSampler
from .losses import SegLoss
from .metrics import scores_from_confusion, update_confusion


def seed_everything(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(frame, mean, std, batch_size, train, workers, seed=C.SEED):
    ds = AgriVisionDataset(frame, mean, std)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=AugmentSampler(len(ds), seed) if train else None,
        drop_last=train,
        num_workers=workers,
        pin_memory=True,
    )


def warmup_cosine(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    return f


@torch.no_grad()
def evaluate(model, loader, num_classes=C.NUM_CLASSES, device="cuda", max_batches=None):
    model.eval()
    loss_fn = SegLoss(num_classes)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    total_loss, n = 0.0, 0
    for i, (x, bm) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, bm = x.to(device, non_blocking=True), bm.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        total_loss += loss_fn(logits, bm).item() * len(x)
        n += len(x)
        update_confusion(cm, logits.float(), bm)
    summary, per_class = scores_from_confusion(cm.cpu())
    return {"loss": total_loss / n, **summary}, per_class, cm.cpu()


def train_model(model, run_name, train_loader, val_loader, epochs, lr, weight_decay,
                warmup_frac=0.05, steps_per_epoch=None, val_batches=None, device="cuda"):
    ckpt_dir, out_dir = C.CKPT_DIR / run_name, C.OUT_DIR / "runs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "warmup_frac": warmup_frac,
              "steps_per_epoch": steps_per_epoch, "batch_size": train_loader.batch_size}
    steps_per_epoch = steps_per_epoch or len(train_loader)
    total = epochs * steps_per_epoch

    model.to(device)
    loss_fn = SegLoss(C.NUM_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, max(1, int(warmup_frac * total))))

    history, start, best = [], 0, -1.0
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        assert ck["params"] == params, f"checkpoint params differ: {ck['params']} vs {params}"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, start, best = ck["history"], ck["epoch"] + 1, ck["best"]
        print(f"[{run_name}] загружен чекпойнт после эпохи {start}")
        for row in history:
            _log(row, epochs)

    for epoch in range(start, epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats(device)
        t0, run_loss, seen = time.time(), 0.0, 0
        for step, (x, bm) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            x, bm = x.to(device, non_blocking=True), bm.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits, bm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(x)
            seen += len(x)
        train_time = time.time() - t0
        val, _, _ = evaluate(model, val_loader, device=device, max_batches=val_batches)
        row = {"epoch": epoch + 1, "train_loss": run_loss / seen, **{f"val_{k}": v for k, v in val.items()},
               "lr_end": sched.get_last_lr()[0], "train_time_s": train_time, "imgs_per_s": seen / train_time,
               "epoch_time_s": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if val["mIoU"] > best:
            best = val["mIoU"]
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "history": history, "epoch": epoch, "best": best,
                    "params": params}, last_path)
        _log(row, epochs)

    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    return pd.DataFrame(history)


def _log(row, epochs):
    print(f"epoch {row['epoch']}/{epochs}: train_loss {row['train_loss']:.4f} | val_loss {row['val_loss']:.4f} | "
          f"val_mIoU {row['val_mIoU']:.4f} | {row['imgs_per_s']:.0f} img/s | {row['epoch_time_s'] / 60:.1f} min")


def load_best(model, run_name, device="cuda"):
    state = torch.load(C.CKPT_DIR / run_name / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
