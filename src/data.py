import hashlib
import shutil
import subprocess
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from . import config as C


def download_file(url, dst, chunk=1 << 22):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    done = dst.stat().st_size if dst.exists() else 0
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers["Content-Length"])
    if done == total:
        return dst
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
    with urllib.request.urlopen(req) as r, open(dst, "ab") as f, tqdm(
        total=total, initial=done, unit="B", unit_scale=True, desc=dst.name, mininterval=30
    ) as bar:
        while block := r.read(chunk):
            f.write(block)
            bar.update(len(block))
    return dst


def prepare_dataset():
    if not C.MANIFEST_PATH.exists():
        download_file(C.MANIFEST_URL, C.MANIFEST_PATH)
    if C.RAW_DIR.exists():
        return
    archive = download_file(C.ARCHIVE_URL, C.DATA_ROOT / "Agriculture-Vision-2021.tar.gz")
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(C.DATA_ROOT)], check=True)


def paths(split, image_id):
    d = C.RAW_DIR / split
    return {
        "rgb": d / "images" / "rgb" / f"{image_id}.jpg",
        "nir": d / "images" / "nir" / f"{image_id}.jpg",
        "boundary": d / "boundaries" / f"{image_id}.png",
        "mask": d / "masks" / f"{image_id}.png",
        **{c: d / "labels" / c / f"{image_id}.png" for c in C.PATTERN_CLASSES + C.EXTRA_LABELS},
    }


def list_ids(folder, suffix):
    return {p.name[: -len(suffix)] for p in Path(folder).glob(f"*{suffix}")} if Path(folder).exists() else set()


def check_completeness(split):
    d = C.RAW_DIR / split
    ids = {
        "rgb": list_ids(d / "images" / "rgb", ".jpg"),
        "nir": list_ids(d / "images" / "nir", ".jpg"),
        "boundary": list_ids(d / "boundaries", ".png"),
        "mask": list_ids(d / "masks", ".png"),
        **{c: list_ids(d / "labels" / c, ".png") for c in C.PATTERN_CLASSES + C.EXTRA_LABELS},
    }
    ref = ids["rgb"]
    report = pd.DataFrame(
        {
            "files": {k: len(v) for k, v in ids.items()},
            "missing_vs_rgb": {k: len(ref - v) for k, v in ids.items()},
            "extra_vs_rgb": {k: len(v - ref) for k, v in ids.items()},
        }
    )
    return sorted(ref), report


def build_index(splits=("train", "val")):
    manifest = pd.read_csv(C.MANIFEST_PATH)
    manifest = manifest[manifest.dataset == "supervised"][["flightcode", "field", "split"]]
    rows = []
    for split in splits:
        ids, _ = check_completeness(split)
        rows += [{"id": i, "official_split": split, "flightcode": i.split("_")[0]} for i in ids]
    index = pd.DataFrame(rows).merge(manifest, on="flightcode", how="left")
    assert index.field.notna().all(), "flight without field in manifest"
    assert (index.split == index.official_split).all(), "manifest split mismatch"
    return index.drop(columns="split").astype({"field": int})


def split_by_field(index, val_frac=0.1, seed=C.SEED):
    train = index[index.official_split == "train"]
    fields = train.groupby("field").size().sample(frac=1.0, random_state=seed)
    val_fields = set(fields.index[fields.cumsum() <= val_frac * len(train)])
    split = np.where(index.official_split == "val", "test", np.where(index.field.isin(val_fields), "val", "train"))
    return index.assign(split=split)


def _encode_one(args):
    split, image_id, out_path = args
    p = paths(split, image_id)
    read = lambda f: np.array(Image.open(f))
    rgb, nir = read(p["rgb"]), read(p["nir"])
    valid = (read(p["mask"]) > 0) & (read(p["boundary"]) > 0)
    bm = np.where(valid, 0, C.INVALID_BIT).astype(np.uint16)
    counts = {}
    n_labels = np.zeros(valid.shape, np.uint8)
    for k, c in enumerate(C.PATTERN_CLASSES):
        lab = read(p[c]) > 0
        bm |= (lab.astype(np.uint16) << k)
        n_labels += lab & valid
        counts[f"px_{c}"] = int((lab & valid).sum())
        counts[f"px_outside_{c}"] = int((lab & ~valid).sum())
    for c in C.EXTRA_LABELS:
        lab = read(p[c]) > 0
        counts[f"px_{c}"] = int((lab & valid).sum())
        counts[f"px_{c}_only"] = int((lab & valid & (n_labels == 0)).sum())
    Image.fromarray(bm).save(out_path)
    return {
        "id": image_id,
        "rgb_shape": rgb.shape,
        "nir_shape": nir.shape,
        "rgb_md5": hashlib.md5(p["rgb"].read_bytes()).hexdigest(),
        "px_valid": int(valid.sum()),
        "px_background": int((valid & (n_labels == 0)).sum()),
        "px_overlap": int((n_labels > 1).sum()),
        **counts,
        **{f"mean_{ch}": float(v) for ch, v in zip("RGB", rgb[valid].mean(0) if valid.any() else [np.nan] * 3)},
        "mean_N": float(nir[valid].mean()) if valid.any() else np.nan,
    }


def build_label_cache(index, workers=8):
    out_dir = C.CACHE_DIR / "labels"
    stats_path = C.CACHE_DIR / "image_stats.csv"
    if stats_path.exists():
        return pd.read_csv(stats_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r.official_split, r.id, out_dir / f"{r.id}.png") for r in index.itertuples()]
    with Pool(workers) as pool:
        rows = list(tqdm(pool.imap(_encode_one, jobs, chunksize=64), total=len(jobs), desc="label cache", mininterval=30))
    stats = pd.DataFrame(rows)
    stats.to_csv(stats_path, index=False)
    return stats


def compute_norm_stats(index, n=2000, seed=C.SEED):
    sample = index[index.split == "train"].sample(n=min(n, (index.split == "train").sum()), random_state=seed)
    s = np.zeros(4)
    s2 = np.zeros(4)
    cnt = 0
    for r in sample.itertuples():
        x = load_image(r.official_split, r.id).reshape(-1, 4).astype(np.float64) / 255
        valid = (load_bitmask(r.id).reshape(-1) & C.INVALID_BIT) == 0
        x = x[valid]
        s += x.sum(0)
        s2 += (x**2).sum(0)
        cnt += len(x)
    mean = s / cnt
    std = np.sqrt(s2 / cnt - mean**2)
    return mean.tolist(), std.tolist()


def load_image(split, image_id):
    p = paths(split, image_id)
    rgb = np.array(Image.open(p["rgb"]))
    nir = np.array(Image.open(p["nir"]))
    return np.dstack([rgb, nir])


def load_bitmask(image_id):
    return np.array(Image.open(C.CACHE_DIR / "labels" / f"{image_id}.png")).astype(np.int32)


class AgriVisionDataset(Dataset):
    def __init__(self, frame, mean, std):
        self.items = list(zip(frame.official_split, frame.id))
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        # item — индекс или пара (индекс, код аугментации 0..7 от AugmentSampler)
        i, aug = item if isinstance(item, tuple) else (item, 0)
        split, image_id = self.items[i]
        x = load_image(split, image_id)
        y = load_bitmask(image_id)
        k, flip = aug % 4, aug >= 4
        x, y = np.rot90(x, k), np.rot90(y, k)
        if flip:
            x, y = x[:, ::-1], y[:, ::-1]
        x = (x.astype(np.float32) / 255 - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y))
        return x, y


class AugmentSampler(Sampler):
    # Порядок снимков и аугментации — детерминированная функция (seed, эпоха):
    # все модели видят одинаковую последовательность данных.
    def __init__(self, n, seed=C.SEED):
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.n, generator=g).tolist()
        augs = torch.randint(0, 8, (self.n,), generator=g).tolist()
        return iter(zip(order, augs))


def decode_targets(bm, num_classes=C.NUM_CLASSES):
    valid = (bm & C.INVALID_BIT) == 0
    bits = torch.arange(num_classes - 1, device=bm.device).view(1, -1, 1, 1)
    patterns = ((bm.unsqueeze(1) >> bits) & 1).bool()
    background = ~patterns.any(1, keepdim=True)
    multi = torch.cat([background, patterns], 1) & valid.unsqueeze(1)
    return multi, valid
