import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import decode_targets


def resolve_targets(pred, multi):
    # Правило Agriculture-Vision: если пиксель имеет несколько меток,
    # предсказание любой из них считается верным.
    first_label = multi.int().argmax(1)
    hit = multi.gather(1, pred.unsqueeze(1)).squeeze(1)
    return torch.where(hit, pred, first_label)


def update_confusion(cm, logits, bm):
    multi, valid = decode_targets(bm, cm.shape[0])
    pred = logits.argmax(1)
    target = resolve_targets(pred, multi)
    n = cm.shape[0]
    idx = target[valid] * n + pred[valid]
    cm += torch.bincount(idx, minlength=n * n).view(n, n)
    return cm


def scores_from_confusion(cm, classes=C.CLASSES):
    cm = cm.double().cpu().numpy() if torch.is_tensor(cm) else np.asarray(cm, np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
    per_class = pd.DataFrame({"IoU": iou, "Dice": dice}, index=classes)
    summary = {
        "mIoU": float(np.nanmean(iou)),
        "mDice": float(np.nanmean(dice)),
        "PixelAcc": float(tp.sum() / cm.sum()),
    }
    return summary, per_class
