import torch

from .data import decode_targets


def any_label_ce(logits, multi, valid):
    # -log(сумма вероятностей всех истинных меток пикселя);
    # для пикселя с одной меткой совпадает с обычной кросс-энтропией.
    multi = multi.clone()
    multi[:, 0] |= ~valid
    lse_all = torch.logsumexp(logits, 1)
    lse_pos = torch.logsumexp(logits.masked_fill(~multi, float("-inf")), 1)
    return (lse_all - lse_pos)[valid].mean()


def soft_dice_loss(logits, multi, valid, eps=1.0):
    probs = logits.softmax(1) * valid.unsqueeze(1)
    target = multi.float()
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    card = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (card + eps)
    present = target.sum(dims) > 0
    return (1 - dice)[present].mean()


class SegLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, bm):
        logits = logits.float()
        multi, valid = decode_targets(bm, self.num_classes)
        if not valid.any():
            return logits.sum() * 0
        return any_label_ce(logits, multi, valid) + soft_dice_loss(logits, multi, valid)
