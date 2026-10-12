import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from . import config as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BACKGROUND_RGB = "#f0efec"
INVALID_RGB = "#383835"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "text.color": INK,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "lines.linewidth": 2,
})

CLASS_COLORS = [BACKGROUND_RGB] + SERIES
MASK_CMAP = ListedColormap(CLASS_COLORS + [INVALID_RGB])
INVALID_LABEL = len(CLASS_COLORS)


def save(fig, name):
    path = C.OUT_DIR / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def bitmask_to_labelmap(bm):
    # Для отображения: при перекрытии показывается класс с меньшим номером.
    bm = np.asarray(bm)
    out = np.zeros(bm.shape, np.uint8)
    for k in reversed(range(len(C.PATTERN_CLASSES))):
        out[((bm >> k) & 1) == 1] = k + 1
    out[(bm & C.INVALID_BIT) != 0] = INVALID_LABEL
    return out


def class_legend(fig, ncol=5):
    handles = [Patch(color=c, label=name) for c, name in zip(CLASS_COLORS, C.CLASSES)]
    handles.append(Patch(color=INVALID_RGB, label="invalid"))
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8)


def show_mask(ax, labelmap, title=None):
    ax.imshow(labelmap, cmap=MASK_CMAP, vmin=0, vmax=INVALID_LABEL, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_rgb(ax, img4, title="RGB"):
    ax.imshow(img4[..., :3])
    ax.set_title(title)
    ax.axis("off")


def show_nir(ax, img4, title="NIR"):
    ax.imshow(img4[..., 3], cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")


def grouped_barh(ax, frame, colors=SERIES, value_fmt="{:.2f}", xlabel=None):
    # frame: строки — категории, столбцы — серии
    n_rows, n_cols = frame.shape
    height = 0.8 / n_cols
    y = np.arange(n_rows)
    for j, col in enumerate(frame.columns):
        pos = y - 0.4 + height * (j + 0.5)
        ax.barh(pos, frame[col].values, height=height * 0.9, color=colors[j], label=col)
        if value_fmt:
            for p, v in zip(pos, frame[col].values):
                ax.text(v, p, " " + value_fmt.format(v), va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, frame.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(loc="lower right", fontsize=8)
