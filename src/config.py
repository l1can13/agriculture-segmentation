import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="IProgress not found")

IN_COLAB = "google.colab" in sys.modules

LAB_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    Path("/content/datasets/agriculture-vision")
    if IN_COLAB
    else LAB_DIR.parent / "datasets" / "agriculture-vision"
)
RAW_DIR = DATA_ROOT / "Agriculture-Vision-2021"
CACHE_DIR = DATA_ROOT / "cache"
MANIFEST_PATH = DATA_ROOT / "data_manifest.csv"
OUT_DIR = LAB_DIR / "outputs"
CKPT_DIR = LAB_DIR / "checkpoints"

S3 = "https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision"
ARCHIVE_URL = f"{S3}/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz"
MANIFEST_URL = f"{S3}/TMLR_benchmarks_2023/data_manifest.csv"

PATTERN_CLASSES = [
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "water",
    "waterway",
    "weed_cluster",
]
# Есть в данных, но не входил в оцениваемые классы соревнования 2021 г.
EXTRA_LABELS = ["storm_damage"]
CLASSES = ["background"] + PATTERN_CLASSES
NUM_CLASSES = len(CLASSES)

INVALID_BIT = 1 << 15
IMG_SIZE = 512
SEED = 42
