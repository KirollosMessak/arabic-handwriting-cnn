"""Dataset download, loading, augmentation and preprocessing for drawn images."""

import io
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR = Path(__file__).parent / "data"
SOURCE_URL = (
    "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/"
    "Arabic%20Handwritten%20Characters%20Dataset%20CSV.zip"
)

LETTERS = list("أبتثجحخدذرزسشصضطظعغفقكلمنهوي")
NAMES = [
    "alef", "beh", "teh", "theh", "jeem", "hah", "khah", "dal", "thal", "reh",
    "zain", "seen", "sheen", "sad", "dad", "tah", "zah", "ain", "ghain", "feh",
    "qaf", "kaf", "lam", "meem", "noon", "heh", "waw", "yeh",
]
NUM_CLASSES = len(LETTERS)


def prepare_data() -> None:
    """Download the AHCD CSV zip (~3 MB) once and convert it to compact .npz files."""
    if (DATA_DIR / "ahcd_train.npz").exists() and (DATA_DIR / "ahcd_test.npz").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "ahcd_csv.zip"
    if not zip_path.exists():
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        for split, n in (("Train", 13440), ("Test", 3360)):
            images = np.loadtxt(io.TextIOWrapper(z.open(f"csv{split}Images {n}x1024.csv")), delimiter=",")
            labels = np.loadtxt(io.TextIOWrapper(z.open(f"csv{split}Label {n}x1.csv")), delimiter=",")
            # The CSV stores each image column-major, so transpose to get upright letters.
            images = images.reshape(-1, 32, 32).transpose(0, 2, 1).astype(np.uint8)
            np.savez_compressed(DATA_DIR / f"ahcd_{split.lower()}.npz", X=images, y=labels.astype(np.int64) - 1)
    print("Data ready in", DATA_DIR)


def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized images (N, 1, 32, 32) in [0, 1] and labels (N,).

    Every image goes through normalize_ink(), the same function the demo uses
    on drawings, so the model sees identical inputs in training and in the app.
    """
    cache = DATA_DIR / f"ahcd_{split}_normalized.pt"
    if cache.exists():
        return torch.load(cache)
    prepare_data()
    data = np.load(DATA_DIR / f"ahcd_{split}.npz")
    # Upscale first so stroke-width measurement isn't limited by the 32 px grid
    raw = F.interpolate(torch.from_numpy(data["X"]).float().unsqueeze(1), scale_factor=8, mode="bilinear")
    x = torch.cat([normalize_ink(img[0].numpy()) for img in raw])
    y = torch.from_numpy(data["y"]).long()
    torch.save((x, y), cache)
    return x, y


def augment(x: torch.Tensor, max_rotate: float = 12, max_shift: float = 0.1, scale=(0.85, 1.1)) -> torch.Tensor:
    """Random rotation, scaling, shear and shift for a batch, done with one affine warp.

    Handwriting varies in angle, size and position; this teaches the model to
    ignore those differences. Flips are not used: a flipped letter is a different letter.
    """
    n = x.shape[0]
    angle = (torch.rand(n) * 2 - 1) * math.radians(max_rotate)
    s = torch.empty(n).uniform_(*scale)
    shear = (torch.rand(n) * 2 - 1) * 0.15
    tx, ty = ((torch.rand(2, n) * 2 - 1) * max_shift * 2)
    cos, sin = torch.cos(angle) / s, torch.sin(angle) / s
    theta = torch.stack(
        [torch.stack([cos, -sin + shear, tx], 1), torch.stack([sin, cos, ty], 1)], 1
    ).to(x)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def preprocess_drawing(img: np.ndarray) -> torch.Tensor:
    """Turn a canvas drawing (any size, RGB/RGBA/gray) into the model's input format."""
    img = img.astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        # Composite over white so transparent background becomes paper.
        alpha = img[..., 3:] / 255.0
        img = img[..., :3] * alpha + 255.0 * (1 - alpha)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return normalize_ink(255.0 - img)  # dark strokes on light paper -> bright strokes


def normalize_ink(ink: np.ndarray) -> torch.Tensor:
    """Bright-strokes-on-black image (any size, 0-255) -> (1, 1, 32, 32) tensor:
    cropped, centered, scaled to a standard size, with a standard stroke width."""
    if ink.max() < 30:
        return torch.zeros(1, 1, 32, 32)

    ys, xs = np.nonzero(ink > 30)
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = ink.shape
    side = max(h, w)
    square = np.zeros((side, side), np.float32)
    square[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = ink

    # Match the training data's statistics: letters span ~18 px of the 32 px
    # image and strokes are ~2.2 px wide, however big or bold the drawing is.
    mask = torch.from_numpy(square > 60).float()[None, None]
    mask = _set_stroke_width(mask, target=2.2 * side / LETTER_SIZE)
    full = round(side * 32 / LETTER_SIZE)
    pad = (full - side) // 2
    t = F.pad(mask, (pad, full - side - pad, pad, full - side - pad))
    t = F.interpolate(t, size=(32, 32), mode="area")
    return (t / t.max()).clamp(0, 1)


LETTER_SIZE = 18  # typical letter height/width in the dataset, in pixels


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(m, k, stride=1, padding=k // 2)


def _set_stroke_width(mask: torch.Tensor, target: float) -> torch.Tensor:
    """Thicken or thin a binary stroke mask so its average stroke width ≈ target.

    Width is estimated as 2 * area / boundary_pixels (a stroke of length L and
    width w has area ≈ wL and ≈ 2L boundary pixels).
    """
    area = mask.sum()
    boundary = (mask - (-_dilate(-mask, 3))).sum()
    width = 2 * area / boundary.clamp(min=1)
    k = 2 * int(round(abs(target - width.item()) / 2)) + 1  # odd kernel; changes width by k - 1
    if k == 1:
        return mask
    if target > width:
        return _dilate(mask, k)
    thinned = -_dilate(-mask, k)
    # Don't erase small parts like dots: keep the original if thinning removes too much
    return thinned if thinned.sum() > 0.25 * area else mask
