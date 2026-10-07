"""Evaluate the best checkpoint on the official AHCD test set (3,360 images, never used in training).

Writes results/metrics.json, results/confusion_matrix.png and results/mistakes.png.
Usage: python evaluate.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import DATA_DIR, LETTERS, NAMES, NUM_CLASSES, load_split, preprocess_drawing
from model import ArabicCNN

# Matplotlib can't shape Arabic text, so plots use the transliterated names.


def main():
    x, y = load_split("test")
    model = ArabicCNN(NUM_CLASSES)
    model.load_state_dict(torch.load("models/best.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        probs = torch.cat([model(x[i : i + 512]).softmax(1) for i in range(0, len(x), 512)])
    pred = probs.argmax(1)

    acc = (pred == y).float().mean().item()
    top3 = (probs.topk(3, 1).indices == y[:, None]).any(1).float().mean().item()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t, p] += 1
    per_class = {f"{LETTERS[c]} ({NAMES[c]})": cm[c, c] / cm[c].sum() for c in range(NUM_CLASSES)}
    off = cm.copy()
    np.fill_diagonal(off, 0)
    confused = [
        {"true": f"{LETTERS[t]} ({NAMES[t]})", "predicted": f"{LETTERS[p]} ({NAMES[p]})", "count": int(off[t, p])}
        for t, p in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:8], off.shape))
    ]

    canvas_acc = simulated_canvas_accuracy(model)

    Path("results").mkdir(exist_ok=True)
    metrics = {
        "test_images": len(y),
        "accuracy": acc,
        "top3_accuracy": top3,
        "simulated_canvas_accuracy": canvas_acc,
        "per_class_accuracy": per_class,
        "most_confused_pairs": confused,
    }
    Path("results/metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_confusion(cm)
    plot_mistakes(x, y, pred)

    print(f"Test accuracy: {acc:.2%}   top-3: {top3:.2%}   ({len(y)} images)")
    print(f"Simulated canvas drawings: {canvas_acc:.2%}")
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
    print("Hardest letters:", ", ".join(f"{k} {v:.0%}" for k, v in worst))
    print("Most confused:", ", ".join(f"{c['true']}→{c['predicted']} ×{c['count']}" for c in confused[:5]))


def simulated_canvas_accuracy(model, n: int = 1000, seed: int = 0) -> float:
    """Accuracy on test letters redrawn the way a demo user would draw them.

    Each raw test image is enlarged to a random size (160-320 px), given a random
    pen thickness, placed at a random spot on a white 360x360 canvas, and then
    passed through the demo's preprocess_drawing(). This checks the whole
    drawing -> prediction pipeline, not just the model.
    """
    raw = np.load(DATA_DIR / "ahcd_test.npz")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(raw["y"]), generator=g)[:n]
    correct = 0
    for i in idx.tolist():
        size = int(torch.randint(160, 321, (1,), generator=g))
        pen = int(torch.randint(0, 3, (1,), generator=g)) * 4 + 1
        img = torch.from_numpy(raw["X"][i]).float()[None, None] / 255
        big = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
        big = F.max_pool2d((big > 0.4).float(), pen, stride=1, padding=pen // 2)[0, 0].numpy() > 0
        canvas = np.full((360, 360, 3), 255, np.uint8)
        top, left = (int(v) for v in torch.randint(0, 360 - size + 1, (2,), generator=g))
        canvas[top : top + size, left : left + size][big] = 0
        with torch.no_grad():
            correct += int(model(preprocess_drawing(canvas)).argmax()) == int(raw["y"][i])
    return correct / n


def plot_confusion(cm: np.ndarray) -> None:
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), NAMES, rotation=90)
    ax.set_yticks(range(NUM_CLASSES), NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix (test set, row-normalized)")
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            if t != p and cm[t, p] >= 3:
                ax.text(p, t, cm[t, p], ha="center", va="center", fontsize=7, color="crimson")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=120)


def plot_mistakes(x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, n: int = 24) -> None:
    wrong = torch.nonzero(pred != y).flatten()[:n]
    cols = 8
    fig, axes = plt.subplots(-(-len(wrong) // cols), cols, figsize=(cols * 1.4, 1.7 * -(-len(wrong) // cols)))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), wrong.tolist()):
        ax.imshow(x[i, 0], cmap="gray")
        ax.set_title(f"{NAMES[y[i]]}→{NAMES[pred[i]]}", fontsize=8)
    fig.suptitle("Sample mistakes (true → predicted)")
    fig.tight_layout()
    fig.savefig("results/mistakes.png", dpi=120)


if __name__ == "__main__":
    main()
