"""Train the CNN with a custom PyTorch loop.

Holds out 10% of the official training set for validation (the official test set
is only used once, in evaluate.py). Saves the best checkpoint by validation accuracy.

Usage: python train.py [--epochs 40] [--batch-size 128]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from data import NUM_CLASSES, augment, load_split
from model import ArabicCNN


def run_epoch(model, x, y, batch_size, loss_fn, optimizer=None, scheduler=None):
    """One pass over (x, y). Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(x)) if training else torch.arange(len(x))
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for i in range(0, len(x), batch_size):
            idx = order[i : i + batch_size]
            xb, yb = x[idx], y[idx]
            if training:
                xb = augment(xb)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=8, help="stop after N epochs without val improvement")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x, y = load_split("train")

    # Stratified 90/10 train/validation split
    val_idx = torch.cat([
        torch.nonzero(y == c).flatten()[torch.randperm(int((y == c).sum()))[: int((y == c).sum()) // 10]]
        for c in range(NUM_CLASSES)
    ])
    mask = torch.ones(len(y), dtype=torch.bool)
    mask[val_idx] = False
    x_train, y_train, x_val, y_val = x[mask], y[mask], x[val_idx], y[val_idx]
    print(f"train {len(x_train)}  val {len(x_val)}")

    model = ArabicCNN(NUM_CLASSES)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    steps_per_epoch = -(-len(x_train) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    history, best_acc, bad_epochs = [], 0.0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, x_train, y_train, args.batch_size, loss_fn, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, x_val, y_val, 512, loss_fn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        flag = ""
        if va_acc > best_acc:
            best_acc, bad_epochs, flag = va_acc, 0, "  ✓ saved"
            torch.save(model.state_dict(), "models/best.pt")
        else:
            bad_epochs += 1
        print(f"epoch {epoch:2d}  train {tr_loss:.3f}/{tr_acc:.2%}  val {va_loss:.3f}/{va_acc:.2%}  {time.time() - start:.0f}s{flag}")
        if bad_epochs >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    Path("results/history.json").write_text(json.dumps(history, indent=2))
    plot_history(history)
    print(f"Best validation accuracy: {best_acc:.2%}")


def plot_history(history: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set(title="Loss", xlabel="epoch")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="validation")
    ax2.set(title="Accuracy", xlabel="epoch")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("results/training_curves.png", dpi=120)


if __name__ == "__main__":
    main()
