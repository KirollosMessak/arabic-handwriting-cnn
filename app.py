"""Draw an Arabic letter and the CNN recognizes it. Runs locally or on Hugging Face Spaces (ZeroGPU)."""

from pathlib import Path

import gradio as gr
import torch

from data import LETTERS, NAMES, NUM_CLASSES, preprocess_drawing
from model import ArabicCNN

try:
    # Free Hugging Face Gradio Spaces run on ZeroGPU, which requires at least one @spaces.GPU function.
    import spaces

    gpu = spaces.GPU(duration=10)
except ImportError:  # running locally
    def gpu(fn):
        return fn

# models/best.pt in the repo; best.pt next to app.py in a flat Hugging Face Space upload
MODEL_PATH = next(p for p in (Path("models/best.pt"), Path("best.pt")) if p.exists())

model = ArabicCNN(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


@gpu
def classify(x: torch.Tensor) -> torch.Tensor:
    # The model is tiny (290K parameters), so CPU inference is fast even on ZeroGPU.
    with torch.no_grad():
        return model(x).softmax(1)[0]


def recognize(drawing):
    if drawing is None or drawing.get("composite") is None:
        return {}, None
    x = preprocess_drawing(drawing["composite"])
    if x.max() == 0:
        return {}, None
    probs = classify(x)
    labels = {f"{LETTERS[i]}  ({NAMES[i]})": float(p) for i, p in enumerate(probs)}
    # Also show what the model actually "sees", enlarged with sharp pixels
    preview = torch.nn.functional.interpolate(x, scale_factor=5, mode="nearest")[0, 0].numpy()
    return labels, preview


with gr.Blocks(title="Arabic Handwriting Recognition") as demo:
    gr.Markdown(
        "# ✍️ Arabic Handwriting Recognition\n"
        "Draw one Arabic letter in its isolated form (e.g. **ب**, **ع**, **ش**) and a CNN built from scratch in PyTorch recognizes it.\n\n"
        "Tips: select the ✏️ brush first, include the dots, and draw alef with a hamza (**أ**), as in the training data."
    )
    with gr.Row():
        canvas = gr.Sketchpad(
            label="Draw here",
            type="numpy",
            canvas_size=(320, 320),
            brush=gr.Brush(default_size=14, colors=["#000000"], color_mode="fixed"),
            layers=False,
        )
        with gr.Column():
            result = gr.Label(num_top_classes=5, label="Prediction")
            seen = gr.Image(label="What the model sees (32×32)", height=160, interactive=False)
    canvas.change(recognize, inputs=canvas, outputs=[result, seen], show_progress="hidden")
    gr.Markdown("Letters: " + "  ".join(LETTERS))

if __name__ == "__main__":
    demo.launch()
