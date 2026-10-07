import numpy as np
import torch

from data import LETTERS, NAMES, NUM_CLASSES, augment, preprocess_drawing
from model import ArabicCNN


def test_labels_cover_28_letters():
    assert NUM_CLASSES == 28 and len(LETTERS) == len(NAMES) == 28
    assert LETTERS[0] == "أ" and LETTERS[-1] == "ي"


def test_model_output_shape():
    out = ArabicCNN(NUM_CLASSES)(torch.rand(4, 1, 32, 32))
    assert out.shape == (4, NUM_CLASSES)


def test_augment_keeps_shape_and_range():
    x = torch.rand(8, 1, 32, 32)
    y = augment(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_preprocess_centers_dark_stroke_on_white_canvas():
    canvas = np.full((300, 300, 3), 255, np.uint8)
    canvas[40:120, 200:220] = 0  # a vertical stroke in the top-right corner
    x = preprocess_drawing(canvas)
    assert x.shape == (1, 1, 32, 32)
    assert x.max() == 1.0
    ys, xs = torch.nonzero(x[0, 0] > 0.5, as_tuple=True)
    assert 10 <= xs.float().mean() <= 22 and 10 <= ys.float().mean() <= 22  # moved to the center
    assert x[0, 0, :6].max() == 0 and x[0, 0, -6:].max() == 0  # ~18 px letter with margin, like the dataset


def test_preprocess_handles_transparent_canvas_and_empty_input():
    rgba = np.zeros((200, 200, 4), np.uint8)  # transparent background
    assert preprocess_drawing(rgba).max() == 0
    rgba[50:150, 95:105] = [0, 0, 0, 255]  # opaque black stroke
    assert preprocess_drawing(rgba).max() == 1.0


def test_preprocess_gives_same_size_and_stroke_for_small_thin_and_big_bold_drawings():
    def draw(size, pen):
        canvas = np.full((400, 400, 3), 255, np.uint8)
        canvas[50 : 50 + size, 100 : 100 + pen] = 0  # vertical bar
        canvas[50 + size - pen : 50 + size, 100 : 100 + size // 2] = 0  # foot, like an "L"
        return preprocess_drawing(canvas)[0, 0]

    small_thin, big_bold = draw(120, 6), draw(300, 30)
    for x in (small_thin, big_bold):
        rows = torch.nonzero(x.amax(1) > 0.5).flatten()
        assert 16 <= rows.max() - rows.min() + 1 <= 20  # letter height ≈ 18 px
    ink_small, ink_big = (small_thin > 0.5).sum(), (big_bold > 0.5).sum()
    assert abs(ink_small - ink_big) <= 0.35 * max(ink_small, ink_big)  # similar stroke width
