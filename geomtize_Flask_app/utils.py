import numpy as np
from numba import njit
from PIL import Image

# ────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────
def clamp(v: int, mn: int, mx: int) -> int:
    """Restrict *v* to the closed interval [mn, mx]."""
    return max(mn, min(v, mx))


@njit(cache=True)                 # heavy loop → numba-compiled
def _image_difference_numba(a, b):
    diff = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(4):
                d = int(a[i, j, k]) - int(b[i, j, k])
                diff += d * d
    return diff


def image_difference(img_a: Image.Image, img_b: Image.Image) -> int:
    """Squared-error distance between two RGBA images."""
    return _image_difference_numba(
        np.array(img_a, np.uint8),
        np.array(img_b, np.uint8)
    )


def blend(base: Image.Image, top: Image.Image) -> Image.Image:
    """Alpha-composite *top* over *base*."""
    return Image.alpha_composite(base, top)
