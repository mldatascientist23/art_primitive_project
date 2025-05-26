import math, random
from typing import List, Tuple

from PIL import Image

from shapes import create_shape, TriangleShape, RectangleShape, EllipseShape
from utils  import image_difference, blend

ShapeJSON = dict   # quick alias for readability


# ────────────────────────────────────────────────────────────
# Simulated-annealing helpers
# ────────────────────────────────────────────────────────────
def _sa_shape(base: Image.Image, target: Image.Image,
              shape, iters, start_t, end_t, pixel_scale) -> object:
    """Optimise a single shape via simulated annealing."""
    w, h = target.size
    curr = shape.copy()
    best = curr.copy()

    curr_diff = image_difference(target, blend(base, curr.rasterize(w, h)))
    best_diff = curr_diff

    for i in range(iters):
        T = start_t * ((end_t / start_t) ** (i / iters))
        cand = curr.copy()
        cand.mutate(w, h, pixel_scale)

        diff = image_difference(target, blend(base, cand.rasterize(w, h)))
        if diff < curr_diff or random.random() < math.exp((curr_diff - diff) / T):
            curr, curr_diff = cand, diff
            if diff < best_diff:
                best, best_diff = cand.copy(), diff
    return best


def _refine(base, target, shape,
            coarse_iters, fine_iters,
            coarse_start_t, coarse_end_t,
            fine_start_t, fine_end_t):
    coarse = _sa_shape(base, target, shape, coarse_iters,
                       coarse_start_t, coarse_end_t, 1.0)
    return _sa_shape(base, target, coarse, fine_iters,
                     fine_start_t, fine_end_t, 0.5)


# ────────────────────────────────────────────────────────────
# Public function
# ────────────────────────────────────────────────────────────
def geometrize(img: Image.Image, shape_type: str, count: int,
               W: int, H: int,
               coarse_iters=1000, fine_iters=500,
               coarse_start_t=100, coarse_end_t=10,
               fine_start_t=10,   fine_end_t=1) -> Tuple[Image.Image, List[ShapeJSON]]:

    # down-scale for speed
    scale = 100 / min(W, H)
    dw, dh = max(1, int(W * scale)), max(1, int(H * scale))
    target = img.convert("RGBA").resize((dw, dh), Image.LANCZOS)

    canvas = Image.new("RGBA", (dw, dh), (255, 255, 255, 255))
    shapes = []

    for _ in range(count):
        s = create_shape(shape_type)
        s.randomize(dw, dh)
        best = _refine(canvas, target, s,
                       coarse_iters, fine_iters,
                       coarse_start_t, coarse_end_t,
                       fine_start_t,   fine_end_t)
        canvas = blend(canvas, best.rasterize(dw, dh))
        shapes.append(best)

    # upscale result to requested W×H
    hi_res = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    for shp in shapes:
        if hasattr(shp, "points"):               # triangle
            shp.points = [(x * W / dw, y * H / dh) for x, y in shp.points]
        else:                                    # rect / ellipse
            shp.x1 *= W / dw; shp.x2 *= W / dw
            shp.y1 *= H / dh; shp.y2 *= H / dh
        hi_res = blend(hi_res, shp.rasterize(W, H))

    # serialise shapes for the front-end animation
    shape_json = []
    for s in shapes:
        if isinstance(s, TriangleShape):
            shape_json.append({"type": "triangle",
                               "points": s.points,
                               "color":  s.color})
        elif isinstance(s, RectangleShape) and not isinstance(s, EllipseShape):
            shape_json.append({"type": "rectangle",
                               "x1": s.x1, "y1": s.y1,
                               "x2": s.x2, "y2": s.y2,
                               "color": s.color})
        else:   # ellipse
            shape_json.append({"type": "ellipse",
                               "x1": s.x1, "y1": s.y1,
                               "x2": s.x2, "y2": s.y2,
                               "color": s.color})
    return hi_res, shape_json
