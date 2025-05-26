# processing.py
import math, random
import numpy as np
from PIL import Image
from numba import njit

# ... (clamp, image_difference_numba, image_difference, blend_image,
#     BaseShape, TriangleShape, RectangleShape, EllipseShape,
#     create_shape, simulated_annealing_shape, refine_shape) ...


def run_geometrize_no_ui(
    target_img: Image.Image,
    shape_type: str,
    shape_count: int,
    new_width: int,
    new_height: int,
    coarse_iterations=1000,
    fine_iterations=500,
    coarse_start_temp=100.0,
    coarse_end_temp=10.0,
    fine_start_temp=10.0,
    fine_end_temp=1.0
) -> Image.Image:
    # prepare
    target = target_img.convert("RGBA").resize((new_width, new_height), Image.LANCZOS)
    width, height = target.size
    current = Image.new("RGBA", (width, height), (255,255,255,255))
    current_diff = image_difference(target, current)

    # add shapes
    for _ in range(shape_count):
        shape = create_shape(shape_type)
        shape.randomize(width, height)
        best_shape, best_diff = refine_shape(
            current, target, shape,
            coarse_iterations, fine_iterations,
            coarse_start_temp, coarse_end_temp,
            fine_start_temp, fine_end_temp
        )
        if best_diff < current_diff:
            current = blend_image(current, best_shape.rasterize(width, height))
            current_diff = best_diff

    return current
