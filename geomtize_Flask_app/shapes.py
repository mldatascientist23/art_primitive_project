import random
from typing import List, Tuple

from PIL import Image, ImageDraw

from utils import clamp

RGBA = Tuple[int, int, int, int]


# ────────────────────────────────────────────────────────────
# Abstract base
# ────────────────────────────────────────────────────────────
class BaseShape:
    def __init__(self) -> None:
        self.color: RGBA = (255, 0, 0, 128)

    # ––– behaviour everyone shares –––
    def randomize_color(self):
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(20, 200),
        )

    # ––– required in subclasses –––
    def copy(self):              raise NotImplementedError
    def randomize(self, w, h):   raise NotImplementedError
    def mutate(self, w, h, amt): raise NotImplementedError
    def rasterize(self, w, h):   raise NotImplementedError


# ────────────────────────────────────────────────────────────
# Concrete shapes
# ────────────────────────────────────────────────────────────
class TriangleShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.points: List[Tuple[int, int]] = [(0, 0)] * 3

    # ------------ behaviour ------------
    def copy(self):
        s = TriangleShape()
        s.color = self.color
        s.points = list(self.points)
        return s

    def randomize(self, w, h):
        self.randomize_color()
        self.points = [
            (random.randint(0, w - 1), random.randint(0, h - 1))
            for _ in range(3)
        ]

    def mutate(self, w, h, amt):
        if random.random() < 0.3:                      # maybe tweak colour
            r, g, b, a = self.color
            self.color = (
                clamp(r + random.randint(-15, 15), 0, 255),
                clamp(g + random.randint(-15, 15), 0, 255),
                clamp(b + random.randint(-15, 15), 0, 255),
                clamp(a + random.randint(-15, 15), 20, 255),
            )

        pts = []
        for x, y in self.points:
            if random.random() < 0.5:
                x = clamp(x + int(random.randint(-5, 5) * amt), 0, w - 1)
            if random.random() < 0.5:
                y = clamp(y + int(random.randint(-5, 5) * amt), 0, h - 1)
            pts.append((x, y))
        self.points = pts

    def rasterize(self, w, h):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img, "RGBA").polygon(self.points, fill=self.color)
        return img


class RectangleShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.x1 = self.y1 = self.x2 = self.y2 = 0

    # keep subclass type (fixes ellipse bug)
    def copy(self):
        s = self.__class__()  # RectangleShape *or* EllipseShape
        s.color = self.color
        s.x1, s.y1, s.x2, s.y2 = self.x1, self.y1, self.x2, self.y2
        return s

    def randomize(self, w, h):
        self.randomize_color()
        self.x1, self.y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        self.x2 = clamp(self.x1 + random.randint(-50, 50), 0, w - 1)
        self.y2 = clamp(self.y1 + random.randint(-50, 50), 0, h - 1)

    def mutate(self, w, h, amt):
        if random.random() < 0.3:
            r, g, b, a = self.color
            self.color = (
                clamp(r + random.randint(-15, 15), 0, 255),
                clamp(g + random.randint(-15, 15), 0, 255),
                clamp(b + random.randint(-15, 15), 0, 255),
                clamp(a + random.randint(-15, 15), 20, 255),
            )
        for attr in ("x1", "y1", "x2", "y2"):
            if random.random() < 0.5:
                val = getattr(self, attr)
                lim = w - 1 if "x" in attr else h - 1
                setattr(
                    self,
                    attr,
                    clamp(val + int(random.randint(-5, 5) * amt), 0, lim),
                )

    def rasterize(self, w, h):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])
        ImageDraw.Draw(img, "RGBA").rectangle([x1, y1, x2, y2], fill=self.color)
        return img


class EllipseShape(RectangleShape):
    def rasterize(self, w, h):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])
        ImageDraw.Draw(img, "RGBA").ellipse([x1, y1, x2, y2], fill=self.color)
        return img


# factory helper
def create_shape(shape_name: str) -> BaseShape:
    return {"triangle": TriangleShape,
            "rectangle": RectangleShape,
            "ellipse":   EllipseShape}[shape_name]()
