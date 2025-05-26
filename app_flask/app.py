#!/usr/bin/env python
# app.py – Flask server for stego‑shape detection + colour‑recipe helper
# ---------------------------------------------------------------------

import base64
import itertools
import math
import os

import cv2
import numpy as np
from flask import (
    Flask, render_template, request,
    session, jsonify
)

from shape_detector import detect_and_decode

# ─────────── Flask setup ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(16)

# ─────────── Colour‑database helpers ────────────────────────────────
def read_color_file(path: str = "color.txt") -> str:
    with open(path, encoding="utf8") as f:
        return f.read()

def parse_color_db(txt: str):
    dbs, cur = {}, None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line[0].isdigit():
            cur = line
            dbs[cur] = []
        else:
            tok = line.split()
            if len(tok) < 3:
                continue
            parts = tok[-2].split(",")
            if len(parts) != 3:
                continue
            try:
                r, g, b = map(int, parts)
            except ValueError:
                continue
            name = " ".join(tok[1:-2])
            dbs[cur].append((name, (r, g, b)))
    return dbs

def convert_db_list_to_dict(lst):
    return {n: list(rgb) for n, rgb in lst}

def mix_colors(recipe):
    total = sum(p for _, p in recipe)
    r = sum(rgb[0] * p for rgb, p in recipe) / total
    g = sum(rgb[1] * p for rgb, p in recipe) / total
    b = sum(rgb[2] * p for rgb, p in recipe) / total
    return (round(r), round(g), round(b))

def color_error(c1, c2):
    return math.dist(c1, c2)

def generate_recipes(target, base_colors, step=10.0):
    base = list(base_colors.items())
    candidates = []

    # single‑colour quick matches
    for name, rgb in base:
        err = color_error(rgb, target)
        if err < 5:
            candidates.append(([(name, 100)], rgb, err))

    # triple‑mix brute‑force search
    for (n1, r1), (n2, r2), (n3, r3) in itertools.combinations(base, 3):
        for p1 in np.arange(0, 101, step):
            for p2 in np.arange(0, 101 - p1, step):
                p3 = 100 - p1 - p2
                recipe = [(n1, p1), (n2, p2), (n3, p3)]
                mixed  = mix_colors([(r1, p1), (r2, p2), (r3, p3)])
                err    = color_error(mixed, target)
                candidates.append((recipe, mixed, err))

    candidates.sort(key=lambda x: x[2])
    top, seen = [], set()
    for rec, mix, err in candidates:
        key = tuple(sorted((n, p) for n, p in rec if p > 0))
        if key not in seen:
            seen.add(key)
            top.append((rec, mix, err))
        if len(top) == 3:
            break
    return top

COLOR_DBS = parse_color_db(read_color_file("color.txt"))

# ──────────────── ROUTES ────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    decoded_b64 = None
    recipes     = None
    colors      = session.get("grouped", [])

    if request.method == "POST" and request.form.get("action") == "decode":
        # user‑selected shape (Triangle / Rectangle / Circle)
        shape = request.form.get("shape", "Triangle")

        # detect + decode + annotate
        ann_png_bytes, _tris_js, grouped_raw = detect_and_decode(
            request.files["image"].read(),
            shape=shape,
            min_size=int(request.form.get("min_size", 5)),
            max_size=int(request.form.get("max_size", 50))
        )

        # --- resize annotated image to 400×400 so the file itself is fixed size ---
        ann_img = cv2.imdecode(np.frombuffer(ann_png_bytes, np.uint8),
                               cv2.IMREAD_COLOR)
        ann_img = cv2.resize(ann_img, (400, 400), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", ann_img)
        if not ok:
            raise RuntimeError("Could not re‑encode resized PNG")
        ann_png_bytes = buf.tobytes()
        # -------------------------------------------------------------------------

        # convert NumPy scalars → int for session JSON
        grouped = [((int(c[0]), int(c[1]), int(c[2])), int(cnt))
                   for c, cnt in grouped_raw]

        session["grouped"] = grouped
        colors = grouped

        decoded_b64 = base64.b64encode(ann_png_bytes).decode("ascii")

    return render_template(
        "index.html",
        decoded_image = decoded_b64,
        colors        = colors,
        dbs           = list(COLOR_DBS.keys()),
        recipes       = recipes,
    )

@app.post("/generate_recipe")
def generate_recipe_endpoint():
    try:
        data   = request.form
        target = tuple(map(int, data["base_color"].split(",")))
        dbkey  = data["db_choice"]
        step   = float(data["step"])

        base_dict = convert_db_list_to_dict(COLOR_DBS[dbkey])
        recipes   = generate_recipes(target, base_dict, step)

        out = []
        for rec, mix, err in recipes:
            out.append({
                "error":  round(err, 2),
                "mix":    mix,
                "recipe": [{"name": n, "perc": round(p, 1)} for n, p in rec]
            })
        return jsonify(ok=True, recipes=out)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 400

# ─────────────── main ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
