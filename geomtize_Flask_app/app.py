#!/usr/bin/env python
import os
import io
import time
import math
import base64
import itertools
import json

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

from geometrize import geometrize

app = Flask(__name__)
app.secret_key = os.urandom(16)

# ── Geometrize defaults ─────────────────────────────────────────────
DEFAULTS = dict(
    geom_checked=True,
    shape_type   ='triangle',
    shape_count  =300,
    new_width    =500,
    new_height   =500,
)
def make_ctx(**extra):
    ctx = DEFAULTS.copy()
    ctx.update(extra)
    return ctx

# ── Colour-DB parsers ────────────────────────────────────────────────
def read_color_file(path="color.txt") -> str:
    with open(path, encoding="utf8") as f: return f.read()

def parse_color_db(txt):
    db, cur = {}, None
    for line in txt.splitlines():
        line = line.strip()
        if not line: continue
        if not line[0].isdigit():
            cur = line
            db[cur] = []
        else:
            parts = line.split()
            rgb = parts[-2].split(",")
            if len(rgb)==3:
                try:
                    r,g,b = map(int, rgb)
                    name = " ".join(parts[1:-2])
                    db[cur].append((name, (r,g,b)))
                except: pass
    return db

def generate_recipes(target, base_colors, step=10.0):
    def err(c1,c2): return math.dist(c1,c2)
    def mix(recipe):
        total = sum(p for _,p in recipe)
        r = sum(rgb[0]*p for rgb,p in recipe)/total
        g = sum(rgb[1]*p for rgb,p in recipe)/total
        b = sum(rgb[2]*p for rgb,p in recipe)/total
        return (round(r),round(g),round(b))

    base = list(base_colors.items())
    cand = []
    # single-colour quick
    for name, rgb in base:
        e = err(rgb, target)
        if e < 5: cand.append(([(name,100)], rgb, e))
    # brute triple mixes
    for (n1,r1),(n2,r2),(n3,r3) in itertools.combinations(base,3):
        for p1 in np.arange(0,101,step):
            for p2 in np.arange(0,101-p1,step):
                p3 = 100-p1-p2
                rec = [(n1,p1),(n2,p2),(n3,p3)]
                m = mix([(r1,p1),(r2,p2),(r3,p3)])
                cand.append((rec, m, err(m, target)))
    cand.sort(key=lambda x:x[2])
    top,seen = [], set()
    for rec,m,e in cand:
        key = tuple(sorted((n,p) for n,p in rec if p>0))
        if key not in seen:
            seen.add(key)
            top.append((rec,m,e))
        if len(top)==3: break
    return top

COLOR_DBS = parse_color_db(read_color_file("color.txt"))

# ── Main Geometrize route ──────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def index():
    ctx = make_ctx()
    if request.method=="POST":
        # 1) pull form
        vals = dict(
            geom_checked=True,
            shape_type   =request.form["shape_type"],
            shape_count  =int(request.form["shape_count"]),
            new_width    =int(request.form["new_width"]),
            new_height   =int(request.form["new_height"]),
        )
        ctx = make_ctx(**vals)

        # 2) handle image
        f = request.files.get("image")
        if f:
            img = Image.open(f.stream).convert("RGBA")
            t0 = time.time()
            still, shapes = geometrize(img, vals["shape_type"],
                                       vals["shape_count"],
                                       vals["new_width"],
                                       vals["new_height"])
            rt = round(time.time()-t0,2)
            buf = io.BytesIO()
            still.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            ctx.update(
                still_png   = b64,
                shapes_json = shapes,
                dims        = (vals["new_width"], vals["new_height"]),
                runtime     = rt,
                max_step    = len(shapes),
            )
        else:
            # reuse previous if no new upload
            if request.form.get("prev_image_b64"):
                ctx.update(
                    still_png   = request.form["prev_image_b64"],
                    shapes_json = json.loads(request.form["prev_shapes_json"]),
                    dims        = json.loads(request.form["prev_dims"]),
                    runtime     = request.form["prev_runtime"],
                    max_step    = int(request.form["prev_max_step"]),
                )

    return render_template(
        "index.html",
        dbs=list(COLOR_DBS.keys()),
        color_db=COLOR_DBS,
        **ctx
    )

# ── AJAX endpoint for color recipes ─────────────────────────────────
@app.post("/generate_recipe")
def generate_recipe_endpoint():
    try:
        base_color = tuple(map(int, request.form["base_color"].split(",")))
        db_choice  = request.form["db_choice"]
        step       = float(request.form["step"])
        base_dict  = dict(COLOR_DBS[db_choice])
        recs       = generate_recipes(base_color, base_dict, step)

        out = []
        for rec, mix, err in recs:
            out.append({
                "error":  round(err,2),
                "mix":    mix,
                "recipe": [
                  {"name":n, "perc":round(p,1), "rgb":base_dict[n]}
                  for n,p in rec
                ]
            })
        return jsonify(ok=True, recipes=out)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 400

if __name__=="__main__":
    os.environ.setdefault("NUMBA_CACHE_DIR", ".numba_cache")
    app.run(debug=True)
