# app.py
from flask import Flask, render_template_string

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Canvas Circles</title>
    <style>
        #myCanvas {
            border: 1px solid #000;
        }
    </style>
</head>
<body>
    <button id="startBtn">Start</button>
    <canvas id="myCanvas" width="400" height="400"></canvas>
    <script>
        const canvas = document.getElementById('myCanvas');
        const ctx = canvas.getContext('2d');
        document.getElementById('startBtn').addEventListener('click', () => {
            const total = 100;
            const radius = 20;
            const circles = Array.from({ length: total }, () => ({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: radius
            }));
            circles.forEach((circle, i) => {
                setTimeout(() => {
                    ctx.beginPath();
                    ctx.arc(circle.x, circle.y, circle.r, 0, Math.PI * 2);
                    ctx.fillStyle = 'blue';
                    ctx.fill();
                }, i * 1000);
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(debug=True)
