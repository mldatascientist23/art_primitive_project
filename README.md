# 🎨 Artistic Vision Studio

An advanced, interactive image stylization and color design platform built with **Streamlit**. This app allows users to transform images into oil paintings, analyze shapes and colors, generate color mixing recipes, and more — tailored for digital artists, designers, and AI/graphics enthusiasts.

---

## 📦 Features

### 🖼️ Image Generator

* Generate shape-based artwork (triangles, rectangles, circles)
* Control shape count, size, and type

### 🔍 Shape Detector & Analyzer

* Detect geometric shapes from images
* Analyze color distribution and extract grouped color data
* Generate painting recipes from detected colors

### 🎨 Oil Painting Generator

* Apply realistic oil paint effects using OpenCV
* Adjustable painting intensity
* Download final stylized images

### 🌈 Color Mixer

* Mix multiple RGB values with adjustable weights
* Visualize mixed colors in real-time
* Useful for digital artists and pigment match simulation

### 🧪 Paint Recipe Generator

* Match target colors with paint recipes using your custom color databases
* Generate up to 3 best-fit recipes based on precision

### 📚 Color Database Manager

* Create, browse, and manage named color databases
* Add or remove custom colors
* Organize into separate themes or palettes

---

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── color.txt              # Color database definitions
├── EnDe.py                # Encode/decode logic for shapes and images
├── painterfun.py          # Oil paint effect implementation
├── shape_art_generator/   # Foogle Man's shape art utilities
├── geometrize.py          # Geometrization extension module
```

---

## ⚙️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/artistic-vision-studio.git
cd artistic-vision-studio
```

2. **Install required dependencies:**
   Create a virtual environment and install from `requirements.txt`

```bash
pip install -r requirements.txt
```

> **requirements.txt**

```txt
streamlit
pillow
numpy
opencv-python-headless
threadpool
matplotlib
scipy
pymixbox
torch
tqdm
numba
```

3. **Run the application:**

```bash
streamlit run app.py
```

---

## 📜 Citing Original Oil Painting Work

This app builds on concepts inspired by the ACM MM 2022 paper:

> Tong, Zhengyan et al. *Im2Oil: Stroke-Based Oil Painting Rendering with Linearly Controllable Fineness Via Adaptive Sampling*. ACM Multimedia 2022. [DOI:10.1145/3503161.3547759](https://doi.org/10.1145/3503161.3547759)

```
@inproceedings{10.1145/3503161.3547759,
author = {Tong, Zhengyan and Wang, Xiaohang and Yuan, Shengchao and Chen, Xuanhong and Wang, Junjie and Fang, Xiangzhong},
title = {Im2Oil: Stroke-Based Oil Painting Rendering with Linearly Controllable Fineness Via Adaptive Sampling},
year = {2022},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {1035–1046},
doi = {10.1145/3503161.3547759}
```

---

## 🤝 Acknowledgements

* Stability AI for image generation research
* OpenCV for image transformation
* pymixbox and mixbox for color blending
* Streamlit for the interactive UI framework

---

## 📬 Contact & Contributions

If you'd like to contribute, improve, or integrate this app into your work:

* Fork the repo
* Create a branch
* Submit a pull request 🚀

---

MIT License
