<div align="center">

# ◈ Document Scanner

**Advanced document scanning using computer vision**

*Edge Detection • Hough Transform • Image Rectification*

[![CV583](https://img.shields.io/badge/Course-CV583-000000.svg?style=flat-square)](https://github.com)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023-000000.svg?style=flat-square)](https://www.mathworks.com)
[![Python](https://img.shields.io/badge/Python-3.8+-000000.svg?style=flat-square)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-000000.svg?style=flat-square)](LICENSE)

[Demo](web/index.html) · [Quick Start](#-quick-start) · [Documentation](#-documentation) · [Algorithm](#-algorithm)

</div>

---

## Overview

A comprehensive document scanning application that transforms photographs of documents into clean, rectified images. This project demonstrates fundamental computer vision concepts through three implementations: **MATLAB**, **Python**, and an interactive **Web Demo**.

### Features

- **Edge Detection** — Gaussian smoothing with Sobel operator
- **Hough Transform** — Parameter space voting for line detection  
- **Corner Detection** — Interactive or automatic point selection
- **Homography** — Perspective transformation for rectification
- **Multi-Platform** — MATLAB, Python, and Web implementations
- **Modern UI** — Minimalist design inspired by shadcn/ui and Japanese aesthetics

---

## 🚀 Quick Start

### Web Demo (No Installation)

```bash
cd web && open index.html
```

### Python

```bash
pip install -r requirements.txt
cd src/python && python document_scanner.py
```

### MATLAB

```matlab
cd src/matlab
run_scanner
```

---

## 📁 Project Structure

```
document-scanner/
├── src/
│   ├── matlab/                    # MATLAB implementation
│   │   ├── DocumentScanner.m      # Main scanner class
│   │   ├── run_scanner.m          # Example script
│   │   └── [legacy scripts]
│   └── python/                    # Python implementation
│       └── document_scanner.py    # OpenCV-based scanner
├── web/                           # Interactive web demo
│   ├── index.html                 # Main page
│   ├── styles.css                 # Minimalist design
│   └── script.js                  # Processing logic
├── examples/                      # Sample input images
│   ├── input1.png
│   └── input2.png
├── outputs/                       # Processing results
├── tests/                         # Test suite
│   └── test_document_scanner.py
├── docs/                          # Documentation
├── .github/workflows/             # CI/CD
├── requirements.txt               # Python dependencies
└── README.md
```

---

## � Documentation

### Python Usage

```python
from document_scanner import DocumentScanner
import numpy as np

# Initialize and process
scanner = DocumentScanner('path/to/image.png')
scanner.detect_edges()
scanner.compute_hough_transform()
scanner.detect_lines()

# Set corners (top-left, top-right, bottom-right, bottom-left)
corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
scanner.find_corners(corners)

# Rectify and save
scanner.rectify_image()
scanner.save_rectified('output.png')
```

### MATLAB Usage

```matlab
% Initialize scanner
scanner = DocumentScanner('path/to/image.png');

% Process
scanner = scanner.detectEdges();
scanner = scanner.computeHoughTransform();
scanner = scanner.detectLines();

% Set corners and rectify
corners = [140, 81; 410, 93; 400, 473; 24, 411];
scanner = scanner.findCorners(corners);
scanner = scanner.rectifyImage();

% Visualize
scanner.visualize('rectified');
```

### Web Demo

1. Open `web/index.html` in a browser
2. Upload an image or select an example
3. Click **Process** to run edge detection
4. Click four corners on the image
5. Download the rectified result

---

## 🔬 Algorithm

### 1. Edge Detection

Gaussian smoothing followed by Sobel gradient computation:

```
G = √(Gx² + Gy²)
```

**Parameters:** σ = 1.0, threshold = adjustable

### 2. Hough Transform

Polar parameterization for line detection:

```
ρ = x·cos(θ) + y·sin(θ)
```

**Range:** θ ∈ [0°, 180°], ρ ∈ [-√(w²+h²), √(w²+h²)]

### 3. Homography

Direct Linear Transform (DLT) for perspective correction:

```
p' = H · p
```

**Output:** 8.5" × 11" at 300 DPI

---

## 🎨 Design

The web interface follows a minimalist design philosophy:

- **shadcn/ui** — Clean component design
- **Apple Design** — Smooth animations
- **Japanese Aesthetics** — Kanso (simplicity), Ma (space), Chōwa (harmony)

### Color Palette

| Token | Light | Dark |
|-------|-------|------|
| Background | `#ffffff` | `#09090b` |
| Foreground | `#09090b` | `#fafafa` |
| Muted | `#f4f4f5` | `#27272a` |
| Border | `#e4e4e7` | `#27272a` |

---

## 🧪 Testing

```bash
# Run Python tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/python
```

---

## � Performance

| Implementation | Edge Detection | Hough Transform | Total |
|---------------|----------------|-----------------|-------|
| MATLAB | ~0.5s | ~2.0s | ~5.5s |
| Python | ~0.3s | ~1.5s | ~3.8s |
| Web (JS) | ~0.8s | ~3.0s | ~5.3s |

*Tested on 512×512 image, MacBook Pro M1*

---

## 🔧 Configuration

### Edge Detection

```python
scanner.edge_threshold = 150  # Lower = more edges
```

### Output Size

```python
scanner.rectify_image(1275, 1650)  # Width, Height
```

### Hough Parameters

```python
scanner.num_peaks = 8        # Number of lines
scanner.peak_threshold = 0.3  # Detection threshold
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## � References

1. Canny, J. (1986). *A Computational Approach to Edge Detection*
2. Duda, R. O., & Hart, P. E. (1972). *Use of the Hough Transformation*
3. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry*

---

## � License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Author:** Mangesh Raut  
**Course:** CV583 — Computer Vision

[⬆ Back to top](#-document-scanner)

</div>
