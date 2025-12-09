# 🚀 Quick Start Guide

Get up and running with the Document Scanner in under 5 minutes!

## Choose Your Path

### 🌐 Web Demo (Easiest - No Installation!)

1. **Open the demo**:
   ```bash
   cd web
   open index.html
   ```

2. **Use it**:
   - Click "Choose Image" or try an example
   - Click "Process Image"
   - Select 4 corners by clicking on the image
   - Download your rectified document!

### 🐍 Python (Recommended)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the example**:
   ```bash
   cd src/python
   python document_scanner.py
   ```

3. **Use in your code**:
   ```python
   from document_scanner import DocumentScanner
   import numpy as np
   
   # Load and process
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

### 📊 MATLAB

1. **Open MATLAB** and navigate to the project directory

2. **Run the example**:
   ```matlab
   cd src/matlab
   run_scanner
   ```

3. **Use the class**:
   ```matlab
   % Initialize
   scanner = DocumentScanner('../../examples/input1.png');
   
   % Process
   scanner = scanner.detectEdges();
   scanner = scanner.computeHoughTransform();
   scanner = scanner.detectLines();
   
   % Set corners
   corners = [140, 81; 410, 93; 400, 473; 24, 411];
   scanner = scanner.findCorners(corners);
   
   % Rectify
   scanner = scanner.rectifyImage();
   
   % Visualize
   scanner.visualize('rectified');
   ```

## 🎯 Common Tasks

### Change Edge Detection Sensitivity

**Python**:
```python
scanner.edge_threshold = 150  # Lower = more edges
scanner.detect_edges()
```

**MATLAB**:
```matlab
scanner.edgeThreshold = 150;
scanner = scanner.detectEdges();
```

### Adjust Output Size

**Python**:
```python
scanner.rectify_image(1275, 1650)  # Width, Height in pixels
```

**MATLAB**:
```matlab
scanner = scanner.rectifyImage(1275, 1650);
```

### Detect More Lines

**Python**:
```python
scanner.num_peaks = 8
scanner.detect_lines()
```

**MATLAB**:
```matlab
scanner.numPeaks = 8;
scanner = scanner.detectLines();
```

## 📝 Example Workflow

```bash
# 1. Clone/download the project
cd "Final Project CV583"

# 2. Install Python dependencies (if using Python)
pip install -r requirements.txt

# 3. Try the web demo
cd web
open index.html

# 4. Or run Python version
cd ../src/python
python document_scanner.py

# 5. Run tests
cd ../../tests
pytest -v
```

## 🎨 Web Demo Features

- **Upload**: Drag & drop or click to upload
- **Examples**: Try pre-loaded sample images
- **Interactive**: Click to select corners
- **Real-time**: See each processing step
- **Download**: Save your results

## 🔧 Troubleshooting

### Python: "Module not found"
```bash
pip install -r requirements.txt
```

### MATLAB: "Function not found"
Make sure you're in the correct directory:
```matlab
cd src/matlab
```

### Web: Images not loading
Make sure you're opening from the `web/` directory and example images are in `examples/`.

### Tests failing
Install test dependencies:
```bash
pip install pytest
```

## 📚 Next Steps

- Read the full [README.md](README.md)
- Check out [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Explore the [examples/](examples/) directory
- Customize parameters for your use case

## 💡 Tips

1. **Start with examples**: Use the provided sample images first
2. **Adjust thresholds**: Different images may need different edge thresholds
3. **Good lighting**: Take photos with good, even lighting
4. **Flat surface**: Place documents on a contrasting background
5. **Clear edges**: Ensure all four edges of the document are visible

## 🆘 Need Help?

- Check the [README.md](README.md) for detailed documentation
- Look at example code in `src/`
- Open an issue on GitHub
- Review the algorithm explanations in the docs

---

**Ready to scan some documents? Let's go! 🚀**
