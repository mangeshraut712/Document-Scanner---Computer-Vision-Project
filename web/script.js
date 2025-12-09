/**
 * Document Scanner - Interactive Web Demo
 * Modern JavaScript with smooth animations and enhanced UX
 * Author: Mangesh Raut | CV583 Final Project
 */

class DocumentScanner {
    constructor() {
        this.inputImage = null;
        this.imageData = null;
        this.edgeImage = null;
        this.corners = [];
        this.edgeThreshold = 100;
        this.houghAccumulator = null;
        
        this.init();
    }
    
    init() {
        this.setupAnimations();
        this.setupEventListeners();
        this.setupSmoothScroll();
    }
    
    // ===== Animations =====
    setupAnimations() {
        const observerOptions = {
            root: null,
            rootMargin: '0px',
            threshold: 0.1
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                }
            });
        }, observerOptions);
        
        document.querySelectorAll('[data-animate]').forEach(el => {
            observer.observe(el);
        });
    }
    
    setupSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
    
    // ===== Event Listeners =====
    setupEventListeners() {
        // File upload
        const uploadBox = document.getElementById('uploadBox');
        const imageInput = document.getElementById('imageInput');
        
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = 'var(--foreground)';
        });
        
        uploadBox.addEventListener('dragleave', () => {
            uploadBox.style.borderColor = 'var(--border)';
        });
        
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = 'var(--border)';
            if (e.dataTransfer.files.length) {
                this.loadImage(e.dataTransfer.files[0]);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.loadImage(e.target.files[0]);
            }
        });
        
        // Example buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const exampleNum = e.currentTarget.dataset.example;
                this.loadExampleImage(exampleNum);
            });
        });
        
        // Process button
        document.getElementById('processBtn').addEventListener('click', () => {
            this.processImage();
        });
        
        // Edge threshold slider
        const slider = document.getElementById('edgeThreshold');
        slider.addEventListener('input', (e) => {
            this.edgeThreshold = parseInt(e.target.value);
            document.getElementById('thresholdValue').textContent = this.edgeThreshold;
            if (this.imageData) {
                this.detectEdges();
            }
        });
        
        // Corner canvas interaction
        const cornerCanvas = document.getElementById('cornerCanvas');
        cornerCanvas.addEventListener('click', (e) => {
            this.handleCornerClick(e);
        });
        
        // Reset corners button
        document.getElementById('resetCorners').addEventListener('click', () => {
            this.resetCorners();
        });
        
        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadResult();
        });
    }
    
    // ===== Image Loading =====
    loadImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.inputImage = img;
                this.showPipeline();
                this.displayOriginalImage();
                this.resetCorners();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    loadExampleImage(num) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            this.inputImage = img;
            this.showPipeline();
            this.displayOriginalImage();
            this.setExampleCorners(num);
        };
        img.onerror = () => {
            console.error('Failed to load example image');
            alert('Could not load example image. Please upload your own image.');
        };
        img.src = `../examples/input${num}.png`;
    }
    
    setExampleCorners(num) {
        if (num === '1') {
            this.corners = [
                { x: 140, y: 81 },
                { x: 410, y: 93 },
                { x: 400, y: 473 },
                { x: 24, y: 411 }
            ];
        } else {
            this.corners = [
                { x: 180, y: 120 },
                { x: 520, y: 140 },
                { x: 500, y: 580 },
                { x: 100, y: 550 }
            ];
        }
        this.updateCornerDisplay();
    }
    
    showPipeline() {
        const pipeline = document.getElementById('pipeline');
        pipeline.style.display = 'block';
        pipeline.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    displayOriginalImage() {
        const canvas = document.getElementById('originalCanvas');
        const ctx = canvas.getContext('2d');
        
        // Scale down if too large
        const maxSize = 600;
        let width = this.inputImage.width;
        let height = this.inputImage.height;
        
        if (width > maxSize || height > maxSize) {
            const scale = Math.min(maxSize / width, maxSize / height);
            width *= scale;
            height *= scale;
        }
        
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(this.inputImage, 0, 0, width, height);
        
        this.imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    
    // ===== Image Processing =====
    processImage() {
        if (!this.inputImage) {
            alert('Please upload an image first!');
            return;
        }
        
        // Add loading state
        const btn = document.getElementById('processBtn');
        btn.disabled = true;
        btn.innerHTML = '<span class="loading-spinner"></span> Processing...';
        
        // Use setTimeout to allow UI to update
        setTimeout(() => {
            this.detectEdges();
            this.computeHoughTransform();
            this.detectLines();
            this.updateCornerDisplay();
            
            btn.disabled = false;
            btn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Process
            `;
        }, 50);
    }
    
    detectEdges() {
        const canvas = document.getElementById('edgeCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = this.imageData.width;
        canvas.height = this.imageData.height;
        
        // Convert to grayscale
        const gray = this.toGrayscale(this.imageData);
        
        // Apply Gaussian blur
        const blurred = this.gaussianBlur(gray, this.imageData.width, this.imageData.height);
        
        // Sobel edge detection
        const edges = this.sobelEdgeDetection(blurred, this.imageData.width, this.imageData.height);
        
        // Display edges
        const edgeImageData = ctx.createImageData(canvas.width, canvas.height);
        for (let i = 0; i < edges.length; i++) {
            const val = edges[i];
            edgeImageData.data[i * 4] = val;
            edgeImageData.data[i * 4 + 1] = val;
            edgeImageData.data[i * 4 + 2] = val;
            edgeImageData.data[i * 4 + 3] = 255;
        }
        
        ctx.putImageData(edgeImageData, 0, 0);
        this.edgeImage = edges;
    }
    
    toGrayscale(imageData) {
        const gray = new Uint8ClampedArray(imageData.width * imageData.height);
        for (let i = 0; i < gray.length; i++) {
            const r = imageData.data[i * 4];
            const g = imageData.data[i * 4 + 1];
            const b = imageData.data[i * 4 + 2];
            gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        }
        return gray;
    }
    
    gaussianBlur(data, width, height) {
        const result = new Uint8ClampedArray(data.length);
        const kernelSize = 5;
        const half = Math.floor(kernelSize / 2);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let sum = 0;
                let count = 0;
                
                for (let ky = -half; ky <= half; ky++) {
                    for (let kx = -half; kx <= half; kx++) {
                        const ny = y + ky;
                        const nx = x + kx;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            sum += data[ny * width + nx];
                            count++;
                        }
                    }
                }
                
                result[y * width + x] = Math.round(sum / count);
            }
        }
        
        return result;
    }
    
    sobelEdgeDetection(data, width, height) {
        const edges = new Uint8ClampedArray(data.length);
        
        const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
        
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let gx = 0;
                let gy = 0;
                
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = (y + ky) * width + (x + kx);
                        const kernelIdx = (ky + 1) * 3 + (kx + 1);
                        
                        gx += data[idx] * sobelX[kernelIdx];
                        gy += data[idx] * sobelY[kernelIdx];
                    }
                }
                
                const magnitude = Math.sqrt(gx * gx + gy * gy);
                edges[y * width + x] = magnitude > this.edgeThreshold ? 255 : 0;
            }
        }
        
        return edges;
    }
    
    computeHoughTransform() {
        if (!this.edgeImage) return;
        
        const width = this.imageData.width;
        const height = this.imageData.height;
        const maxRho = Math.sqrt(width * width + height * height);
        const thetaSteps = 180;
        const rhoSteps = Math.ceil(maxRho * 2);
        
        const accumulator = new Array(rhoSteps).fill(0).map(() => new Array(thetaSteps).fill(0));
        
        // Accumulate votes
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (this.edgeImage[y * width + x] > 0) {
                    for (let thetaIdx = 0; thetaIdx < thetaSteps; thetaIdx++) {
                        const theta = (thetaIdx * Math.PI) / 180;
                        const rho = x * Math.cos(theta) + y * Math.sin(theta);
                        const rhoIdx = Math.round(rho + maxRho);
                        
                        if (rhoIdx >= 0 && rhoIdx < rhoSteps) {
                            accumulator[rhoIdx][thetaIdx]++;
                        }
                    }
                }
            }
        }
        
        this.visualizeHoughTransform(accumulator);
        this.houghAccumulator = accumulator;
    }
    
    visualizeHoughTransform(accumulator) {
        const canvas = document.getElementById('houghCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = accumulator[0].length * 2;
        canvas.height = Math.min(accumulator.length, 400);
        
        // Find max value for normalization
        let maxVal = 0;
        for (let i = 0; i < accumulator.length; i++) {
            for (let j = 0; j < accumulator[i].length; j++) {
                maxVal = Math.max(maxVal, accumulator[i][j]);
            }
        }
        
        // Create heatmap
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const scaleY = accumulator.length / canvas.height;
        const scaleX = accumulator[0].length / (canvas.width / 2);
        
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const accY = Math.floor(y * scaleY);
                const accX = Math.floor((x / 2) * scaleX);
                
                if (accY < accumulator.length && accX < accumulator[0].length) {
                    const value = accumulator[accY][accX];
                    const normalized = (value / maxVal);
                    
                    const idx = (y * canvas.width + x) * 4;
                    // Grayscale heatmap
                    const intensity = Math.floor(normalized * 255);
                    imageData.data[idx] = intensity;
                    imageData.data[idx + 1] = intensity;
                    imageData.data[idx + 2] = intensity;
                    imageData.data[idx + 3] = 255;
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }
    
    detectLines() {
        if (!this.houghAccumulator) return;
        
        const canvas = document.getElementById('linesCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = this.imageData.width;
        canvas.height = this.imageData.height;
        
        // Draw edge image as background
        const edgeCanvas = document.getElementById('edgeCanvas');
        ctx.drawImage(edgeCanvas, 0, 0);
        
        // Find peaks in Hough space
        const peaks = this.findHoughPeaks(6);
        
        // Draw lines
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        
        const maxRho = Math.sqrt(canvas.width * canvas.width + canvas.height * canvas.height);
        
        for (const peak of peaks) {
            const theta = (peak.theta * Math.PI) / 180;
            const rho = peak.rho - maxRho;
            
            if (Math.abs(Math.sin(theta)) > 0.1) {
                const m = -Math.cos(theta) / Math.sin(theta);
                const b = rho / Math.sin(theta);
                
                ctx.beginPath();
                ctx.moveTo(0, b);
                ctx.lineTo(canvas.width, m * canvas.width + b);
                ctx.stroke();
            } else {
                const x = rho / Math.cos(theta);
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
        }
    }
    
    findHoughPeaks(numPeaks) {
        const accumulator = this.houghAccumulator.map(row => [...row]);
        const peaks = [];
        
        let maxVal = 0;
        for (let i = 0; i < accumulator.length; i++) {
            for (let j = 0; j < accumulator[i].length; j++) {
                maxVal = Math.max(maxVal, accumulator[i][j]);
            }
        }
        
        const threshold = maxVal * 0.5;
        
        for (let i = 0; i < numPeaks; i++) {
            let max = 0;
            let maxRho = 0;
            let maxTheta = 0;
            
            for (let rho = 0; rho < accumulator.length; rho++) {
                for (let theta = 0; theta < accumulator[rho].length; theta++) {
                    if (accumulator[rho][theta] > max) {
                        max = accumulator[rho][theta];
                        maxRho = rho;
                        maxTheta = theta;
                    }
                }
            }
            
            if (max < threshold) break;
            
            peaks.push({ rho: maxRho, theta: maxTheta, value: max });
            
            // Suppress neighborhood
            for (let dr = -10; dr <= 10; dr++) {
                for (let dt = -10; dt <= 10; dt++) {
                    const r = maxRho + dr;
                    const t = maxTheta + dt;
                    if (r >= 0 && r < accumulator.length && t >= 0 && t < accumulator[0].length) {
                        accumulator[r][t] = 0;
                    }
                }
            }
        }
        
        return peaks;
    }
    
    // ===== Corner Selection =====
    handleCornerClick(e) {
        if (this.corners.length >= 4) {
            this.resetCorners();
        }
        
        const canvas = document.getElementById('cornerCanvas');
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        this.corners.push({ x, y });
        this.updateCornerDisplay();
        
        if (this.corners.length === 4) {
            this.rectifyImage();
        }
    }
    
    updateCornerDisplay() {
        const canvas = document.getElementById('cornerCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = this.inputImage.width;
        canvas.height = this.inputImage.height;
        
        // Scale down if too large
        const maxSize = 600;
        let width = this.inputImage.width;
        let height = this.inputImage.height;
        
        if (width > maxSize || height > maxSize) {
            const scale = Math.min(maxSize / width, maxSize / height);
            width *= scale;
            height *= scale;
            canvas.width = width;
            canvas.height = height;
        }
        
        ctx.drawImage(this.inputImage, 0, 0, width, height);
        
        // Scale corners
        const scaleX = width / this.inputImage.width;
        const scaleY = height / this.inputImage.height;
        
        // Draw corners and lines
        ctx.fillStyle = '#000000';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < this.corners.length; i++) {
            const corner = this.corners[i];
            const x = corner.x * scaleX;
            const y = corner.y * scaleY;
            
            // Draw point
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, 2 * Math.PI);
            ctx.fill();
            
            // White border
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw line to previous point
            if (i > 0) {
                const prevX = this.corners[i - 1].x * scaleX;
                const prevY = this.corners[i - 1].y * scaleY;
                ctx.beginPath();
                ctx.strokeStyle = '#000000';
                ctx.moveTo(prevX, prevY);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        }
        
        // Close the shape
        if (this.corners.length === 4) {
            const firstX = this.corners[0].x * scaleX;
            const firstY = this.corners[0].y * scaleY;
            const lastX = this.corners[3].x * scaleX;
            const lastY = this.corners[3].y * scaleY;
            ctx.beginPath();
            ctx.strokeStyle = '#000000';
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(firstX, firstY);
            ctx.stroke();
        }
        
        // Update counter
        document.getElementById('cornerCount').textContent = `${this.corners.length}/4`;
    }
    
    resetCorners() {
        this.corners = [];
        if (this.inputImage) {
            this.updateCornerDisplay();
        }
        document.getElementById('cornerCount').textContent = '0/4';
        document.getElementById('downloadBtn').style.display = 'none';
    }
    
    // ===== Rectification =====
    rectifyImage() {
        const outputWidth = Math.round(8.5 * 100);
        const outputHeight = Math.round(11 * 100);
        
        const canvas = document.getElementById('rectifiedCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = outputWidth;
        canvas.height = outputHeight;
        
        // Create source canvas
        const srcCanvas = document.createElement('canvas');
        const srcCtx = srcCanvas.getContext('2d');
        srcCanvas.width = this.inputImage.width;
        srcCanvas.height = this.inputImage.height;
        srcCtx.drawImage(this.inputImage, 0, 0);
        const srcData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
        
        const dstData = ctx.createImageData(outputWidth, outputHeight);
        
        // Bilinear interpolation mapping
        for (let y = 0; y < outputHeight; y++) {
            for (let x = 0; x < outputWidth; x++) {
                const u = x / outputWidth;
                const v = y / outputHeight;
                
                const srcX = Math.round(
                    (1 - u) * (1 - v) * this.corners[0].x +
                    u * (1 - v) * this.corners[1].x +
                    u * v * this.corners[2].x +
                    (1 - u) * v * this.corners[3].x
                );
                
                const srcY = Math.round(
                    (1 - u) * (1 - v) * this.corners[0].y +
                    u * (1 - v) * this.corners[1].y +
                    u * v * this.corners[2].y +
                    (1 - u) * v * this.corners[3].y
                );
                
                if (srcX >= 0 && srcX < srcCanvas.width && srcY >= 0 && srcY < srcCanvas.height) {
                    const srcIdx = (srcY * srcCanvas.width + srcX) * 4;
                    const dstIdx = (y * outputWidth + x) * 4;
                    
                    dstData.data[dstIdx] = srcData.data[srcIdx];
                    dstData.data[dstIdx + 1] = srcData.data[srcIdx + 1];
                    dstData.data[dstIdx + 2] = srcData.data[srcIdx + 2];
                    dstData.data[dstIdx + 3] = 255;
                }
            }
        }
        
        ctx.putImageData(dstData, 0, 0);
        document.getElementById('downloadBtn').style.display = 'flex';
    }
    
    downloadResult() {
        const canvas = document.getElementById('rectifiedCanvas');
        const link = document.createElement('a');
        link.download = 'rectified_document.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.scanner = new DocumentScanner();
});
