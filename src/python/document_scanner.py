"""
Document Scanner - Python Implementation
A Python version of the document scanner using OpenCV and NumPy
Author: Mangesh Raut
Course: CV583 - Computer Vision
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class DocumentScanner:
    """
    A class for detecting and rectifying documents in images.
    Implements edge detection, Hough Transform, and homography-based rectification.
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the scanner with an input image.
        
        Args:
            image_path: Path to the input image file
        """
        self.input_image = cv2.imread(image_path)
        if self.input_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        self.edge_image = None
        self.hough_transform = None
        self.detected_lines = None
        self.corner_points = None
        self.rectified_image = None
        
        # Parameters
        self.sigma = 1.0
        self.edge_threshold = 290
        self.num_peaks = 6
        self.peak_threshold = 0.5
    
    def detect_edges(self) -> 'DocumentScanner':
        """
        Detect edges using Gaussian smoothing and Sobel operator.
        
        Returns:
            Self for method chaining
        """
        # Gaussian smoothing
        kernel_size = 2 * int(3 * self.sigma) + 1
        smoothed = cv2.GaussianBlur(self.gray_image, (kernel_size, kernel_size), self.sigma)
        
        # Sobel edge detection
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold
        self.edge_image = (grad_mag > self.edge_threshold).astype(np.uint8) * 255
        
        return self
    
    def compute_hough_transform(self) -> 'DocumentScanner':
        """
        Compute Hough Transform for line detection.
        
        Returns:
            Self for method chaining
        """
        # Define parameter ranges
        theta_range = np.arange(0, 181, 1)
        height, width = self.edge_image.shape
        rho_max = int(np.sqrt(height**2 + width**2))
        rho_range = np.arange(-rho_max, rho_max + 1, 1)
        
        # Initialize accumulator
        H = np.zeros((len(rho_range), len(theta_range)), dtype=np.uint64)
        
        # Find edge pixels
        edge_pixels = np.argwhere(self.edge_image > 0)
        
        # Accumulate votes
        for y, x in edge_pixels:
            for theta_idx, theta_deg in enumerate(theta_range):
                theta = np.deg2rad(theta_deg)
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                rho_idx = rho + rho_max
                if 0 <= rho_idx < len(rho_range):
                    H[rho_idx, theta_idx] += 1
        
        self.hough_transform = {
            'H': H,
            'theta_range': theta_range,
            'rho_range': rho_range,
            'rho_max': rho_max
        }
        
        return self
    
    def detect_lines(self) -> 'DocumentScanner':
        """
        Detect lines from Hough Transform peaks.
        
        Returns:
            Self for method chaining
        """
        H = self.hough_transform['H'].copy()
        theta_range = self.hough_transform['theta_range']
        rho_range = self.hough_transform['rho_range']
        
        # Find peaks
        peaks = []
        thresh = self.peak_threshold * np.max(H)
        
        for _ in range(self.num_peaks):
            max_val = np.max(H)
            if max_val < thresh:
                break
            
            max_idx = np.argmax(H)
            rho_idx, theta_idx = np.unravel_index(max_idx, H.shape)
            peaks.append((rho_idx, theta_idx))
            
            # Suppress neighboring bins
            rho_start = max(0, rho_idx - 10)
            rho_end = min(H.shape[0], rho_idx + 11)
            theta_start = max(0, theta_idx - 10)
            theta_end = min(H.shape[1], theta_idx + 11)
            H[rho_start:rho_end, theta_start:theta_end] = 0
        
        self.detected_lines = {
            'peaks': peaks,
            'theta_range': theta_range,
            'rho_range': rho_range
        }
        
        return self
    
    def find_corners(self, manual_corners: Optional[np.ndarray] = None) -> 'DocumentScanner':
        """
        Find corner points (manual or automatic).
        
        Args:
            manual_corners: Optional array of shape (4, 2) with corner coordinates
            
        Returns:
            Self for method chaining
        """
        if manual_corners is not None:
            self.corner_points = np.array(manual_corners, dtype=np.float32)
        else:
            raise NotImplementedError("Automatic corner detection not yet implemented")
        
        return self
    
    def rectify_image(self, output_width: int = None, output_height: int = None) -> 'DocumentScanner':
        """
        Rectify the document using homography.
        
        Args:
            output_width: Width of output image (default: 8.5" at 300 DPI)
            output_height: Height of output image (default: 11" at 300 DPI)
            
        Returns:
            Self for method chaining
        """
        if output_width is None:
            output_width = int(8.5 * 300)
        if output_height is None:
            output_height = int(11 * 300)
        
        # Define rectified corner points
        corners_rect = np.array([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ], dtype=np.float32)
        
        # Compute homography
        H = cv2.getPerspectiveTransform(self.corner_points, corners_rect)
        
        # Warp image
        self.rectified_image = cv2.warpPerspective(
            self.input_image, H, (output_width, output_height)
        )
        
        return self
    
    def visualize(self, step: str, save_path: Optional[str] = None):
        """
        Visualize results at different processing steps.
        
        Args:
            step: Which step to visualize ('edges', 'hough', 'lines', 'corners', 'rectified')
            save_path: Optional path to save the visualization
        """
        if step == 'edges':
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(self.edge_image, cmap='gray')
            axes[1].set_title('Edge Detection')
            axes[1].axis('off')
            
        elif step == 'hough':
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(self.hough_transform['H'], 
                          extent=[0, 180, 
                                 self.hough_transform['rho_range'][0],
                                 self.hough_transform['rho_range'][-1]],
                          aspect='auto', cmap='hot')
            ax.set_xlabel('Theta (degrees)')
            ax.set_ylabel('Rho (pixels)')
            ax.set_title('Hough Transform')
            plt.colorbar(im, ax=ax)
            
        elif step == 'lines':
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(self.edge_image, cmap='gray')
            self._draw_detected_lines(ax)
            ax.set_title('Detected Lines')
            ax.axis('off')
            
        elif step == 'corners':
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB))
            
            # Draw corners
            ax.plot(self.corner_points[:, 0], self.corner_points[:, 1], 
                   'ro', markersize=10, linewidth=2)
            
            # Draw lines between corners
            for i in range(4):
                next_i = (i + 1) % 4
                ax.plot([self.corner_points[i, 0], self.corner_points[next_i, 0]],
                       [self.corner_points[i, 1], self.corner_points[next_i, 1]],
                       'r-', linewidth=2)
            
            ax.set_title('Detected Corners')
            ax.axis('off')
            
        elif step == 'rectified':
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(cv2.cvtColor(self.rectified_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Rectified Image')
            axes[1].axis('off')
        
        else:
            raise ValueError(f"Unknown visualization step: {step}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def _draw_detected_lines(self, ax):
        """Helper function to draw detected lines on an axis."""
        peaks = self.detected_lines['peaks']
        theta_range = self.detected_lines['theta_range']
        rho_range = self.detected_lines['rho_range']
        
        height, width = self.gray_image.shape
        
        for rho_idx, theta_idx in peaks:
            rho = rho_range[rho_idx]
            theta = np.deg2rad(theta_range[theta_idx])
            
            if abs(np.sin(theta)) > 0.1:
                # Line is not too vertical
                m = -np.cos(theta) / np.sin(theta)
                b = rho / np.sin(theta)
                x1, y1 = 0, b
                x2, y2 = width, m * width + b
            else:
                # Line is vertical
                x1 = x2 = rho / np.cos(theta)
                y1, y2 = 0, height
            
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
    
    def save_rectified(self, output_path: str):
        """
        Save the rectified image to a file.
        
        Args:
            output_path: Path where to save the image
        """
        if self.rectified_image is None:
            raise ValueError("No rectified image to save. Run rectify_image() first.")
        
        cv2.imwrite(output_path, self.rectified_image)


if __name__ == "__main__":
    # Example usage
    print("Document Scanner - Python Implementation")
    print("=" * 50)
    
    # Initialize scanner
    scanner = DocumentScanner("../../examples/input1.png")
    
    # Process image
    scanner.detect_edges()
    scanner.compute_hough_transform()
    scanner.detect_lines()
    
    # Manual corners for image 1
    corners = np.array([[140, 81], [410, 93], [400, 473], [24, 411]], dtype=np.float32)
    scanner.find_corners(corners)
    
    # Rectify
    scanner.rectify_image()
    
    # Visualize results
    scanner.visualize('edges', '../../outputs/edges/edges_python.png')
    scanner.visualize('hough', '../../outputs/hough/hough_python.png')
    scanner.visualize('lines', '../../outputs/lines/lines_python.png')
    scanner.visualize('corners', '../../outputs/lines/corners_python.png')
    scanner.visualize('rectified', '../../outputs/rectified/rectified_python.png')
    
    # Save final output
    scanner.save_rectified('../../outputs/rectified/final_output_python.png')
    
    print("\nProcessing complete!")
