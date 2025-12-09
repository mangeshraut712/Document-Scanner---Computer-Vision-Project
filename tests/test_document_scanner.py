"""
Unit tests for Document Scanner
"""

import pytest
import numpy as np
import cv2
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from document_scanner import DocumentScanner


class TestDocumentScanner:
    """Test suite for DocumentScanner class"""
    
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create a sample test image"""
        # Create a simple test image
        img = np.ones((500, 400, 3), dtype=np.uint8) * 255
        # Draw a rectangle (document)
        cv2.rectangle(img, (50, 50), (350, 450), (0, 0, 0), 2)
        
        img_path = tmp_path / "test_image.png"
        cv2.imwrite(str(img_path), img)
        return str(img_path)
    
    def test_initialization(self, sample_image_path):
        """Test scanner initialization"""
        scanner = DocumentScanner(sample_image_path)
        
        assert scanner.input_image is not None
        assert scanner.gray_image is not None
        assert scanner.input_image.shape[2] == 3  # RGB
        assert len(scanner.gray_image.shape) == 2  # Grayscale
    
    def test_invalid_image_path(self):
        """Test initialization with invalid path"""
        with pytest.raises(ValueError):
            DocumentScanner("nonexistent_image.png")
    
    def test_edge_detection(self, sample_image_path):
        """Test edge detection"""
        scanner = DocumentScanner(sample_image_path)
        scanner.detect_edges()
        
        assert scanner.edge_image is not None
        assert scanner.edge_image.dtype == np.uint8
        assert scanner.edge_image.shape == scanner.gray_image.shape
        # Should have some edges detected
        assert np.sum(scanner.edge_image > 0) > 0
    
    def test_hough_transform(self, sample_image_path):
        """Test Hough Transform computation"""
        scanner = DocumentScanner(sample_image_path)
        scanner.detect_edges()
        scanner.compute_hough_transform()
        
        assert scanner.hough_transform is not None
        assert 'H' in scanner.hough_transform
        assert 'theta_range' in scanner.hough_transform
        assert 'rho_range' in scanner.hough_transform
        
        # Accumulator should be 2D
        H = scanner.hough_transform['H']
        assert len(H.shape) == 2
    
    def test_line_detection(self, sample_image_path):
        """Test line detection"""
        scanner = DocumentScanner(sample_image_path)
        scanner.detect_edges()
        scanner.compute_hough_transform()
        scanner.detect_lines()
        
        assert scanner.detected_lines is not None
        assert 'peaks' in scanner.detected_lines
        # Should detect some lines
        assert len(scanner.detected_lines['peaks']) > 0
    
    def test_corner_detection_manual(self, sample_image_path):
        """Test manual corner detection"""
        scanner = DocumentScanner(sample_image_path)
        
        corners = np.array([
            [50, 50],
            [350, 50],
            [350, 450],
            [50, 450]
        ], dtype=np.float32)
        
        scanner.find_corners(corners)
        
        assert scanner.corner_points is not None
        assert scanner.corner_points.shape == (4, 2)
        np.testing.assert_array_equal(scanner.corner_points, corners)
    
    def test_corner_detection_automatic_not_implemented(self, sample_image_path):
        """Test that automatic corner detection raises NotImplementedError"""
        scanner = DocumentScanner(sample_image_path)
        
        with pytest.raises(NotImplementedError):
            scanner.find_corners()
    
    def test_rectification(self, sample_image_path):
        """Test image rectification"""
        scanner = DocumentScanner(sample_image_path)
        
        corners = np.array([
            [50, 50],
            [350, 50],
            [350, 450],
            [50, 450]
        ], dtype=np.float32)
        
        scanner.find_corners(corners)
        scanner.rectify_image(400, 500)
        
        assert scanner.rectified_image is not None
        assert scanner.rectified_image.shape == (500, 400, 3)
    
    def test_method_chaining(self, sample_image_path):
        """Test that methods can be chained"""
        scanner = DocumentScanner(sample_image_path)
        
        result = scanner.detect_edges().compute_hough_transform().detect_lines()
        
        assert result is scanner
        assert scanner.edge_image is not None
        assert scanner.hough_transform is not None
        assert scanner.detected_lines is not None
    
    def test_save_rectified(self, sample_image_path, tmp_path):
        """Test saving rectified image"""
        scanner = DocumentScanner(sample_image_path)
        
        corners = np.array([
            [50, 50],
            [350, 50],
            [350, 450],
            [50, 450]
        ], dtype=np.float32)
        
        scanner.find_corners(corners)
        scanner.rectify_image(400, 500)
        
        output_path = tmp_path / "rectified.png"
        scanner.save_rectified(str(output_path))
        
        assert output_path.exists()
        
        # Verify saved image
        saved_img = cv2.imread(str(output_path))
        assert saved_img is not None
        assert saved_img.shape == (500, 400, 3)
    
    def test_save_without_rectification(self, sample_image_path):
        """Test that saving without rectification raises error"""
        scanner = DocumentScanner(sample_image_path)
        
        with pytest.raises(ValueError):
            scanner.save_rectified("output.png")
    
    def test_parameter_modification(self, sample_image_path):
        """Test modifying scanner parameters"""
        scanner = DocumentScanner(sample_image_path)
        
        # Modify parameters
        scanner.sigma = 2.0
        scanner.edge_threshold = 150
        scanner.num_peaks = 8
        scanner.peak_threshold = 0.3
        
        # Run with modified parameters
        scanner.detect_edges()
        scanner.compute_hough_transform()
        scanner.detect_lines()
        
        assert scanner.edge_image is not None
        assert scanner.detected_lines is not None


class TestImageProcessing:
    """Test image processing utilities"""
    
    def test_grayscale_conversion(self):
        """Test RGB to grayscale conversion"""
        # Create a colored image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Red channel
        
        scanner = DocumentScanner.__new__(DocumentScanner)
        scanner.input_image = img
        scanner.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scanner.imageData = type('obj', (object,), {
            'width': 100,
            'height': 100,
            'data': img.flatten()
        })()
        
        gray = scanner.toGrayscale(scanner.imageData)
        
        assert gray is not None
        assert len(gray) == 100 * 100
        # Red should convert to specific grayscale value
        assert gray[0] > 0


def test_integration_full_pipeline(tmp_path):
    """Integration test for complete pipeline"""
    # Create test image
    img = np.ones((600, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 100), (400, 500), (0, 0, 0), 3)
    
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)
    
    # Run complete pipeline
    scanner = DocumentScanner(str(img_path))
    
    corners = np.array([
        [100, 100],
        [400, 100],
        [400, 500],
        [100, 500]
    ], dtype=np.float32)
    
    scanner.detect_edges() \
           .compute_hough_transform() \
           .detect_lines() \
           .find_corners(corners) \
           .rectify_image(300, 400)
    
    output_path = tmp_path / "output.png"
    scanner.save_rectified(str(output_path))
    
    assert output_path.exists()
    result = cv2.imread(str(output_path))
    assert result.shape == (400, 300, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
