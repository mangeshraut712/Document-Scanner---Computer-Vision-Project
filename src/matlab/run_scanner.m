% Document Scanner -
    Main Script %
        This script demonstrates the usage of the DocumentScanner class %
        Author : Mangesh Raut % Course : CV583 -
    Computer Vision

        clc;
clear;
close all;

% % Configuration % Path to input image(relative to project root) imagePath =
    '../../examples/input1.png';

% Manual corner points for image 1
% Format: [x, y] for each corner
corners_image1 = [140, 81; 410, 93; 400, 473; 24, 411];

% Output dimensions(8.5 x 11 inch at 300 DPI) outputWidth = round(8.5 * 300);
outputHeight = round(11 * 300);

% % Step 1 : Initialize Scanner fprintf('Initializing Document Scanner...\n');
scanner = DocumentScanner(imagePath);

% % Step 2 : Edge Detection fprintf('Detecting edges...\n');
scanner = scanner.detectEdges();
scanner.visualize('edges');
saveas(gcf, '../../outputs/edges/edge_detection.png');

% % Step 3 : Hough Transform fprintf('Computing Hough Transform...\n');
scanner = scanner.computeHoughTransform();
scanner.visualize('hough');
saveas(gcf, '../../outputs/hough/hough_transform.png');

% % Step 4 : Line Detection fprintf('Detecting lines...\n');
scanner = scanner.detectLines();
scanner.visualize('lines');
saveas(gcf, '../../outputs/lines/detected_lines.png');

% % Step 5 : Corner Detection fprintf('Finding corners...\n');
scanner = scanner.findCorners(corners_image1);
scanner.visualize('corners');
saveas(gcf, '../../outputs/lines/corners.png');

% % Step 6 : Image Rectification fprintf('Rectifying image...\n');
scanner = scanner.rectifyImage(outputWidth, outputHeight);
scanner.visualize('rectified');
saveas(gcf, '../../outputs/rectified/rectified_image.png');

% % Save final output fprintf('Saving rectified image...\n');
imwrite(scanner.rectifiedImage, '../../outputs/rectified/final_output.png');

fprintf('\nDocument scanning complete!\n');
fprintf('Results saved to outputs/ directory\n');
