classdef DocumentScanner
    % DocumentScanner - A class for detecting and rectifying documents in images
    % This implements edge detection, Hough Transform, and homography-based rectification
    
    properties
        inputImage          % Original RGB image
        grayImage           % Grayscale version
        edgeImage           % Binary edge map
        houghTransform      % Hough accumulator array
        detectedLines       % Detected line parameters
        cornerPoints        % Four corner points of document
        rectifiedImage      % Final rectified output
        
        % Parameters
        sigma = 1;          % Gaussian smoothing sigma
        edgeThreshold = 290; % Edge detection threshold
        numPeaks = 6;       % Number of Hough peaks to detect
        peakThreshold = 0.5; % Hough peak threshold (fraction of max)
    end
    
    methods
        function obj = DocumentScanner(imagePath)
            % Constructor - load and initialize with an image
            obj.inputImage = imread(imagePath);
            obj.grayImage = rgb2gray(obj.inputImage);
        end
        
        function obj = detectEdges(obj)
            % Detect edges using Gaussian smoothing and Sobel operator
            
            % Create Gaussian smoothing kernel
            filt_size = 2 * ceil(3 * obj.sigma) + 1;
            G = fspecial('gaussian', filt_size, obj.sigma);
            
            % Apply smoothing
            im_smooth = conv2(obj.grayImage, G, 'same');
            
            % Define Sobel kernels
            sobel_x = [-1 0 1; -2 0 2; -1 0 1];
            sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
            
            % Compute gradients
            im_grad_x = conv2(im_smooth, sobel_x, 'same');
            im_grad_y = conv2(im_smooth, sobel_y, 'same');
            
            % Compute magnitude
            im_grad_mag = sqrt(im_grad_x.^2 + im_grad_y.^2);
            
            % Threshold to get binary edge map
            obj.edgeImage = im_grad_mag > obj.edgeThreshold;
        end
        
        function obj = computeHoughTransform(obj)
            % Compute Hough Transform for line detection
            
            % Define parameter ranges
            theta_range = 0:180;
            rho_max = round(sqrt(size(obj.edgeImage, 1)^2 + size(obj.edgeImage, 2)^2));
            rho_range = -rho_max:rho_max;
            
            % Initialize accumulator
            H = zeros(length(rho_range), length(theta_range));
            
            % Find edge pixels
            [edge_rows, edge_cols] = find(obj.edgeImage);
            num_edge_pixels = length(edge_rows);
            
            % Accumulate votes
            for i = 1:num_edge_pixels
                x = edge_cols(i);
                y = edge_rows(i);
                
                for theta_idx = 1:length(theta_range)
                    theta = deg2rad(theta_range(theta_idx));
                    rho = round(x * cos(theta) + y * sin(theta));
                    rho_idx = rho + rho_max + 1;
                    H(rho_idx, theta_idx) = H(rho_idx, theta_idx) + 1;
                end
            end
            
            obj.houghTransform = struct('H', H, 'theta_range', theta_range, ...
                                       'rho_range', rho_range, 'rho_max', rho_max);
        end
        
        function obj = detectLines(obj)
            % Detect lines from Hough Transform peaks
            
            H = obj.houghTransform.H;
            theta_range = obj.houghTransform.theta_range;
            rho_range = obj.houghTransform.rho_range;
            
            % Find peaks
            peaks = [];
            thresh = obj.peakThreshold * max(H(:));
            H_temp = H;
            
            while size(peaks, 1) < obj.numPeaks
                [max_val, max_idx] = max(H_temp(:));
                if max_val < thresh
                    break;
                end
                
                [rho_idx, theta_idx] = ind2sub(size(H_temp), max_idx);
                peaks = [peaks; rho_idx, theta_idx];
                
                % Suppress neighboring bins
                rho_start = max(1, rho_idx - 10);
                rho_end = min(size(H_temp, 1), rho_idx + 10);
                theta_start = max(1, theta_idx - 10);
                theta_end = min(size(H_temp, 2), theta_idx + 10);
                H_temp(rho_start:rho_end, theta_start:theta_end) = 0;
            end
            
            obj.detectedLines = struct('peaks', peaks, 'theta_range', theta_range, ...
                                      'rho_range', rho_range);
        end
        
        function obj = findCorners(obj, manualCorners)
            % Find corner points (manual or automatic)
            % For now, accepts manual corners as input
            % Future: implement automatic corner detection
            
            if nargin > 1
                obj.cornerPoints = manualCorners;
            else
                % Automatic corner detection would go here
                error('Automatic corner detection not yet implemented. Please provide manual corners.');
            end
        end
        
        function obj = rectifyImage(obj, outputWidth, outputHeight)
            % Rectify the document using homography
            
            if nargin < 3
                % Default to 8.5 x 11 inch at 300 DPI
                outputWidth = 8.5 * 300;
                outputHeight = 11 * 300;
            end
            
            % Define rectified corner points
            corners_rect = [0, 0; outputWidth, 0; outputWidth, outputHeight; 0, outputHeight];
            
            % Compute homography matrix
            H = obj.computeHomography(obj.cornerPoints, corners_rect);
            
            % Create blank rectified image
            rectified = uint8(zeros(outputHeight, outputWidth, size(obj.inputImage, 3)));
            
            % Inverse homography
            H_inv = inv(H);
            
            % Map each pixel
            for y = 1:outputHeight
                for x = 1:outputWidth
                    point_rect = [x; y; 1];
                    point_orig = H_inv * point_rect;
                    point_orig = point_orig ./ point_orig(3);
                    
                    x_orig = round(point_orig(1));
                    y_orig = round(point_orig(2));
                    
                    if x_orig >= 1 && x_orig <= size(obj.inputImage, 2) && ...
                       y_orig >= 1 && y_orig <= size(obj.inputImage, 1)
                        rectified(y, x, :) = obj.inputImage(y_orig, x_orig, :);
                    end
                end
            end
            
            obj.rectifiedImage = rectified;
        end
        
        function H = computeHomography(~, srcPoints, dstPoints)
            % Compute homography matrix from point correspondences
            
            A = [];
            for i = 1:4
                x_src = srcPoints(i, 1);
                y_src = srcPoints(i, 2);
                x_dst = dstPoints(i, 1);
                y_dst = dstPoints(i, 2);
                
                A = [A; -x_src, -y_src, -1, 0, 0, 0, x_src*x_dst, y_src*x_dst, x_dst;
                        0, 0, 0, -x_src, -y_src, -1, x_src*y_dst, y_src*y_dst, y_dst];
            end
            
            [~, ~, V] = svd(A);
            H = reshape(V(:, end), 3, 3)';
        end
        
        function visualize(obj, step)
            % Visualize results at different processing steps
            
            switch step
                case 'edges'
                    figure;
                    subplot(1, 2, 1);
                    imshow(obj.inputImage);
                    title('Original Image');
                    subplot(1, 2, 2);
                    imshow(obj.edgeImage);
                    title('Edge Detection');
                    
                case 'hough'
                    figure;
                    imshow(obj.houghTransform.H, [], 'XData', obj.houghTransform.theta_range, ...
                           'YData', obj.houghTransform.rho_range, 'InitialMagnification', 'fit');
                    title('Hough Transform');
                    xlabel('Theta (degrees)');
                    ylabel('Rho (pixels)');
                    colorbar;
                    
                case 'lines'
                    figure;
                    imshow(obj.edgeImage);
                    hold on;
                    obj.drawDetectedLines();
                    hold off;
                    title('Detected Lines');
                    
                case 'corners'
                    figure;
                    imshow(obj.inputImage);
                    hold on;
                    plot(obj.cornerPoints(:, 1), obj.cornerPoints(:, 2), 'ro', ...
                         'MarkerSize', 10, 'LineWidth', 2);
                    for i = 1:4
                        if i < 4
                            next_idx = i + 1;
                        else
                            next_idx = 1;
                        end
                        line([obj.cornerPoints(i, 1), obj.cornerPoints(next_idx, 1)], ...
                             [obj.cornerPoints(i, 2), obj.cornerPoints(next_idx, 2)], ...
                             'Color', 'r', 'LineWidth', 2);
                    end
                    hold off;
                    title('Detected Corners');
                    
                case 'rectified'
                    figure;
                    subplot(1, 2, 1);
                    imshow(obj.inputImage);
                    title('Original Image');
                    subplot(1, 2, 2);
                    imshow(obj.rectifiedImage);
                    title('Rectified Image');
                    
                otherwise
                    error('Unknown visualization step');
            end
        end
        
        function drawDetectedLines(obj)
            % Helper function to draw detected lines
            
            peaks = obj.detectedLines.peaks;
            theta_range = obj.detectedLines.theta_range;
            rho_range = obj.detectedLines.rho_range;
            
            [height, width] = size(obj.grayImage);
            
            for i = 1:size(peaks, 1)
                rho_idx = peaks(i, 1);
                theta_idx = peaks(i, 2);
                rho = rho_range(rho_idx);
                theta = deg2rad(theta_range(theta_idx));
                
                if abs(sin(theta)) > 0.1
                    % Line is not too vertical
                    m = -cos(theta) / sin(theta);
                    b = rho / sin(theta);
                    x1 = 1;
                    y1 = m * x1 + b;
                    x2 = width;
                    y2 = m * x2 + b;
                else
                    % Line is vertical
                    x1 = rho / cos(theta);
                    y1 = 1;
                    x2 = x1;
                    y2 = height;
                end
                
                line([x1, x2], [y1, y2], 'Color', 'red', 'LineWidth', 2);
            end
        end
    end
end
