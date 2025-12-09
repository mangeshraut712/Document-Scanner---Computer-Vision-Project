clc;
clear;
close all;



% Part 1: Edge Detection

% Load the image
im = imread('Final Project/input2.png');

% Convert to grayscale
im_gray = rgb2gray(im);

% Define a smoothing kernel
sigma = 1;
filt_size = 2 * ceil(3 * sigma) + 1;
G = fspecial('gaussian', filt_size, sigma);

% Apply the smoothing kernel
im_smooth = conv2(im_gray, G, 'same');

% Define an edge kernel
sobel_x = [-1 0 1; -2 0 2; -1 0 1];
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];

% Compute the gradient of the smoothed image
im_grad_x = conv2(im_smooth, sobel_x, 'same');
im_grad_y = conv2(im_smooth, sobel_y, 'same');

% Compute the magnitude and direction of the gradient
im_grad_mag = sqrt(im_grad_x.^2 + im_grad_y.^2);
im_grad_dir = atan2(im_grad_y, im_grad_x);

% Threshold the gradient magnitude to obtain edge pixels
thresh = 290;
im_edges = im_grad_mag > thresh;

figure;
subplot(1, 2, 1);
imshow(im);
title('Original Image');
subplot(1, 2, 2);
imshow(im_edges);
title('Edge Image');





% Part 2: Hough Transform for Line Detection

% Define range of theta and rho values
theta_range = 0:180;
rho_max = round(sqrt(size(im_edges, 1)^2 + size(im_edges, 2)^2));
rho_range = -rho_max:rho_max;

% Initialize Hough accumulator array
H = zeros(length(rho_range), length(theta_range));

% Loop over all edge pixels
[edge_rows, edge_cols] = find(im_edges);
num_edge_pixels = length(edge_rows);
for i = 1:num_edge_pixels
    x = edge_cols(i);
    y = edge_rows(i);
    % Loop over all theta values and compute associated rho value
    for theta_idx = 1:length(theta_range)
        theta = deg2rad(theta_range(theta_idx));
        rho = round(x * cos(theta) + y * sin(theta));
        rho_idx = rho + rho_max + 1;
        H(rho_idx, theta_idx) = H(rho_idx, theta_idx) + 1;
    end
end

figure;
imshow(H,[],'XData',theta_range,'YData',rho_range,'InitialMagnification','fit');
title('Hough Transform');





% Part 3: Relevant Line Identification

% Find peaks in Hough Transform
peaks = [];
num_peaks = 6;
thresh = 0.5 * max(H(:)); % set a threshold for peak detection

while size(peaks, 1) < num_peaks
    [max_val, max_idx] = max(H(:));
    if max_val < thresh
        break;
    end
    [rho_idx, theta_idx] = ind2sub(size(H), max_idx);
    peaks = [peaks; rho_idx, theta_idx];
    % Suppress neighboring bins
    rho_start = max(1, rho_idx - 1);
    rho_end = min(size(H, 1), rho_idx + 1);
    theta_start = max(1, theta_idx - 1);
    theta_end = min(size(H, 2), theta_idx + 1);
    H(rho_start:rho_end, theta_start:theta_end) = 0;
end

% Superimpose peaks on the Hough transform
figure;
imshow(H,[],'XData',theta_range,'YData',rho_range,'InitialMagnification','fit');
title('Hough Transform with Detected Peaks');
hold on;
for i = 1:size(peaks, 1)
    iPeak = peaks(i, 1);
    jPeak = peaks(i, 2);
    plot(theta_range(jPeak), rho_range(iPeak), 'rx', 'MarkerSize', 5, 'LineWidth', 2);
end
hold off;

% Choose the number of clusters
k = 4;

% Randomly select k peaks as initial cluster centers
rng(0); % For reproducibility
cluster_centers = peaks(randperm(size(peaks, 1), k), :);

% Perform k-means clustering
clusters = zeros(num_peaks, 1);
for iter = 1:10
    % Assign each peak to the nearest cluster center
    for i = 1:num_peaks
        distances = sqrt(sum((peaks(i, :) - cluster_centers).^2, 2));
        [~, cluster_idx] = min(distances);
        clusters(i) = cluster_idx;
    end

    % Update the cluster centers
    for j = 1:k
        cluster_points = peaks(clusters == j, :);
        if ~isempty(cluster_points)
            cluster_centers(j, :) = mean(cluster_points);
        end
    end
end

% Plot lines grouped by cluster
figure;
imshow(im_edges);
hold on;

for i = 1:num_peaks
    rho_idx = peaks(i, 1);
    theta_idx = peaks(i, 2);
    rho = rho_range(rho_idx);
    theta = deg2rad(theta_range(theta_idx));
    if abs(theta) < pi/4
        % Line is closer to horizontal
        m = -cos(theta) / sin(theta);
        b = rho / sin(theta);
        x1 = 1;
        y1 = m * x1 + b;
        x2 = size(im_gray, 2);
        y2 = m * x2 + b;
    else
        % Line is closer to vertical
        m = -sin(theta) / cos(theta);
        b = rho / cos(theta);
        y1 = 1;
        x1 = m * y1 + b;
        y2 = size(im_gray, 1);
        x2 = m * y2 + b;
    end
    line([x1, x2], [y1, y2], 'Color', 'red', 'LineWidth', 2);
end

hold off;





% Part 4: Line Intersections

% Store the provided corner points
corners = [34, 230; 570, 148; 730, 858; 136, 984]; % for image 2


% Draw lines using the corner points
figure;
imshow(im_edges);
hold on;
colors = ['r', 'g', 'b', 'y'];
for i = 1:size(corners, 1)
    if i < size(corners, 1)
        next_point = corners(i+1, :);
    else
        next_point = corners(1, :);
    end
    line([corners(i, 1), next_point(1)], [corners(i, 2), next_point(2)], 'Color', colors(i), 'LineWidth', 2);
end
hold off;
title('Detected Lines');





% Part 5: Image Rectification

% Define the coordinates of the four corner points in the rectified image
corner_points_rectified = [0, 0; 8.5*300, 0; 8.5*300, 11*300; 0, 11*300];

% Compute the homography matrix using the four corner correspondences
A = [];
for i = 1:4
    x_src = corners(i, 1);
    y_src = corners(i, 2);
    x_dst = corner_points_rectified(i, 1);
    y_dst = corner_points_rectified(i, 2);

    A = [A; -x_src, -y_src, -1, 0, 0, 0, x_src*x_dst, y_src*x_dst, x_dst;
             0, 0, 0, -x_src, -y_src, -1, x_src*y_dst, y_src*y_dst, y_dst];
end

[U, S, V] = svd(A);
H = reshape(V(:, end), 3, 3)';

% Inverse homography transformation to map points from rectified image to original image
H_inv = inv(H);

% Create a blank rectified image
rectified_width = 8.5*300; % Assuming 300 dpi resolution for the rectified image
rectified_height = 11*300;
rectified_image = uint8(zeros(rectified_height, rectified_width, size(im, 3)));

% Iterate over each location in the rectified image
for y = 1:rectified_height
    for x = 1:rectified_width
        % Compute the corresponding location in the original image
        point_rectified = [x, y, 1]';
        point_original = H_inv * point_rectified;
        point_original = point_original ./ point_original(3);

        % Copy the pixel at the corresponding location to the rectified image
        if point_original(1) >= 1 && point_original(1) <= size(im, 2) && ...
                point_original(2) >= 1 && point_original(2) <= size(im, 1)
            rectified_image(y, x, :) = im(round(point_original(2)), round(point_original(1)), :);
        end
    end
end

% Display the rectified image
figure;
subplot(1,2,1); imshow(im); title('Original image');
subplot(1,2,2); imshow(rectified_image); title('Rectified image');
