% Generate synthetic dataset using MATLAB's built-in functions
datasetType = 'circles'; % Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'

% Dataset parameters
samples = 100;
noise = 0.1;

% kNN parameters
k = 5;
h = 1;

% ------------------------------------------------------------------------------------------ %

[data1, labels1, data2, labels2] = generate_dataset(datasetType, samples, noise);
complete_set = [data1, labels1; data2, labels2];

% Determine the range of the data
min_x1 = min(complete_set(:, 1));
max_x1 = max(complete_set(:, 1));
min_x2 = min(complete_set(:, 2));
max_x2 = max(complete_set(:, 2));

% Generate a grid of points within the data range
x1 = linspace(min_x1 - 1, max_x1 + 1, 100);
x2 = linspace(min_x2 - 1, max_x2 + 1, 100);
[X1, X2] = meshgrid(x1, x2);
X = [X1(:), X2(:), zeros(length(X1(:)), 1)];

% Classify the points
for i = 1:size(X, 1)
  X(i, 3) = mykNN(X(i, 1:2), complete_set, k, h);
end

% Reshape the labels to match the grid
labels = reshape(X(:, 3), size(X1));

% Plot the decision boundary
contour(X1, X2, labels, 'LineColor', 'k');
hold on;
scatter(data1(:, 1), data1(:, 2), 'r', 'filled');
scatter(data2(:, 1), data2(:, 2), 'b', 'filled');
hold off;
