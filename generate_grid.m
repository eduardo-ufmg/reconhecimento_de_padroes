function [X, X1, X2] = generate_grid(complete_set, resolution)

  % Determine the range of the data
  min_x1 = min(complete_set(:, 1));
  max_x1 = max(complete_set(:, 1));
  min_x2 = min(complete_set(:, 2));
  max_x2 = max(complete_set(:, 2));

  % Generate a grid of points within the data range
  x1 = linspace(min_x1 - 1, max_x1 + 1, resolution);
  x2 = linspace(min_x2 - 1, max_x2 + 1, resolution);
  [X1, X2] = meshgrid(x1, x2);
  X = [X1(:), X2(:), zeros(length(X1(:)), 1)];

end