function plot_boundaries(X, X1, X2, data1, data2, pt_colors, cmap)
  % Plot the decision boundaries of a binary classifier
  
  % Convert labels from -1 and 1 to 0 and 1
  labels = (X(:, 3) + 1) / 2;

  % Reshape the labels to match the grid
  labels = reshape(labels, size(X1));

  % Plot the decision boundary
  contourf(X1, X2, labels, 'LineColor', 'none');
  colormap(cmap);

  hold on;
  scatter(data1(:, 1), data1(:, 2), 'filled', 'MarkerFaceColor', pt_colors(1, :));
  scatter(data2(:, 1), data2(:, 2), 'filled', 'MarkerFaceColor', pt_colors(2, :));
  hold off;

end