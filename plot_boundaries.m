function plot_boundaries(X, X1, X2, data1, data2)
  % Plot the decision boundaries of a binary classifier

  % Reshape the labels to match the grid
  labels = reshape(X(:, 3), size(X1));

  % Define a colormap for the background
  cmap = [0.8 0.8 0; 0.6 0.6 1];

  % Plot the decision boundary
  contourf(X1, X2, labels, 'LineColor', 'none');
  colormap(cmap);

  hold on;
  scatter(data1(:, 1), data1(:, 2), 'b', 'filled');
  scatter(data2(:, 1), data2(:, 2), 'y', 'filled');
  hold off;
  
end
