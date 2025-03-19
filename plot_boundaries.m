function plot_boundaries(X, X1, X2, data1, data2, print, filename)
  % Plot the decision boundaries of a binary classifier
  
  % Convert labels from -1 and 1 to 0 and 1
  labels = (X(:, 3) + 1) / 2;

  % Reshape the labels to match the grid
  labels = reshape(labels, size(X1));

  % Define colors for the background and points
  bg_color1 = [0.6, 0.6, 1]; % Blueish color
  bg_color2 = [0.8, 0.8, 0]; % Yellowish color

  % Define a colormap for the background
  cmap = [bg_color2; bg_color1];

  % Create a new figure
  if print
    fig = figure('Visible', 'off');
  else
    fig = figure('Visible', 'on');
  end

  % Plot the decision boundary
  contourf(X1, X2, labels, 'LineColor', 'none');
  colormap(cmap);

  % Define colors for the points
  pt_color1 = [0, 0, 1]; % Blue
  pt_color2 = [1, 1, 0]; % Yellow

  hold on;
  scatter(data1(:, 1), data1(:, 2), 'filled', 'MarkerFaceColor', pt_color1);
  scatter(data2(:, 1), data2(:, 2), 'filled', 'MarkerFaceColor', pt_color2);
  hold off;
  
  if print
    saveas(fig, './output/' + filename);
    close(fig);
  end

end