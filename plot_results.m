function plot_results(X, X1, X2, data1, data2, Q1s, Q2s, samples, print, filename)
  % Plot the decision boundaries and characteristic space of a binary classifier

  if print
    fig = figure('Visible', 'off');
  else
    fig = figure('Visible', 'on');
  end

  % Define colors for the background and points
  bg_color1 = [0.6, 0.6, 1]; % Blueish color
  bg_color2 = [0.8, 0.8, 0]; % Yellowish color

  % Define a colormap for the background
  cmap = [bg_color2; bg_color1];

  % Define colors for the points
  pt_color1 = [0, 0, 1]; % Blue
  pt_color2 = [1, 1, 0]; % Yellow

  % Define colors for the points
  pt_colors = [pt_color1; pt_color2];

  % Plot the decision boundaries
  subplot(1, 2, 1);
  plot_boundaries(X, X1, X2, data1, data2, pt_colors, cmap);
  title('Decision boundaries');

  % Plot the characteristic space
  subplot(1, 2, 2);
  plot_charspace(Q1s, Q2s, samples, pt_colors);
  title('Characteristic space');

  if print
    saveas(fig, './output/' + filename);
    close(fig);
  end

end