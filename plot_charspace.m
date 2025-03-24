function plot_charspace(Q1s, Q2s, labels, pt_colors)
  hold on;
  scatter(Q1s(labels == 1), Q2s(labels == 1), 'filled', 'MarkerFaceColor', pt_colors(1, :));
  scatter(Q1s(labels == -1), Q2s(labels == -1), 'filled', 'MarkerFaceColor', pt_colors(2, :));
  hold off;
end
