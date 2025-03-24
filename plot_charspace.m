function plot_charspace(Q1s, Q2s, samples, pt_colors)
  hold on;
  scatter(Q1s(1:samples/2), Q2s(1:samples/2), 'filled', 'MarkerFaceColor', pt_colors(1, :));
  scatter(Q1s(samples/2+1:samples), Q2s(samples/2+1:samples), 'filled', 'MarkerFaceColor', pt_colors(2, :));
  hold off;
end