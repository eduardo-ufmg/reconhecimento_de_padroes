function generate_and_evaluate(datasetType, samples, h, k, noise, print)

  % Generate the dataset
  [data1, labels1, data2, labels2] = generate_dataset(datasetType, samples, noise);
  complete_set = [data1, labels1; data2, labels2];

  % Generate a grid for plotting
  [X, X1, X2] = generate_grid(complete_set, 100);

  % Classify the points
  for i = 1:size(X, 1)
    X(i, 3) = mykNN(X(i, 1:2), complete_set, k, h);
  end

  % Plot the decision boundaries
  plot_boundaries(X, X1, X2, data1, data2, print, string(k) + '_' + string(noise) + '.png');

end