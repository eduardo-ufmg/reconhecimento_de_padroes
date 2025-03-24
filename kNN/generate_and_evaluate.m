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

  % Transform the training set to characteristic space
  Q1s = zeros(samples, 1);
  Q2s = zeros(samples, 1);

  for i = 1:samples
    [Q1s(i), Q2s(i)] = to_characteristic_space(complete_set(i, 1:2), complete_set, k, h);
  end

  plot_results(X, X1, X2, data1, data2, Q1s, Q2s, samples, print, k, h);

end