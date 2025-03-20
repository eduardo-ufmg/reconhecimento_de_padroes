% Generate synthetic dataset using MATLAB's built-in functions
datasetType = 'blobs'; % Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'

% Dataset parameters
samples = 100;
noise = 2;

% kNN parameters
k = 5;
h = 1;

% ------------------------------------------------------------------------------------------ %
for k = 1:20:50
  for noise = 1:1:3

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
    plot_boundaries(X, X1, X2, data1, data2, true, string(k) + '_' + string(noise) + '.png');

  end
end
