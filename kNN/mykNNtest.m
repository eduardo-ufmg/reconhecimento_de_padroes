% Generate synthetic dataset using MATLAB's built-in functions
datasetType = 'blobs'; % Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'

% Dataset parameters
samples = 100;
noise = 2;

% kNN parameters
k = 5;
h = 1;

% ------------------------------------------------------------------------------------------ %

max_noise = 4;
max_tests = 10;
test_counter = 0;

for k = 1:(samples / 10):(samples / 2)
  for noise = 0:(max_noise / 4):max_noise
    fprintf('k = %d, noise = %.1f ', k, noise);

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
    plot_boundaries(X, X1, X2, data1, data2, true, 'kNN_' + string(k) + '_' + string(noise) + '.png');
    
    fprintf('done\n');

    test_counter = test_counter + 1;

    if test_counter == max_tests
      break;
    end

  end

  if test_counter == max_tests
    break;
  end

end
