% Generate synthetic dataset using MATLAB's built-in functions
datasetType = 'blobs'; % Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'

% Dataset parameters
samples = 200;
noise = 1;

% ------------------------------------------------------------------------------------------ %

for k = [1 25 50]
  for h = [0.0001 0.01 1]

    fprintf('k = %d, h = %.2f\t', k, h);

    generate_and_evaluate(datasetType, samples, h, k, noise, true);

    fprintf('Done\n');

  end
end
