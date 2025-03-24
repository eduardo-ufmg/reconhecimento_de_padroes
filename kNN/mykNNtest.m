% Generate synthetic dataset using MATLAB's built-in functions
datasetType = 'blobs'; % Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'

% Dataset parameters
samples = 300;
noise = 3;

% ------------------------------------------------------------------------------------------ %

for k = 25
  for h = 2

    fprintf('k = %d, h = %.2f\t', k, h);

    generate_and_evaluate(datasetType, samples, h, k, noise, false);

    fprintf('Done\n');

  end
end
