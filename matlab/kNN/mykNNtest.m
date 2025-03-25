% Generate synthetic dataset using MATLAB's built-in functions
datasetType = 'blobs'; % Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'

% Dataset parameters
samples = 200;
noise = 1;

% Test parameters
evaluate_noise = true;
evaluate_h = true;

% ------------------------------------------------------------------------------------------ %

if evaluate_h
  for k = [1 25 50]
    for h = [0.0001 0.01 1]

      fprintf('k = %d, h = %.2f\t', k, h);

      generate_and_evaluate(datasetType, samples, k, h, noise, true, 'h');

      fprintf('Done\n');

    end
  end
end

if evaluate_noise
  for k = [1 25 50]
    for noise = [1 2 3]

      fprintf('k = %d, noise = %d\t', k, noise);

      generate_and_evaluate(datasetType, samples, k, 1, noise, true, 'noise');

      fprintf('Done\n');

    end
  end
end
