% Generate synthetic dataset using MATLAB's built-in functions
datasetType = 'blobs'; % Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'

% Dataset parameters
samples = 100;
noise = 2;

% kNN parameters
k = 20;
h = 1;

% ------------------------------------------------------------------------------------------ %

generate_and_evaluate(datasetType, samples, h, k, noise, false, '');

% return; % don't need to save plots while testing

for k = 1:20:50
  for noise = 1:1:3

    generate_and_evaluate(datasetType, samples, h, k, noise, true, 'kNN_' + string(k) + '_' + string(noise) + '.png');

  end
end
