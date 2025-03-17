function [data1, labels1, data2, labels2] = generate_dataset(datasetType, n, noise)
  % Generate synthetic dataset using MATLAB's built-in functions
  % datasetType: Choose between 'blobs', 'spirals', 'moons', 'xor', 'circles'
  % n: Number of samples
  % noise: Noise level

  switch datasetType
    case 'blobs'
      data1 = mvnrnd([2, 2], [0.5, 0; 0, 0.5], n/2);
      data2 = mvnrnd([-2, -2], [0.5, 0; 0, 0.5], n/2);
    case 'spirals'
      t = linspace(0, 4*pi, n/2);
      data1 = [t'.*cos(t'), t'.*sin(t')];
      data2 = [-t'.*cos(t'), -t'.*sin(t')];
    case 'moons'
      [data, labels] = make_moons(n, noise);
      data1 = data(labels == 1, :);
      data2 = data(labels == 0, :);
    case 'xor'
      data1 = [randn(n/4, 2) + 2; randn(n/4, 2) - 2];
      data2 = [randn(n/4, 2) + [-2, 2]; randn(n/4, 2) - [-2, 2]];
    case 'circles'
      [data, labels] = make_circles(n, noise);
      data1 = data(labels == 1, :);
      data2 = data(labels == 0, :);
    otherwise
      error('Unknown dataset type');
  end

  % Assign labels
  labels1 = ones(size(data1, 1), 1);
  labels2 = -ones(size(data2, 1), 1);
end

% Helper functions for generating datasets
function [data, labels] = make_moons(n, noise)
  theta = linspace(0, pi, n/2)';
  r = 1 + noise * randn(n/2, 1);
  data1 = [r .* cos(theta), r .* sin(theta)];
  data2 = [1 - r .* cos(theta), -r .* sin(theta) - 0.5];
  data = [data1; data2];
  labels = [ones(n/2, 1); zeros(n/2, 1)];
end

function [data, labels] = make_circles(n, noise)
  theta = linspace(0, 2*pi, n/2)';
  r1 = 1 + noise * randn(n/2, 1);
  r2 = 0.5 + noise * randn(n/2, 1);
  data1 = [r1 .* cos(theta), r1 .* sin(theta)];
  data2 = [r2 .* cos(theta), r2 .* sin(theta)];
  data = [data1; data2];
  labels = [ones(n/2, 1); zeros(n/2, 1)];
end