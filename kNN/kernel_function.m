function value = kernel_function(x, h, mu)
  % Gaussian kernel function
  % x: point to evaluate
  % h: radius of the kernel
  % mu: marginal distribution average vector

  n = length(x); % number of features
  K = h * eye(n); % covariance matrix

  % Evaluate the kernel function
  value = (1 / (sqrt((2 * pi)^n * det(K)))) * exp(-0.5 * (x - mu) * (K \ (x - mu)'));

end