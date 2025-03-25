function [C1, C2] = charspace_closeness(Q1s, Q2s, labels)
  % Compute the average distance from each point to the other points in the same class
  % Q1s: a vector of the first dimension of the characteristic space
  % Q2s: a vector of the second dimension of the characteristic space
  % labels: a vector of class labels

  % Get the number of points
  n = numel(Q1s);

  % Initialize the distances
  D1 = zeros(n, 1);
  D2 = zeros(n, 1);

  % Compute the distances
  for i = 1:n
    % Get the class of the current point
    label = labels(i);

    % Compute the distances to the other points in the same class
    D1(i) = mean(sqrt((Q1s(labels == label) - Q1s(i)).^2 + (Q2s(labels == label) - Q2s(i)).^2));
    D2(i) = mean(sqrt((Q1s(labels == label) - Q1s(i)).^2 + (Q2s(labels == label) - Q2s(i)).^2));
  end

  % Compute the closeness
  C1 = 1 ./ D1;
  C2 = 1 ./ D2;
end
