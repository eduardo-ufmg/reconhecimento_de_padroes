function [Q1, Q2] = to_characteristic_space(x, complete_set, k, h)
  % Transform the data to characteristic space
  % x: point to be transformed
  % complete_set: training set with class labels
  % k: number of nearest neighbors
  % h: kernel bandwidth
  % Q1, Q2: coordinates in characteristic space

  [sorted_distances, sorted_indexes, class_labels] = preparekNN(x, complete_set, k);

  % init Q1 and Q2
  Q1 = 0;
  Q2 = 0;

  for i = 1:k

    mu = complete_set(sorted_indexes(i), 1:end-1);
    kernel_value = kernel_function(x, h, mu);

    if class_labels(i) == 1
      Q1 = Q1 + kernel_value;
    else
      Q2 = Q2 + kernel_value;
    end
  end

end
