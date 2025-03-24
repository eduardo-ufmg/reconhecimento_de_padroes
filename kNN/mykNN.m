function label = mykNN(x, complete_set, k, h)
  % mykNN: binary classifier: label is {-1, 1}
  % x: point to be classified
  % complete_set: training set with class labels
  % k: number of nearest neighbors
  % h: kernel bandwidth
  % label: class label of x

  [sorted_distances, sorted_indexes, class_labels] = preparekNN(x, complete_set, k);

  % compute kernel weights
  weights = zeros(k, 1);

  for i = 1:k

    mu = complete_set(sorted_indexes(i), 1:end-1);
    weights(i) = kernel_function(x, h, mu);

  end

  % compute class label
  label = sign(sum(weights .* class_labels));

end
