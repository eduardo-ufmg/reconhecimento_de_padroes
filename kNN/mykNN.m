function label = mykNN(x, complete_set, k, h)
  % mykNN: binary classifier: label is {-1, 1}
  % x: point to be classified
  % complete_set: training set with class labels
  % k: number of nearest neighbors
  % h: kernel bandwidth
  % label: class label of x

  % number of training samples
  training_set_point_qtty = size(complete_set, 1);

  % number of features
  feature_qtty = size(complete_set, 2) - 1;

  % initialize distances vector
  distances = zeros(training_set_point_qtty, 1);

  % compute distances
  for i = 1:training_set_point_qtty
    distances(i) = euclidean_distance(x, complete_set(i, 1:feature_qtty));
  end

  % sort distances
  [sorted_distances, sorted_indexes] = sort(distances);

  % initialize class labels vector
  class_labels = zeros(k, 1);

  % get class labels of k nearest neighbors
  for i = 1:k
    class_labels(i) = complete_set(sorted_indexes(i), feature_qtty + 1);
  end

  % compute kernel weights
  weights = zeros(k, 1);

  for i = 1:k
    weights(i) = kernel_function(sorted_distances(i), h);
  end

  % compute class label
  label = sign(sum(weights .* class_labels));

end

function distance = euclidean_distance(x, y)
  % euclidean_distance: computes the euclidean distance between two points
  % x, y: points
  % distance: euclidean distance between x and y

  distance = sqrt(sum((x - y) .^ 2));

end

function value = kernel_function(distance, h)
  % kernel_function: computes the kernel value using a normal distribution
  % distance: distance between points
  % h: kernel bandwidth
  % value: kernel value

  value = normpdf(distance, 0, h);

end
