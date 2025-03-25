function [sorted_distances, sorted_indexes, class_labels] = preparekNN(x, complete_set, k)
  % Prepare kNN algorithm
  % x: point to be classified
  % complete_set: training set with class labels
  % k: number of nearest neighbors
  % sorted_distances: sorted distances
  % class_labels: class labels of k nearest neighbors

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

end
