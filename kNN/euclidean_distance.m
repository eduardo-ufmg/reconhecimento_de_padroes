function distance = euclidean_distance(x, y)
  % euclidean_distance: computes the euclidean distance between two points
  % x, y: points
  % distance: euclidean distance between x and y

  distance = sqrt(sum((x - y) .^ 2));

end
