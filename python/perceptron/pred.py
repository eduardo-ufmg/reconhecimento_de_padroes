import numpy as np

def pred(x: np.ndarray, w: np.ndarray, t: float) -> int:
  """
    Predict the class of the input x using the perceptron model
    x: vector of input features
    w: vector of weights
    t: threshold
    return: predicted class (1 or -1)
    If the input has n features, the weight vector input must also have n elements
    The threshold is appended to the weight vector as the (n+1)th element
    A -1 input is added to the input feature vector
    Prediction is -1 if w^T · x < 0, 1 otherwise
  """

  # Append -1 to the input vector
  x = np.append(x, -1)
  # Append the threshold to the weight vector
  w = np.append(w, t)
  # Compute the dot product
  z = np.dot(w, x)
  # Return the predicted class
  return 1 if z >= 0 else -1

def pred_batch(X: np.ndarray, w: np.ndarray, t: float) -> np.ndarray:
  """
    Predict the class of each input sample in X using the perceptron model
    X: matrix of input samples (n_samples, n_features)
    w: vector of weights
    t: threshold
    return: predicted classes (1 or -1)
  """

  # Append -1 to each input sample
  X = np.hstack((X, -np.ones((X.shape[0], 1))))
  # Append the threshold to the weight vector
  w = np.append(w, t)
  # Compute the dot product for all samples
  z = np.dot(X, w)
  # Return the predicted classes
  return np.where(z >= 0, 1, -1)