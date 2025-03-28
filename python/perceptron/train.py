import numpy as np

from perceptron.pred import pred

def train(X: np.ndarray, Y: np.ndarray, w: np.ndarray, t: float, eta: float, epochs: int) -> tuple[np.ndarray, float, np.ndarray]:
  """
    Train the perceptron model using the given data
    X: matrix of input features (n_samples, n_features)
    Y: vector of target classes (n_samples)
    w: guess vector of weights (n_features)
    t: guess threshold
    eta: learning rate
    epochs: number of epochs
    return: tuple of (weights, threshold, errors)
  """

  # Initialize the error list
  errors = []

  for epoch in range(epochs):
    # Initialize the number of errors
    n_errors = 0

    # Iterate over each sample
    for i in range(X.shape[0]):
      # Predict the class of the input sample
      y_pred = pred(X[i], w, t)

      y_expect = Y[i]

      # Update the weights and threshold if there is an error
      if y_pred != y_expect:
        n_errors += 1
        w += eta * (Y[i] - y_pred) * X[i]
        t += eta * (Y[i] - y_pred) * (-1)

    # Append the number of errors to the list
    errors.append(n_errors)

    # Stop training if there are no errors
    if n_errors == 0:
      break
    
  # Return the final weights, threshold, and errors
  return w, t, np.array(errors)
