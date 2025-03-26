from kNN.run import generate_and_evaluate

if __name__ == "__main__":
  # Define the type of dataset and the number of samples to generate
  DATASET_TYPE = 'blobs'
  SAMPLES = 200
  
  # Evaluate the effect of different bandwidth (h) values on the model
  for k in [1, 25, 50]:  # Iterate over different values of k (number of neighbors)
    for h in [0.0001, 0.01, 1]:  # Iterate over different bandwidth values
      # Generate the dataset, evaluate the model, and save the results
      generate_and_evaluate(DATASET_TYPE, SAMPLES, k, h, noise=1, which='h', save=True)
  
  # Evaluate the effect of different noise levels on the model
  for k in [1, 25, 50]:  # Iterate over different values of k (number of neighbors)
    for noise in [1, 2, 3]:  # Iterate over different noise levels
      # Generate the dataset, evaluate the model, and save the results
      generate_and_evaluate(DATASET_TYPE, SAMPLES, k, h=1, noise=noise, which='noise', save=True)