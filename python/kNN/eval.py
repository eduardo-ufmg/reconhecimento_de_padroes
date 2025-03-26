from kNN.run import generate_and_evaluate

if __name__ == "__main__":
  DATASET_TYPE = 'blobs'
  SAMPLES = 200
  
  # Evaluate bandwidth (h)
  for k in [1, 25, 50]:
    for h in [0.0001, 0.01, 1]:
      generate_and_evaluate(DATASET_TYPE, SAMPLES, k, h, noise=1, which='h', save=True)
  
  # Evaluate noise
  for k in [1, 25, 50]:
    for noise in [1, 2, 3]:
      generate_and_evaluate(DATASET_TYPE, SAMPLES, k, h=1, noise=noise, which='noise', save=True)
      