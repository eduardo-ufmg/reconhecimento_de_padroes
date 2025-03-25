import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_boundaries(ax, X, xx, yy, data1, data2):
  Z = X[:, 2].reshape(xx.shape)
  cmap = ListedColormap(['#FFFF80', '#9999FF'])
  ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
  ax.scatter(data1[:, 0], data1[:, 1], c='b', edgecolors='k')
  ax.scatter(data2[:, 0], data2[:, 1], c='y', edgecolors='k')
  ax.set_title('Decision boundaries')

def plot_charspace(ax, Q1s, Q2s, labels):
  ax.scatter(Q1s[labels == 1], Q2s[labels == 1], c='b')
  ax.scatter(Q1s[labels == -1], Q2s[labels == -1], c='y')
  max_val = max(np.max(Q1s), np.max(Q2s))
  ax.plot([0, max_val], [0, max_val], 'k--')
  ax.set_title('Characteristic space')

def plot_results(X_grid, xx, yy, data1, data2, Q1s, Q2s, labels, k, h, noise, which, save=False):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  # Plot decision boundaries
  plot_boundaries(ax1, X_grid, xx, yy, data1, data2)
  
  # Plot characteristic space
  plot_charspace(ax2, Q1s, Q2s, labels)
  
  # Add parameter info
  param_val = h if which == 'h' else noise
  fig.suptitle(f'k = {k}, {which} = {param_val:.2f}')
  
  if save:
    os.makedirs('output', exist_ok=True)
    plt.savefig(f'output/kNN_{k}_{which}{param_val:.2f}.png')
    plt.close()
  else:
    plt.show()