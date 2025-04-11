import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def pred(X, args0, args1, method):
    """
    Predict class labels using the specified method.
    
    Args:
        X: Input features (n_samples, n_features)
        args0: Model parameters for class 0
        args1: Model parameters for class 1
        method: 'normal', 'gaussian_mix', or 'kde'
    
    Returns:
        Predicted class labels (n_samples,)
    """

    methods = {
        'normal': predict_normal,
        'gaussian_mix': predict_gaussian_mix,
        'kde': predict_kde
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}")
    return methods[method](X, args0, args1)

def predict_normal(X, params0, params1):
    """Predict using Gaussian distributions."""
    mean0, cov0, prior0 = params0
    mean1, cov1, prior1 = params1
    
    log_likelihood0 = multivariate_normal.logpdf(X, mean=mean0, cov=cov0)
    log_likelihood1 = multivariate_normal.logpdf(X, mean=mean1, cov=cov1)
    
    log_posterior0 = np.log(prior0) + log_likelihood0
    log_posterior1 = np.log(prior1) + log_likelihood1
    
    return np.argmax(np.vstack([log_posterior0, log_posterior1]), axis=0)

def predict_gaussian_mix(X, params0, params1):
    """Predict using Gaussian Mixture Models."""
    gmm0, prior0 = params0
    gmm1, prior1 = params1
    
    log_likelihood0 = gmm0.score_samples(X)
    log_likelihood1 = gmm1.score_samples(X)
    
    log_posterior0 = np.log(prior0) + log_likelihood0
    log_posterior1 = np.log(prior1) + log_likelihood1
    
    return np.argmax(np.vstack([log_posterior0, log_posterior1]), axis=0)

def predict_kde(X, params0, params1):
    """Predict using Kernel Density Estimators."""
    kde0, prior0 = params0
    kde1, prior1 = params1
    
    log_likelihood0 = kde0.score_samples(X)
    log_likelihood1 = kde1.score_samples(X)
    
    log_posterior0 = np.log(prior0) + log_likelihood0
    log_posterior1 = np.log(prior1) + log_likelihood1
    
    return np.argmax(np.vstack([log_posterior0, log_posterior1]), axis=0)