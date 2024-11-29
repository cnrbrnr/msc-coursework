# Boilerplate
import os

import numpy as np
import pandas as pd

# Define functions
def sign(x):
    '''returns -1, 0, 1 if x is negative, zero, positive, resp.'''
    return int(x / np.abs(x)) if x !=0 else 0

def soft_thresh(x, tau):
    '''reduces the magnitude of x by tau, and returns zero if |x|<t'''
    return sign(x) * (np.abs(x) - tau) if np.abs(x)>tau else 0

def ols_LASSO(X, y, reg, tol=1e-10, max_iter=1e3, verbose=False):

    # initialize
    delta = 1 # tracks fluctuations in coefficients
    n = X.shape[0] # rows are samples
    p = X.shape[1] # columns are predictors
    beta = np.zeros((p,)).reshape((p,-1)) # Initialize regression coefficients
    it = 0 # iterations

    while it < max_iter and delta > tol:
        it += 1
        prev_beta = beta

        # iterate over predictor coordinates
        for j in range(p):
            K = [c for c in range(p) if c != j] # predictor indices excluding coordinate
            rj = y - (X[:, K] @ beta[K, :]) # form aggregate response
            beta[j] = soft_thresh((1 / n) * np.sum(X[:, j] * rj), reg)

        delta = np.amax(np.abs(beta - prev_beta))
    
    if verbose == True:
        if it >= max_iter:
            print('Optimization timeout after {} iterations; delta={}'.format(it, round(delta, 4)))
        else:
            print('Estimate stabilized after {} iterations; tol={}, delta={}'.format(it, round(tol, 4), round(delta, 4)))
    
    return beta

def elastic_net(X, y, reg, alpha, tol=1e-10, max_iter=1e3, verbose=False):

    # initialize
    delta = 1 # tracks fluctuations in coefficients
    n = X.shape[0] # rows are samples
    p = X.shape[1] # columns are predictors
    beta = np.zeros((p,)).reshape((p,-1)) # Initialize regression coefficients
    it = 0 # iterations
    descent_constants = np.sum(X**2, axis=0) + (n * reg * (1 - alpha))

    while it < max_iter and delta > tol:
        it += 1
        prev_beta = beta

        # iterate over predictor coordinates
        for j in range(p):
            K = [c for c in range(p) if c != j] # predictor indices excluding coordinate
            rj = y - (X[:, K] @ beta[K, :]) # form aggregate response
            dc = descent_constants[j]
            beta[j] = soft_thresh(np.sum(X[:, j] * rj) / dc, (n* reg * alpha) / dc)

    if verbose == True:

        if it >= max_iter:
            print('Optimization timeout after {} iterations; delta={}'.format(it, round(delta, 4)))
        else:
            print('Estimate stabilized after {} iterations; tol={}, delta={}'.format(it, round(tol, 4), round(delta, 4)))

    return beta

# Load the data
X = pd.read_csv('./STAT857/Assignment1/data/A1Q1_X.csv').values
y = pd.read_csv('./STAT857/Assignment1/data/A1Q1_y.csv').values

# Standardize predictors and response
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

# Estimate elastic net coefficients by coordinate descent
reg = 0.1
alpha = 0.6
#beta = elastic_net(X, y, reg, alpha, max_iter=1e3, verbose=True)


print(beta)