# ========== STAT857 FINAL EXAM: CONNOR BRAUN ==========
# ########## README ####################################
# - The following code requires Numpy v1.26, Pandas v2.2.2.
# - The file contains a function 'my_EM' which is my submission for 
#   part (d) of the final exam.
# - To run the function from this file directly, adjust the
#   'filename' and'data_path' variables in the 'main' function. The
#   program takes roughly 30s to execute on my computer.
# - The output corresponding to my best estimates can be found as a
#   comment at the end of the file.
# ######################################################
# ======================================================

# Imports
import os

import numpy as np
import pandas as pd

# EM algorithm implementation
def my_EM(
    Y, # response vector (n x 1 ndarray)
    X, # feature vector (n x 1 ndarray)
    params=np.array([0.1, 0.1, 0.1, 0.1]), # set to initialization producing "correct" estimates
    max_iter=100, # maximum allowable iterations
    tol=1e-8, # termination criterion
    verbose=True # print results to terminal
):

    # Define helper functions
    def phi(y, x, c, var, mu_coeff):
        '''Compute Gaussian density with mean depending on the feature variable'''
        mu = mu_coeff * [1 if x > c else 0][0]
        return (1 / (np.sqrt(2*np.pi*var))) * np.exp((-1 / (2*var)) * (y - mu)**2)
    
    def xi_hat(y, x, c_1, c_0, var, p):
        '''Compute the expected value of the latent bernoulli variable given data and parameter estimate'''
        return (p * phi(y, x, c_1, var, 1)) / (p * phi(y, x, c_1, var, 1) + (1 - p) * phi(y, x, c_0, var, 2))
    vec_xi_hat = np.vectorize(xi_hat, otypes=['float'], excluded=['c_1', 'c_0', 'var', 'p']) # vectorize for fast computation along multiple arrays simultaneously
    
    def greater_than(x, c):
        '''Simple 'strictly greater than' indicator function'''
        return 1 if x > c else 0
    vec_greater_than = np.vectorize(greater_than, otypes=['float'], excluded=['c']) # vectorize for fast computation along multiple arrays simultaneously
    
    # Initialization
    Y = Y.reshape((-1, 1))
    X = X.reshape((-1, 1))
    candidate_region_boundaries = np.unique(X) # permissable region boundaries to check during M-step
    n = Y.shape[0] # sample size
    status = 0 # exit status
    
    # Repeat E-M cycle until maximum iterations reached or estimates converge
    for i in range(max_iter):

        # E-step: compute expected value of latent variable at each data point given current parameterization
        xi_vec = vec_xi_hat(Y, X, params[0], params[1], params[2], params[3])

        # M-step: find new parameter estimates maximizing likelihood derived analytically
        p_new = (1 / n) * np.sum(xi_vec[:, 0]) # simple formula for Bernoulli probability update

        # Compute best c_1, c_0 by taking argmin of weighted loss over all region boundaries
        c_1_errs = np.empty((n, 1)) # initialize errors for 
        c_0_errs = np.empty((n, 1))
        for j, c in enumerate(candidate_region_boundaries):
            c_1_errs[j, 0] = np.sum(xi_vec * ((Y - (vec_greater_than(X, c))) ** 2)) # loss function given candidate c_1
            c_0_errs[j, 0] = np.sum((1 - xi_vec) * ((Y - (2 * vec_greater_than(X, c))) ** 2)) # loss function given candidate c_0
        
        # Take new c_1, c_0 to be those minimizing loss
        c_1_new = candidate_region_boundaries[np.argmin(c_1_errs)]
        c_0_new = candidate_region_boundaries[np.argmin(c_0_errs)]

        # Compute maximizing variance using plug-in estimates for c_1, c_0
        var_new = (1 / n) * np.sum((xi_vec * ((Y - (vec_greater_than(X, params[0]))) ** 2)) + ((1 - xi_vec) * ((Y - (2 * vec_greater_than(X, params[1]))) ** 2)))

        # Compute termination criterion
        if np.amax((params - np.array([c_1_new, c_0_new, var_new, p_new])) ** 2) < tol:
            status = 1 # update exit status
            params[:] = np.array([c_1_new, c_0_new, var_new, p_new])[:] # update parameter estimate
            break
        else:
            params[:] = np.array([c_1_new, c_0_new, var_new, p_new])[:] # update parameter estimate
    
    # Compute observed data log-likelihood using estimated parameters
    loglik = np.sum(np.log((1 / np.sqrt(2*np.pi*params[2])) * (params[3] * np.exp((-1 / (2 * params[2])) * (Y - vec_greater_than(X, params[0])) ** 2) + (1 - params[3]) * np.exp((-1 / (2 * params[2])) * (Y - 2*vec_greater_than(X, params[1])) ** 2))))

    # Print result summary to console
    if verbose == True:
        messages = ['exceeded maximum iterations', 'termination criterion met']
        print('EM algo terminated after {} steps; STATUS {}: {}\nEstimates:\n\tc_1: {}\n\tc_0: {}\n\tvar: {}\n\tp: {}\n\tlog-likelihood: {}'.format(i, status, messages[status], *params, loglik))
    
    return *params, loglik

def main():

    # Specify file name and path
    filename = 'final_train_data.csv' # csv filename 
    data_path ='/home/conbraun/Projects/msc-coursework/STAT857/FinalExam/code/data/{}'.format(filename) # path to csv file

    # Import data (requires Pandas)
    data = pd.read_csv(data_path).to_numpy()
    X = data[:, 0].reshape((-1, 1))
    Y = data[:, 1].reshape((-1, 1))

    # Run algorithm: takes ~30s, depending on initialization (default initialization converges to "correct" estimates)
    em_c_1, em_c_0, em_var, em_p, loglik = my_EM(Y, X)

    # ========== BEST ESTIMATES ==========
    # c_1 : 0.580318685451573
    # c_0 : 0.295725633211195
    # var : 0.043335428330704
    # p   : 0.338971237503968
    #
    # observed data log-likelihood evaluated at these estimates:
    #   l(theta|D) = -81.71166877054995

if __name__ == '__main__':
    main()