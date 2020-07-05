"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Wed Jun  3 18:44:48 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""

import numpy as np
import seaborn as sns
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from data_simu import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class Model:
    def __init__(self, X, init_dict, iterations, q , prior_param, xi = None):
        """
        Function to initalialize gibbs scheme
        params: X, d by n data matrix,
        params: iterations, number of iterations in gibbs sampling
        params: q, Maximum numebr of components, must be consistent with init_list
        params: init_list, dict of q by n array Z0, 
                                   d by q array w0, 
                                   1 by q array alpha,
                                   scalar sigma sqaure
        params: prior_param, dict of prior for alpha_j, 1 by q vector a_aj and beta_aj
                                     prior for sigma sqaure, scalar beta_sigma2, scalar a_sigma2
        params: xi, scalar, approxmiate sampling coefficient
        return: pandas dataframe of inference on sigma2, Z, W and alpha
    
        """
        self.d = np.shape(X)[0]
  
        self.n_sample = np.shape(X)[1]         

        self.q = q        
        self.iterations = iterations
        self.X = X
        # prior_params
        self.beta_sigma2 = prior_param['beta_sigma2']
        self.a_sigma2 = prior_param['a_sigma2']
        self.a_aj = prior_param['a_aj']
        self.beta_aj = prior_param['beta_aj']
        
        # init_list
        self.Z_list = [init_dict['Z0']]
        self.sigma2_list = [init_dict['sigma20']]
        self.W_list = [init_dict['w0']]
        self.alpha_list = [init_dict['alpha0']]
        if xi is None:
            self.xi = 1
        else:
            self.xi = xi
    def sample_sigma2(self):
        
        alpha_sigma2_temp = self.n_sample * self.xi * self.d / 2 + self.a_sigma2
        X_WZ = (self.X - np.dot(self.W_list[-1], self.Z_list[-1]))
        S_x = np.trace(np.dot(X_WZ.T, X_WZ))
        beta_sigma2_temp = (0.5 * (S_x * self.xi + 2 * self.beta_sigma2))
        self.sigma2_list.append(
            1 / np.random.gamma(alpha_sigma2_temp, 1 / beta_sigma2_temp))
        
        
    def sample_z(self):
        C = self.xi / self.sigma2_list[-1] * \
            np.dot(self.W_list[-1].T, self.W_list[-1]) + np.diag(np.ones([self.q]))

        first = np.linalg.inv(C)

        second = self.xi / self.sigma2_list[-1] * np.dot(self.W_list[-1].T, self.X)

        Z_hat = np.dot(first, second)

        Z_sigma2 = self.xi / \
            self.sigma2_list[-1] * \
            np.linalg.inv(
                np.dot(self.W_list[-1].T, self.W_list[-1]) + np.diag(np.ones(self.q)))

        Z_temp = np.random.normal(0, 1, [self.q, self.n_sample])

        Chol = np.linalg.cholesky(Z_sigma2)

        self.Z_list.append(np.dot(Chol, Z_temp) + Z_hat)
    def sample_w(self):
        
         # sampling for (w_j)_{dx1}

        nominator = w_mu_nominator(self.X, self.Z_list)

        denominator = (
            self.xi / self.sigma2_list[-1] * self.alpha_list[-1] + np.sum(self.Z_list[-1]**2, axis=1))

        mu_w = nominator.T / denominator.T

        sigma2_w = self.sigma2_list[-1] / (self.xi / self.sigma2_list[-1]
                                      * self.alpha_list[-1] + np.sum(self.Z_list[-1]**2, axis=1))

        sigma2_w_temp = [np.diag(np.repeat(i, self.d)) for i in sigma2_w]

        self.W_list.append(
            np.array(list(map(np.random.multivariate_normal, mu_w.T, sigma2_w_temp))).T)

    def sample_alpha(self):
        
        # sampling for (alpha)_{1xq}
        alpha_a = self.d / 2 + self.a_aj

        beta_a = 0.5 * np.diag(np.dot(self.W_list[-1].T, self.W_list[-1])) + self.beta_aj

        self.alpha_list.append(np.random.gamma(alpha_a, 1 / beta_a))

    def gibbs_step(self, X):
        self.X = X
        self.n_sample = np.shape(X)[1]            
        self.sample_sigma2()
        self.sample_z()
        self.sample_w()
        self.sample_alpha()        
        
    def sample_x(self):
#        mu_z = np.zeros([self.q])
#        sigma2_z = np.diag(np.ones([self.q]))
#        Z_star = np.random.multivariate_normal(mu_z, sigma2_z, self.n_sample).T
#    
#        mu_w = np.zeros([self.d])
#        W = np.zeros([self.d, self.q])
#        for j in range(self.q):
#            sigma2_w = np.diag(self.alpha_list[-1][j] * np.ones(self.d))
#            W_star_j = np.random.multivariate_normal(mu_w, sigma2_w)
#            W[:, j] = W_star_j
#        X = np.dot(W, Z_star) + np.random.normal(0, self.sigma2_list[-1], [self.d, self.n_sample])                        
        X = np.dot(self.W_list[-1], self.Z_list[-1]) + np.random.normal(0, self.sigma2_list[-1], [self.d, self.n_sample])        
        
        
        return X
    
    def gibbs_result(self):
        for i in range(self.iterations):
            self.gibbs_step(self.X)

        return dict({'sigma2_list': self.sigma2_list,
                 'Z_list': self.Z_list,
                 'W_list': self.W_list,
                 'alpha_list': self.alpha_list,
                 })



def w_mu_nominator(X, Z_list):
    """
        Function to calculate the mu vector of the conditional W matrix
    """
    q = np.shape(Z_list[-1])[0]
    d = np.shape(X)[0]
    w_temp = np.zeros([q, d])
    for j in range(q):
        w_temp[j, :] = np.sum(X * Z_list[-1][j, :], axis=1)
    return w_temp





if __name__ == '__main__':

    # data generation parameter
    d = 5
    q_star = 1
    q = d-1
    n_sample = 1000
    sigma2_star = 1
    a_star_list = 1 / np.linspace(1, 10, q_star)
    X = generate_data(d, q_star, n_sample, sigma2_star, a_star_list)

    plt.figure(figsize=(10, 6))
    pd.plotting.scatter_matrix(pd.DataFrame(X).T)
    plt.xlabel('Component i')
    plt.ylabel('Component j')

    prior_param = dict({'beta_sigma2': 2,
                        'a_sigma2': 10,
                        'a_aj': 1 / np.linspace(1, 10, q),
                        'beta_aj': 1 / np.linspace(1, 10, q)
                        })

    init_dict = dict({'Z0': np.random.multivariate_normal(np.zeros([q]), np.diag(np.ones([q])), n_sample).T,
                      "sigma20": 1.5,
                      "w0": np.random.normal(0, 0.6, [d, q]),
                      "alpha0": np.ones(q)
                      })

    iterations = 1000

    inference = Model(X, init_dict, iterations, q, prior_param)
    
    

