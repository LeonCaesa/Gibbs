#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Sat Jul  4 20:43:37 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


from gibbs import Model, w_mu_nominator# analysis:ignore
from data_simu import generate_data

import numpy as np
import matplotlib.pyplot as plt


from mcmc_plot import trace_plot, get_trace_list




def forward_sample(d, q_star, n_sample, prior_param):
    """
        Function to simulate ppca data using X= WZ + \sigma^2 I_n
        param: d, dimension of data
        param: q_star, true dimension of principle components
        param: n_sample, numebr of observations
        param: prior_param, dictonary of prior setup
    """
    # sampling for z
    mu_z = np.zeros([q_star])
    sigma2_z = np.diag(np.ones([q_star]))
    Z_star = np.random.multivariate_normal(mu_z, sigma2_z, n_sample).T


    # sampling for sigma2
    sigma2_star = 1/ np.random.gamma(prior_param['a_sigma2'], 1/prior_param['beta_sigma2'])
    
    # sampling for alpha
    a_star_list = 1/ np.random.gamma(prior_param['a_vj'], 1/prior_param['beta_vj'])
    
  #  print(sigma2_star)
    # sampling for w
    mu_w = np.zeros([d])
    W = np.zeros([d, q_star])
    for j in range(q_star):
        sigma2_w = np.diag(a_star_list[j] * np.ones(d))
        W_star_j = np .random.multivariate_normal(mu_w, sigma2_w)
        W[:, j] = W_star_j

    X = np.dot(W, Z_star) + np.random.normal(0, np.sqrt(sigma2_star), [d, n_sample])

    return X, W, Z_star, sigma2_star, a_star_list



def forward_only_sigma2_Z(W, n_sample, prior_param):
    
#    Z_star = np.random.multivariate_normal(mu_z, sigma2_z, n_sample).T
    Z_star = np.random.normal(0, 1, [q, n_sample])
    sigma2_star = 1/ np.random.gamma(prior_param['a_sigma2'], 1/prior_param['beta_sigma2'])
    X = np.dot(W, Z_star) + np.random.normal(0, np.sqrt(sigma2_star), [d, n_sample])
    
    return X
    
    
    

def geweke(iterations, d, q_star, prior_param, init_dict, n_sample, xi = None):
    """
        Function to simulate ppca data using X= WZ + \sigma^2 I_n
        param: iteration, number of gibbs steps
        param: q_star, true dimension of principle components
        param: n_sample, numebr of observations
        param: xi, the power posterior parameter
        param: prior_param, dictonary of prior setup
        param: d, dimension of the data x
    """
    
    X, W, Z_star, sigma2_star, a_star_list = forward_sample(d, q_star, n_sample, prior_param)
    gibbs_results = []   

                   
    inference = Model(X, init_dict, iterations, q, prior_param, xi = xi)  
    inference.sigma2_list = [sigma2_star]
    inference.W_list = [W]
    inference.Z_list = [Z_star]
    inference.a_star_list = [a_star_list]
    
    
    forward_results = []
    for i in range(iterations):
        X_i = forward_only_sigma2_Z(W, n_sample, prior_param)
        forward_results.append(np.mean(np.std(X_i,axis=1)))

    
    
    
    count =0
#    sigma2_list = []
    while len(gibbs_results) != iterations:
        count += 1
        inference.gibbs_step(X_i)                
        X_i = inference.sample_x()
        if count %15 ==0:
            gibbs_results.append(np.mean(np.std(X_i,axis=1)))
            
            
            
#            X_WZ = (inference.X - np.dot(inference.W_list[-1], inference.Z_list[-1]))
#            S_x = np.trace(np.dot(X_WZ.T, X_WZ))        
##            sigma2_list.append(S_x)

#    plt.plot(sigma2_list)
#    
##    Analysis on only sigma2
#    true_sigma2_sample = 1/ np.random.gamma(prior_param['a_sigma2'], 1/prior_param['beta_sigma2'], len(sigma2_list))
#    sns.distplot(true_sigma2_sample, label='True sigma2')    
#    
#    sns.distplot(inference.sigma2_list, label='MCM sigma2')    
#    plt.legend()
#    plt.show()
#    
#    
#    plt.scatter(np.sort(np.array(true_sigma2_sample)), np.sort(np.array(sigma2_list)/n_sample/d))
#    plt.xlabel('True')
#    plt.ylabel('Mcmc Sampled')
#    plt.show()
     
    
    return forward_results, gibbs_results
  


if __name__ == '__main__':

    d = 5
    q_star = d-1
    q = d-1
    xi = 1    

        
    a_vj = 0.5 * d * np.ones(q) + 1
    epislon = 0.1
    
    #a_vj = 10 * d * np.ones(q)
    prior_param = dict({'beta_sigma2': 0.5,
                            'a_sigma2': 3,
                            'a_vj': a_vj ,
                            'beta_vj': epislon * (a_vj-1)
                            })
    
    init_dict = dict({'Z0':np.random.normal(0,1, [q, 1]),
        "sigma20":np.random.gamma(3,0.1),
        "w0":np.random.normal(0, 1, [d,q]),
        "v0": np.random.gamma(1,2, d-1)})   

    iterations = 1000
    n_sample = 50
    forward_results, gibbs_results = geweke(iterations, d, q_star, prior_param, init_dict, n_sample)
            
    
    
    indx = 0
    #plt.scatter(np.sort(np.array(forward_results)[:,indx]), np.sort(np.array(gibbs_results)[:,indx]))
    plt.scatter(np.sort(np.array(forward_results)), np.sort(np.array(gibbs_results)))
    plt.xlabel('True Sample')
    plt.ylabel('MCMC Sample')
    plt.show()
    
    
print(np.mean(np.array(forward_results))/np.mean(gibbs_results))
    
    
import seaborn as sns    
sns.distplot(np.array(forward_results[500:]))
sns.distplot(np.array(gibbs_results[500:]))
plt.show()

a=gibbs_results
b= np.mean(gibbs_results)
plt.acorr(a-b, normed=True, usevlines=False, maxlags=10, label=u'thinned')
plt.show()


