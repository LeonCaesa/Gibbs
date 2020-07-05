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





def geweke(iterations, d, q_star, sigma2_star, a_star_list, prior_param, init_dict, n_sample, xi = None):
    forward_results = []
    for i in range(iterations):
        X_i = generate_data(d, q_star, n_sample, sigma2_star, a_star_list)
        forward_results.append(np.mean(np.std(X_i,axis=1)))
    
 
    gibbs_results = []                      
    X_i = generate_data(d, q_star, n_sample, sigma2_star, a_star_list)
    inference = Model(X_i, init_dict, iterations, q, prior_param, xi = xi)  
    inference.gibbs_result()
    
    count =0
#    for i in range(iterations*5):
    while len(gibbs_results) != iterations:
        count += 1
        inference.gibbs_step(X_i)
        X_i = inference.sample_x()
        if count %10 ==0:
            gibbs_results.append(np.mean(np.std(X_i,axis=1)))
        
    return forward_results, gibbs_results
  
#trace_plot([inference.gibbs_result()], var_list= ['sigma2_list', 'alpha_list'])      
        


if __name__ == '__main__':

    d = 5
    q_star = 4
    n_sample = 1000
    sigma2_star = 0.8
    a_star_list = 1 / np.linspace(1, 10, q_star)
    q = d-1
    

        
    prior_param = dict({'beta_sigma2':2,
          'a_sigma2':10,
          'a_aj': 1 / np.linspace(1,10,q),
          'beta_aj':1 / np.linspace(1,10,q)    
    })
    
    init_dict = dict({'Z0':np.random.normal(0,1, [q, 1]),
        "sigma20":np.random.gamma(4,1/4),
        "w0":np.random.normal(0, 1, [d,q]),
        "alpha0": np.random.uniform(0, 2, q)})   

    iterations = 1000
    n_sample = 3
    forward_results, gibbs_results = geweke(iterations, d, q_star, sigma2_star, a_star_list, prior_param, init_dict, n_sample)
    
    
    
    indx = 0
    #plt.scatter(np.sort(np.array(forward_results)[:,indx]), np.sort(np.array(gibbs_results)[:,indx]))
    plt.scatter(np.sort(np.array(forward_results[500:])), np.sort(np.array(gibbs_results[500:])))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    
print(np.mean(np.array(forward_results))/np.mean(gibbs_results))
    
    
import seaborn as sns    
sns.distplot(np.array(forward_results))
sns.distplot(np.array(gibbs_results))
plt.show()

a=gibbs_results
b= np.mean(gibbs_results)
plt.acorr(a-b, normed=True, usevlines=False, maxlags=10, label=u'thinned')
plt.show()