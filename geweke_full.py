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


import seaborn as sns
from gibbs import Model, w_mu_nominator  # analysis:ignore
#from data_simu import generate_data  # analysis:ignore

import numpy as np
import matplotlib.pyplot as plt


def forward_sample(d, q_star, n_sample, prior_param, verbose = False):
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
    sigma2_star = 1 / \
        np.random.gamma(prior_param['a_sigma2'],
                        1 / prior_param['beta_sigma2'])
    if verbose:
        print('true sigma2_star is' +str(sigma2_star))
    # sampling for alpha
    v_star_list = 1 / \
        np.random.gamma(prior_param['a_vj'], 1 / prior_param['beta_vj'])

    # sampling for w
    mu_w = np.zeros([d])
    W = np.zeros([d, q_star])
    for j in range(q_star):
        sigma2_w = np.diag(v_star_list[j] * np.ones(d))
        W_star_j = np .random.multivariate_normal(mu_w, sigma2_w)
        W[:, j] = W_star_j

    X = np.dot(W, Z_star) + np.random.normal(0,
                                             np.sqrt(sigma2_star), [d, n_sample])

    return X, W, Z_star, sigma2_star, v_star_list


def geweke(iterations, d, q_star, prior_param, init_dict, n_sample, xi=None, thining = 50):
    """
        Function to simulate ppca data using X= WZ + \sigma^2 I_n
        param: iteration, number of gibbs steps
        param: q_star, true dimension of principle components
        param: n_sample, numebr of observations
        param: xi, the power posterior parameter
        param: prior_param, dictonary of prior setup
        param: d, dimension of the data x
    """

    forward_results = []
    W_list = []
    Z_list = []
    sigma2_list = []
    v_star_list = []
    for i in range(iterations):
        sample_result = forward_sample(d, q_star, n_sample, prior_param)
        X_i = sample_result[0]
        W_list.append(sample_result[1])
        Z_list.append(sample_result[2])
        sigma2_list.append(sample_result[3])
        v_star_list.append(sample_result[4])
        forward_results.append(np.mean(np.std(X_i, axis=1)))

    gibbs_results = []
    X_i_origin = forward_sample(d, q_star, n_sample, prior_param)[0]
    inference = Model(X_i_origin, init_dict, iterations, q_star, prior_param, xi=xi)
    inference.sigma2_list = sigma2_list
    inference.W_list = W_list
    inference.Z_list = Z_list
    inference.v_list = v_star_list

    count = 0
    X_i = X_i_origin.copy()
    while len(gibbs_results) != iterations:
        count += 1
        inference.gibbs_step(X_i)
        X_i = inference.sample_x()
        if count % thining == 0:
            gibbs_results.append(np.mean(np.std(X_i, axis=1)))

    return forward_results, gibbs_results


if __name__ == '__main__':

    d = 5
    q_star = d - 1
    q = d - 1
    xi = 1

    a_vj = 0.5 * d * np.ones(q) + 1
    epislon = 0.1

    #a_vj = 10 * d * np.ones(q)
    prior_param = dict({'beta_sigma2': 0.5,
                        'a_sigma2': 3,
                        'a_vj': a_vj,
                        'beta_vj': epislon * (a_vj - 1)
                        })

    init_dict = dict({'Z0': np.random.normal(0, 1, [q, 1]),
                      "sigma20": np.random.gamma(3, 0.1),
                      "w0": np.random.normal(0, 1, [d, q]),
                      "v0": np.random.gamma(1, 2, d - 1)})

    iterations = 500
    n_sample = 100
    forward_results, gibbs_results = geweke(
        iterations, d, q_star, prior_param, init_dict, n_sample)

    plt.scatter(np.sort(np.array(forward_results)),
                np.sort(np.array(gibbs_results)))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


    print(np.mean(np.array(forward_results)) / np.mean(gibbs_results))


    sns.distplot(np.array(forward_results))
    sns.distplot(np.array(gibbs_results))
    plt.show()
    
    a = gibbs_results
    b = np.mean(gibbs_results)
    plt.acorr(a - b, normed=True, usevlines=False, maxlags=10, label=u'thinned')
    plt.show()
