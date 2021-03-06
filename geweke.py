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
from data_simu import generate_data

import numpy as np
import matplotlib.pyplot as plt


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
    sigma2_star = 1 / \
        np.random.gamma(prior_param['a_sigma2'],
                        1 / prior_param['beta_sigma2'])

    # sampling for alpha
    v_star_list = 1 / \
        np.random.gamma(prior_param['a_vj'], 1 / prior_param['beta_vj'])

  #  print(sigma2_star)
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


def forward_only_Z(W, n_sample, sigma2_star, prior_param):

    #    Z_star = np.random.multivariate_normal(mu_z, sigma2_z, n_sample).T
    Z_star = np.random.normal(0, 1, [q, n_sample])
    X = np.dot(W, Z_star) + np.random.normal(0,
                                             np.sqrt(sigma2_star), [d, n_sample])

    return X


def forward_only_sigma2_Z(W, n_sample, prior_param):

    #    Z_star = np.random.multivariate_normal(mu_z, sigma2_z, n_sample).T
    Z_star = np.random.normal(0, 1, [q, n_sample])
    sigma2_star = 1 / \
        np.random.gamma(prior_param['a_sigma2'],
                        1 / prior_param['beta_sigma2'])
    X = np.dot(W, Z_star) + np.random.normal(0,
                                             np.sqrt(sigma2_star), [d, n_sample])

    return X


def forward_only_sigma2_Z_W(n_sample, prior_param, v_star_list):

    mu_w = np.zeros([d])
    W = np.zeros([d, q_star])
    for j in range(q_star):
        sigma2_w = np.diag(v_star_list[j] * np.ones(d))
        W_star_j = np .random.multivariate_normal(mu_w, sigma2_w)
        W[:, j] = W_star_j

    Z_star = np.random.normal(0, 1, [q, n_sample])
    sigma2_star = 1 / \
        np.random.gamma(prior_param['a_sigma2'],
                        1 / prior_param['beta_sigma2'])
    X = np.dot(W, Z_star) + np.random.normal(0,
                                             np.sqrt(sigma2_star), [d, n_sample])

    return X


def forward_only_W(n_sample, Z_star, sigma2_star, v_star_list):

    mu_w = np.zeros([d])
    W = np.zeros([d, q_star])
    for j in range(q_star):
        sigma2_w = np.diag(v_star_list[j] * np.ones(d))
        W_star_j = np .random.multivariate_normal(mu_w, sigma2_w)
        W[:, j] = W_star_j

    X = np.dot(W, Z_star) + np.random.normal(0,
                                             np.sqrt(sigma2_star), [d, n_sample])

    return X


def geweke(iterations, d, q_star, prior_param, init_dict, n_sample, xi=None):
    """
        Function to simulate ppca data using X= WZ + \sigma^2 I_n
        param: iteration, number of gibbs steps
        param: q_star, true dimension of principle components
        param: n_sample, numebr of observations
        param: xi, the power posterior parameter
        param: prior_param, dictonary of prior setup
        param: d, dimension of the data x
    """

    X, W, Z_star, sigma2_star, v_star_list = forward_sample(
        d, q_star, n_sample, prior_param)
    gibbs_results = []

    inference = Model(X, init_dict, iterations, q, prior_param, xi=xi)
    inference.sigma2_list = [sigma2_star]
    inference.W_list = [W]
    inference.Z_list = [Z_star]
    inference.v_list = [v_star_list]

    forward_results = []
    for i in range(iterations):
#        X_i = forward_only_sigma2_Z(W, n_sample, prior_param) #sampling only z sigam2
#        X_i = forward_only_Z(W, n_sample, sigma2_star, prior_param)
        X_i = forward_only_sigma2_Z_W(n_sample, prior_param, v_star_list)
#        X_i = forward_only_W( n_sample, Z_star, sigma2_star, v_star_list)
        forward_results.append(np.mean(np.std(X_i, axis=1)))

    count = 0
    while len(gibbs_results) != iterations:
        inference.gibbs_step(X_i)
        X_i = inference.sample_x()
        count += 1

        if count % 100 == 0:
            gibbs_results.append(np.mean(np.std(X_i, axis=1)))

    return forward_results, gibbs_results


if __name__ == '__main__':

    d = 5
    q_star = d - 1
    q = d - 1
    xi = 1

    a_vj = 0.5 * d * np.ones(q) + 1
    epislon = 0.1

    prior_param = dict({'beta_sigma2': 0.5,
                        'a_sigma2': 3,
                        'a_vj': a_vj,
                        'beta_vj': epislon * (a_vj - 1)
                        })

    init_dict = dict({'Z0': np.random.normal(0, 1, [q, 1]),
                      "sigma20": np.random.gamma(3, 0.1),
                      "w0": np.random.normal(0, 1, [d, q]),
                      "v0": np.random.gamma(1, 2, d - 1)})

    iterations = 1000
    n_sample = 50
    forward_results, gibbs_results = geweke(
        iterations, d, q_star, prior_param, init_dict, n_sample)

    indx = 0
    plt.scatter(np.sort(np.array(forward_results)),
                np.sort(np.array(gibbs_results)))
    plt.xlabel('True Sample')
    plt.ylabel('MCMC Sample')
    plt.show()


print(np.mean(np.array(forward_results)) / np.mean(gibbs_results))


sns.distplot(np.array(forward_results), label='Forward')
sns.distplot(np.array(gibbs_results), label='MCMC')
plt.legend()
plt.show()


a = gibbs_results
b = np.mean(gibbs_results)
plt.acorr(a - b, normed=True, usevlines=False, maxlags=10, label=u'thinned')
plt.show()
