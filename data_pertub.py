#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Fri May 22 14:42:54 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
import seaborn as sns
import configparser
# np.set_printoptions(precision=2)


def forward_sample(d, q_star, n_sample, prior_param, verbose=False):
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
        print('true sigma2_star is' + str(sigma2_star))
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

    return X


def CRP(n_sample, alpha=5):
    """
        Function to sample a partition from Chinese Resturant Process
        params: n_sample, #of data
        params: alpha, divergence parameter        
        return: a list of partition 
    """

    count = []
    n = 0
    while n < n_sample:
        prob = np.zeros(len(count) + 1)

        for i in range(len(count)):  # for the exiesting tables:
            prob[i] = count[i] / (n + alpha)  # prob of i-th table assignment

        prob[-1] = alpha / (n + alpha)  # new table prob
        prob = prob / sum(prob)

        assignment = choice(range(len(prob)), p=prob)

        if assignment == len(count):  # new table created
            count.append(0)
        count[assignment] += 1
        n += 1

    return count

def sample_perturbation2(y_true, n_sample, alpha, noise =0.02):
    """
        Function to sample perturbation Po by taking a random draw of a Polya urn scheme 
        Dirichlet process mixture with base distribution PθI , default concentration parameter 500.
        params: w0, weights of probability
        params: mu0, weights of component gaussian mean
        params: sigma0, weights of component gaussian sigma  
        params: n_sample # of data
        parama: alpha, CRP divergence parameter
        return: pertubated sample
    """
    count = CRP(n_sample, alpha)
    
    repeat_sample = np.repeat(y_true[:,:len(count)], count, axis=1)
    
#    return repeat_sample
    return np.random.normal(repeat_sample, noise)
    
    
def sample_perturbation(data_config, noise=0.02):
    """
        Function to sample perturbation Po by taking a random draw of a Polya urn scheme 
        Dirichlet process mixture with base distribution PθI , default concentration parameter 500.
        params: w0, weights of probability
        params: mu0, weights of component gaussian mean
        params: sigma0, weights of component gaussian sigma  
        params: n_sample # of data
        parama: alpha, CRP divergence parameter
        return: pertubated sample
    """
    n_sample = int(data_config['n_sample'])
    alpha = float(data_config['alpha'])

    count = CRP(n_sample, alpha)

    d = int(data_config['d'])
    q_star = int(data_config['q_star'])
    a_vj = float(data_config['a_vj']) * np.ones(q_star)
    epislon = float(data_config['epislon'])
    beta_vj = epislon * (a_vj - 1)

    prior_param = dict({'beta_sigma2': 0.5,
                        'a_sigma2': 3,
                        'a_vj': a_vj,
                        'beta_vj': beta_vj
                        })

    unique_sample = [forward_sample(
        d, q_star, 1, prior_param).ravel() for _ in range(len(count))]

    repeat_sample = np.repeat(unique_sample, count, axis=0)

#    return np.random.normal(repeat_sample, 0.25)
    return repeat_sample.T
#    return np.random.normal(repeat_sample.T, noise)


if __name__ == '__main__':
    # read the data
    config = configparser.ConfigParser()
    config.sections()
    # for i in os.listdir('param2/'):
    config.read("10000.ini")
    data_config = dict(config['data_config'])

    n_sample = int(data_config['n_sample'])
    alpha = float(data_config['alpha'])

    count = CRP(n_sample, alpha)

    d = int(data_config['d'])
    q_star = int(data_config['q_star'])
    a_vj = float(data_config['a_vj']) * np.ones(q_star)
    epislon = float(data_config['epislon'])
    beta_vj = epislon * (a_vj - 1)

    prior_param = dict({'beta_sigma2': 0.5,
                        'a_sigma2': 3,
                        'a_vj': a_vj,
                        'beta_vj': beta_vj
                        })

    y_true = forward_sample(d, q_star, n_sample, prior_param, verbose=False)
    y_per = sample_perturbation(data_config)

    sns.distplot(y_true.ravel(), label='y_true')
    sns.distplot(y_per.ravel(), label='y_perturbation')
    plt.legend()
