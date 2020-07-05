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


def generate_data(d, q_star, n_sample, sigma2_star, a_star_list):
    """
        Function to simulate ppca data using X= WZ + \sigma^2 I_n
        param: d, dimension of data
        param: q_star, true dimension of principle components
        param: n_sample, numebr of observations
        param: a_star_list, jx1 vector, 1/variance parameter to generate W_j ~ N(0, 1/a_j I_d)
    """
    mu_z = np.zeros([q_star])
    sigma2_z = np.diag(np.ones([q_star]))
    Z_star = np.random.multivariate_normal(mu_z, sigma2_z, n_sample).T

    mu_w = np.zeros([d])
    W = np.zeros([d, q_star])
    for j in range(q_star):
        sigma2_w = np.diag(a_star_list[j] * np.ones(d))
        W_star_j = np .random.multivariate_normal(mu_w, sigma2_w)
        W[:, j] = W_star_j

    X = np.dot(W, Z_star) + np.random.normal(0, sigma2_star, [d, n_sample])

    return X


if __name__ == '__main__':

    d = 5
    q_star = 2
    n_sample = 1000
    sigma2_star = 1
    a_star_list = 1 / np.linspace(1, 10, q_star)

    X = generate_data(d, q_star, n_sample, sigma2_star, a_star_list)

    print(X)
