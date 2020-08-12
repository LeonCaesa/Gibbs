#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Tue Jul 28 15:34:33 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import numpy as np
import pystan
from data_pertub import *
from mcmc_plot import *

ppca_code = """
data { 
    int D; //number of dimensions
    int N; //number of data
    int Q; //number of principle components
//    matrix[N,D] x; // data
    vector[D] x[N]; //data
    real a_vj; // w_j prior 
//    real epsilon;// w_j mean
    real beta_vj; //w_j prior
    real xi; // power parameter
    real a_sigma2; // sigma2 prior 
    real beta_sigma2;// sigma2 mean
 //   vector[Q] v; // true v_j
}

transformed data {
    matrix[D,D] S;
    S = x[1] * x[1]';
    
    for (n in 2:N){
    S += x[n] * x[n]';
    }
    S = S/N;
    
}
parameters {
//    vector[Q] v; // v_j
    ordered[Q] v; // v_j
    real<lower=0> sigma2; //data sigma2
    matrix[Q,D] W; //projection matrix
}
model {
    matrix[D,D] C; //covaraince matrix
    for(j in 1:Q){
//        v[j] ~ inv_gamma(a_vj, epsilon * (a_vj -1));
        v[j] ~ inv_gamma(a_vj, beta_vj);
        W[j] ~ multi_normal(rep_vector(0,D), v[j] * diag_matrix(rep_vector(1, D)));
        }
        
    sigma2 ~ inv_gamma(a_sigma2, beta_sigma2);
    
    C = crossprod(W)+ sigma2 * diag_matrix(rep_vector(1, D));
    

    target += - xi * N/2 *(log_determinant (C) + trace( C\S));
}

generated quantities {
    vector[D] y_hat[N]; //predictive
    for (n in 1:N) {
        y_hat[n] = multi_normal_rng(rep_vector(0,D), crossprod(W)+ sigma2 * diag_matrix(rep_vector(1, D)));
    }
}

"""
if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.sections()
    # for i in os.listdir('param2/'):
    config.read("10000.ini")
    data_config = dict(config['data_config'])
    mcmc_setup = dict(config['mcmc_setup'])

    n_sample = int(data_config['n_sample'])
    alpha = float(data_config['alpha'])


    d = int(data_config['d'])
    q_star = int(data_config['q_star'])
    a_vj = float(data_config['a_vj']) * np.ones(q_star)
    epsilon = float(data_config['epsilon'])
    beta_vj = epsilon * (a_vj - 1)
    beta_sigma2 = float(data_config['beta_sigma2'])
    a_sigma2 = float(data_config['a_sigma2'])

    prior_param_true = dict({'beta_sigma2': beta_sigma2,
                             'a_sigma2': a_sigma2,
                             'a_vj': a_vj,
                             'beta_vj': beta_vj
                             })
    y_true, W_star, Z_star, sigma2_star, v_star_list = forward_sample(
        d, q_star, n_sample, prior_param_true, verbose=True)
    y_perturbation = sample_perturbation2(y_true, n_sample, alpha)
    #y_perturbation = forward_sample(d, q_star, n_sample, prior_param_true, verbose = True)

    for j in range(d):
        sns.distplot(y_true[j, :], label='y_true')
        sns.distplot(y_perturbation[j, :], label='y_perturbation')
        plt.legend()
        plt.title('Component ' + str(j+1))
        plt.show()

    X = y_true
    
    config = configparser.ConfigParser()
    config.sections()
    # for i in os.listdir('param2/'):
    config.read("10000.ini")
    mcmc_setup = dict(config['mcmc_setup'])

    # inference parameter
    d = int(data_config['d'])
    q = int(mcmc_setup['q'])
    xi = float(mcmc_setup['xi'])


    # prior parameter
    beta_sigma2 = float(mcmc_setup['beta_sigma2'])
    a_sigma2 = float(mcmc_setup['a_sigma2'])
    a_vj = float(mcmc_setup['a_vj']) * np.ones(q)
    epsilon = float(mcmc_setup['epsilon'])
    beta_vj = epsilon * (a_vj - 1)


    # sampling parameter
    prior_param_mcmc = dict({'beta_sigma2': beta_sigma2,
                             'a_sigma2': a_sigma2,
                             'a_vj': a_vj,
                             'beta_vj': beta_vj
                             })

    ppca_dat_standard = {'D': d,
                         'N': n_sample,
                         'Q': q,
                         'x': X.reshape([n_sample, d]),
                         'xi': 1,
                         'a_vj': a_vj[0],
                         'epsilon': epsilon,
                         'a_sigma2': a_sigma2,
                         'beta_sigma2': beta_sigma2
                         }
    ppca_dat_exact = ppca_dat_standard.copy()
    ppca_dat_exact['xi'] = xi

    n_chains = int(mcmc_setup['n_chains'])
    iterations = int(mcmc_setup['iterations'])
#    iterations = 50
    init_list = []
    for i_ in range(n_chains):
        temp_dict = {
            # 'v': np.repeat(v_star_list, q).ravel(),
             'v': sorted(v_star_list),
           # 'v': sorted(1/np.random.gamma(a_vj, 1 / beta_vj)),
            'sigma2': sigma2_star,
             "w0": W_star.T

        }
        init_list.append(temp_dict)
    sm = pystan.StanModel(model_code=ppca_code)

    fit_standard = sm.sampling(
        data=ppca_dat, iter=iterations, chains=n_chains, init=init_list)

    fit_exact = sm.sampling(data=ppca_dat, iter=iterations,
                            chains=n_chains, init=init_list)


ppca_standard2 = """
data { 
    int D; //number of dimensions
    int N; //number of data
    int Q; //number of principle components
    vector[D] x[N]; //data
    real a_vj; // w_j prior 
    real epsilon;// w_j mean
    real xi; // power parameter 
    real a_sigma2; // sigma2 prior 
    real beta_sigma2;// sigma2 mean
 }

parameters {
    vector<lower=0>[Q] v; // v_j
    real<lower=0> sigma2; //data sigma2
    matrix[Q,D] W; //projection matrix
}

model {
    matrix[D,D] C; //covaraince matrix
    matrix[D,D] L_C; //covaraince matrix


    for(j in 1:Q){
        v[j] ~ inv_gamma(a_vj, epsilon * (a_vj -1));
        W[j] ~ multi_normal(rep_vector(0,D), v[j] * diag_matrix(rep_vector(1, D)));
    }

    sigma2 ~ inv_gamma(a_sigma2, beta_sigma2);
    C = crossprod(W)+ sigma2 * diag_matrix(rep_vector(1, D));
    L_C = cholesky_decompose(C);

    for(n in 1:N){
    x[n] ~ multi_normal_cholesky(rep_vector(0, D), L_C);
    }
}
"""
# for(n in 1:N){
# x[n] ~ multi_normal_cholesky(rep_vector(0, D), L_C);
# }
# }
