#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Thu Jul 23 13:55:46 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""
import numpy as np
import pystan
from data_pertub import *
from mcmc_plot import *


ppca_standard = """
data { 
 int D; //number of dimensions
 int N; //number of data
 int Q; //number of principle components
 vector[D] x[N]; //data
 real a_vj; // w_j prior 
 real epislon;// w_j mean
 
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
    v[j] ~ inv_gamma(a_vj, epislon * (a_vj -1));
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


"""
Frame work
"""



def parse_data_config(dict_config):
    data_config = {}
    for item in dict_config:
        try:
            data_config.update({item: int(dict_config[item])})
        except:
            data_config.update(
                {item: np.array(dict_config[item][1:-1].split(','), dtype=float)})

    return data_config


def model_run(model_names, path, config_dict):

    data_config = config_dict['data_config']
    mcmc_setup = config_dict['mcmc_setup']

    pertubation_data = sample_perturbation2(data_config)
    true_data = [mixture_generation(data_config)
                 for i in range(data_config['n_sample'])]

    mixture_dat = {'N': data_config['n_sample'],
                   'y': pertubation_data.reshape([data_config['n_sample'], 1]),
                   'D': 1,
                   'K': mcmc_setup['cluster_maximum'],
                   'alpha': model_setup['gem_alpha'],
                   'power': model_setup['power']
                   }

    with open(path + '/data' + str(data_config['n_sample']) + '.pickle', "wb") as f:
        pickle.dump(
            {'true': true_data, 'pertubation': pertubation_data}, f, protocol=-1)

    if 'fit_standard' in model_names:
        sm_standard = pystan.StanModel(model_code=mixture_model_standard)
        fit_standard = sm_standard.sampling(
            data=mixture_dat, iter=mcmc_setup['iterations'], chains=mcmc_setup['n_chains'])
        with open(path + '/fit_standard'+str(data_config['n_sample'])+'.pickle', "wb") as f:
            pickle.dump({'model': sm_standard,
                         'fit': fit_standard}, f, protocol=-1)

        print('Finished Standard Fitting')
    if 'fit_exact' in model_names:
        sm_correct = pystan.StanModel(model_code=mixture_model_correct)
        fit_correct = sm_correct.sampling(
            data=mixture_dat, iter=mcmc_setup['iterations'], chains=mcmc_setup['n_chains'])
        with open(path + '/fit_exact'+str(data_config['n_sample'])+'.pickle', "wb") as f:
            pickle.dump({'model': sm_correct,
                         'fit': fit_correct}, f, protocol=-1)

        print('Finished Correct Fitting')
    if 'fit_approx' in model_names:
        sm_approx = pystan.StanModel(model_code=mixture_model_approx)
        fit_approx = sm_approx.sampling(
            data=mixture_dat, iter=mcmc_setup['iterations'], chains=mcmc_setup['n_chains'])
        with open(path + '/fit_approx'+str(data_config['n_sample'])+'.pickle', "wb") as f:
            pickle.dump({'model': sm_approx,
                         'fit': fit_approx}, f, protocol=-1)

        print('Finished Approxmiate Fitting')


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.sections()
    for i in os.listdir('param/'):
        try:
            print(i)
            config.read("param/" + str(i))
            mcmc_setup = {k: int(v)
                          for k, v in dict(config['mcmc_setup']).items()}
            model_setup = {k: float(v) for k, v in dict(
                config['model_setup']).items()}
            data_config = parse_data_config(dict(config['data_config']))

            config_dict = dict(data_config=data_config,
                               mcmc_setup=mcmc_setup,
                               model_setup=model_setup,
                               )

            pertubation_data = sample_perturbation(data_config)
            true_data = [mixture_generation(data_config)
                         for i in range(data_config['n_sample'])]

            mixture_dat = {'N': data_config['n_sample'],
                           'y': pertubation_data.reshape([data_config['n_sample'], 1]),
                           'D': 1,
                           'K': mcmc_setup['cluster_maximum'],
                           'alpha': model_setup['gem_alpha'],
                           'power': model_setup['power']
                           }
            path = "/projectnb2/powermc/robustvb/Gibbs/param"

           # model_names = ['fit_standard', 'fit_correct', 'fit_approx']
            model_names = [ 'fit_exact']

            model_run(model_names, path, config_dict)
        except:
            print('entered')
            continue
