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
epislon = float(data_config['epislon'])
beta_vj = epislon * (a_vj - 1)
beta_sigma2 = float(data_config['beta_sigma2'])
a_sigma2 = float(data_config['a_sigma2'])

prior_param_true = dict({'beta_sigma2': beta_sigma2,
                    'a_sigma2': a_sigma2,
                    'a_vj': a_vj,
                    'beta_vj': beta_vj
                    })


y_true = forward_sample(d, q_star, n_sample, prior_param_true, verbose = True)
y_perturbation = sample_perturbation2(y_true, n_sample, 500)
#y_perturbation = forward_sample(d, q_star, n_sample, prior_param_true, verbose = True)

for j in range(d):    
    sns.distplot(y_true[j,:], label='y_true')
    sns.distplot(y_perturbation[j,:], label='y_perturbation')
    plt.legend()
    plt.title('Component '+ str(j+1))
    plt.show()
    
X = y_perturbation    
#int<lower=1> N;
#       int<lower=1> D;
#       vector[D] x[N];
#}


#generated quantities {
# vector[N] x;
# 
# 
#x = mu + L * eta;
#
#}

#
#transformed parameters{
#for(j in 1:Q)
#  {
#    W[j] = v[j]*(1-v[j-1])*theta[j-1]/v[j-1]; //projection matrix
#  }   
#}
#   
ppca_standard = """
data { 
 int D; //number of dimensions
 int Q; //number of principle components
 int N; //number of data
 vector[D] x[N]; //data

 real a_vj; // w_j prior 
 real epislon;// w_j mean
 
 real a_sigma2; // sigma2 prior 
 real beta_sigma2;// sigma2 mean
 
 }

parameters {
 real <lower=0,upper=1> v[Q]; // v_j
 real<lower=0> sigma2; //data sigma2
}

  
model {
eta ~ std_normal(); 
for(k in 1:Q){
    v[k] ~ inv_gamma(a_vj, epislon * (a-vj -1));
    W[j] ~ v[k] * eta;
}


sigma2 ~ inv_gamma(2, 2);
C = dot_product(W, W')+ sigma2 * eta;
L_C = cholesky_decompose(C);
x ~ multi_normal_cholesky(0, L_C);
}
"""


# inference parameter
d = int(data_config['d'])
q = int(mcmc_setup['q'])
xi = float(mcmc_setup['xi'])


# prior parameter
beta_sigma2 = float(mcmc_setup['beta_sigma2'])
a_sigma2 = float(mcmc_setup['a_sigma2'])
a_vj = float(mcmc_setup['a_vj']) * np.ones(q_star)
epislon = float(mcmc_setup['epislon'])
beta_vj = epislon * (a_vj - 1)


# sampling parameter
prior_param_mcmc = dict({'beta_sigma2': beta_sigma2,
                    'a_sigma2': a_sigma2,
                    'a_vj': a_vj,
                    'beta_vj': beta_vj
                    })



ppca_dat = {'N': n_sample,
            'x': X.reshape([n_sample, d]),
            'D': d,
            'Q': q,
            'a_vj': a_vj,
            'epislon': epislon,
            'a_sigma2': a_sigma2,
            'beta_sigma2': beta_sigma2
                           }


n_chains = int(mcmc_setup['n_chains'])
iterations = int(mcmc_setup['iterations'])


sm_standard = pystan.StanModel(model_code=ppca_standard)
fit_standard = sm_standard.sampling(
    data=ppca_dat, iter=iterations, chains=n_chains)
