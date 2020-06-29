import numpy as np 
import seaborn as sns
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from data_simu import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def gibbs_scheme(X,init_dict, iterations, q, prior_param, xi = None):
        """
        Function to implement gibbs scheme
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
        if xi is None:
            xi = 1
        
        d = np.shape(X)[0]
        n_sample = np.shape(X)[1]
        # inference parameters
        beta_sigma2 = prior_param['beta_sigma2']
        a_sigma2 = prior_param['a_sigma2']
        a_aj = prior_param['a_aj']
        beta_aj = prior_param['beta_aj']

        
        # initialization
        Z0 = init_dict['Z0']
        sigma20 = init_dict['sigma20']
        W0 = init_dict['w0']
        alpha0 = init_dict['alpha0']

        sigma2_list = [sigma20]
        Z_list = [Z0]
        W_list = [W0]
        alpha_list = [alpha0]

        
        for j in range(iterations):

            # sampling for sigma2, scalar 
            alpha_sigma2_temp = n_sample * xi * d/2 + a_sigma2
            X_WZ = (X- np.dot(W_list[-1], Z_list[-1]))
            S_x = np.trace(np.dot(X_WZ.T, X_WZ))
            beta_sigma2_temp = (0.5 * (S_x * xi + 2 * beta_sigma2))
            sigma2_list.append(1/np.random.gamma(alpha_sigma2_temp, 1/beta_sigma2_temp))


            # sampling for (Z)_{qxn}
            C = xi/sigma2_list[-1] * np.dot( W_list[-1].T, W_list[-1]) + np.diag(np.ones([q]))

            first = np.linalg.inv(C)

            second = xi/sigma2_list[-1]* np.dot(W_list[-1].T, X)

            Z_hat = np.dot(first, second)

            Z_sigma2 = xi/sigma2_list[-1]* np.linalg.inv (np.dot(W_list[-1].T, W_list[-1]) + np.diag(np.ones(q)))

            Z_temp = np.random.normal(0, 1, [q, n_sample])

            Chol = np.linalg.cholesky(Z_sigma2)

            Z_list.append(np.dot(Chol , Z_temp)  + Z_hat)



            # sampling for (w_j)_{dx1}

            nominator = w_mu_nominator(X,Z_list)

            denominator = (xi / sigma2_list[-1] * alpha_list[-1] + np.sum(Z_list[-1]**2,axis=1))

            mu_w = nominator.T/denominator.T

            sigma2_w = sigma2_list[-1]/ (xi / sigma2_list[-1] * alpha_list[-1] + np.sum(Z_list[-1]**2,axis=1))

            sigma2_w_temp = [np.diag(np.repeat(i,d)) for i in sigma2_w]

            W_list.append(np.array(list(map(np.random.multivariate_normal, mu_w.T, sigma2_w_temp))).T)

            # sampling for (alpha)_{1xq}
            alpha_a = d/2 + a_aj 

            beta_a = 0.5* np.diag(np.dot(W_list[-1].T,W_list[-1])) + beta_aj

            alpha_list.append(np.random.gamma(alpha_a, 1/beta_a))

        return [sigma2_list, Z_list, W_list, alpha_list]

    
def w_mu_nominator(X,Z_list):
    """
        Function to calculate the mu vector of the conditional W matrix
    """
    q = np.shape(Z_list[-1])[0]
    d = np.shape(X)[0]
    w_temp = np.zeros([q,d])
    for j in range(q):
        w_temp[j,:]= np.sum(X*Z_list[-1][j,:],axis=1)
    return w_temp


    
if __name__ == '__main__':
    
    #data generation parameter
    d = 5
    q_star = 1
    n_sample = 1000
    sigma2_star = 1
    a_star_list = 1 / np.linspace(1,10,q_star)
    X = generate_data(d, q_star, n_sample, sigma2_star, a_star_list)

    plt.figure(figsize=(10,6))
    pd.plotting.scatter_matrix(pd.DataFrame(X).T)
    plt.xlabel('Component i')
    plt.ylabel('Component j')



    prior_param = dict({'beta_sigma2':2,
          'a_sigma2':10,
          'a_aj': 1 / np.linspace(1,10,q),
          'beta_aj':1 / np.linspace(1,10,q)    
    })

    init_dict = dict({'Z0':np.random.multivariate_normal(np.zeros([q]), np.diag(np.ones([q])), n_sample).T,
                "sigma20":1.5,
                "w0":np.random.normal(0, 0.6, [d,q]),
                "alpha0": np.ones(q)    
    })
    
    iterations = 1000
    
    infer_list = gibbs_scheme(X, init_dict, iterations, q, prior_param)

    sigma2_list = infer_list[0]
    Z_list = infer_list[1]
    W_list = infer_list[2]
    alpha_list = infer_list[3]
    
        
        
    
    
    print(pd.DataFrame(alpha_list).mean())
    pd.DataFrame(alpha_list).plot()
    #pd.DataFrame(W_list[-1]).plot.hist()
    plt.show()
    plt.plot(sigma2_list)


    
"""



# inference parameters
q = d-1
iterations = 1000
beta_sigma2 = 5
a_sigma2 = 10
a_aj = 1 / np.linspace(1,10,q)
beta_aj = 1 / np.linspace(1,10,q)

# initialization
Z0 = np.random.multivariate_normal(np.zeros([q]), np.diag(np.ones([q])), n_sample).T
sigma20 = 1.5
W0 = np.random.normal(0, 0.6, [d,q])
alpha0 = np.ones(q)

# inference trace list
sigma2_list = [sigma20]
Z_list = [Z0]
W_list = [W0]
alpha_list = [alpha0]




for j in range(iterations):
    
    # sampling from alpha, scalar 
    alpha_sigma2_temp = n_sample * d/2 + a_sigma2
    X_WZ = (X- np.dot(W_list[-1], Z_list[-1]))
    S_x = np.trace(np.dot(X_WZ.T, X_WZ))
    beta_sigma2_temp = (0.5 * (S_x + 2 * beta_sigma2))
    sigma2_list.append(1/np.random.gamma(alpha_sigma2_temp, 1/beta_sigma2_temp))
    
    
    # sampling for (Z)_{qxn}
    C = 1/sigma2_list[-1] * np.dot( W_list[-1].T, W_list[-1]) + np.diag(np.ones([q]))

    first = np.linalg.inv(C)
    
    second = 1/sigma2_list[-1]* np.dot(W_list[-1].T, X)
    
    Z_hat = np.dot(first, second)
    
    Z_sigma2 = 1/sigma2_list[-1]* np.linalg.inv (np.dot(W_list[-1].T, W_list[-1]) + np.diag(np.ones(q)))
    
    Z_temp = np.random.normal(0, 1, [q, n_sample])

    Chol = np.linalg.cholesky(Z_sigma2)
    
    Z_list.append(np.dot(Chol , Z_temp)  + Z_hat)
    

        
    # sampling for (w_j)_{dx1}
    
    nominator = w_mu_nominator(X,Z_list, d=d, q=q)
    
    denominator = (sigma2_list[-1] * alpha_list[-1] + np.sum(Z_list[-1]**2,axis=1))

    mu_w = nominator.T/denominator.T
    
    sigma2_w = sigma2_list[-1]/ (sigma2_list[-1] * alpha_list[-1] + np.sum(Z_list[-1]**2,axis=1))

    sigma2_w_temp = [np.diag(np.repeat(i,d)) for i in sigma2_w]

    W_list.append(np.array(list(map(np.random.multivariate_normal, mu_w.T, sigma2_w_temp))).T)
    
    # sampling for (alpha)_{1xq}
    alpha_a = d/2 + a_aj 
    
    beta_a = 0.5* np.diag(np.dot(W_list[-1].T,W_list[-1])) + beta_aj
    
    alpha_list.append(np.random.gamma(alpha_a, 1/beta_a))
        
    
"""       
    





"""
temp=0
for i in range(n_sample):

    temp+= np.sum((X[:,i]-np.dot(W_list[-1],Z_list[-1][:,i]))**2)
    
"""