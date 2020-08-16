# Gibbs
Gibbs sampling for PPCA


Implemented gibbs sampling for Probalistic PCA.

Mainly to replicate the idea of [Bayesian PCA](https://papers.nips.cc/paper/1549-bayesian-pca.pdf)

Prior on $\alpha_j$ is implemented as Gamma(a, b), all other setups are the same to the article


See [Perturbation_Consise](https://github.com/LeonCaesa/Gibbs/blob/master/Perturbation_Consise.ipynb) for the main result.


Also implemented the HMC through Pystan with result in [Match](https://github.com/LeonCaesa/Gibbs/blob/master/Match.ipynb)
