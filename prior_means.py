"""Collection of one-dimensional prior mean functions for Gaussian process regression.
Add more prior means if needed.

Args:
    x (np.ndarray): Input data.
    a (float or np.ndarray): Coefficients of the polynomial mean function.
    b (float): Lower boundary of the domain.
    c (float): Upper boundary of the domain.
    
Returns:
    (np.ndarray): Prior mean function evaluated at x.
"""

import numpy as np


# Polynomial mean for 1-dim. real-valued inputs
def poly_mean_1d(x, a=0.0):
    if np.isscalar(a):
        mean = a
    else:
        X = x.reshape(-1,1)**np.arange(len(a))
        mean = np.sum(a*X, axis=-1)
    return mean


# Sinusoidal mean for 1-dim. real-valued inputs x_i in x=(x_1,x_2,...)
def sin_mean_1d(x):
    return np.sin(x)


# Sinusoidal mean with Dirichlet boundaries for 1-dim. real-valued inputs x_i in x=(x_1,x_2,...)
def sin_mean_bnd_1d(x, b=-2, c=2):
    return np.sin(x)*np.abs(x-b)*np.abs(x-c)


# Sinusoidal mean with molecular boundaries for 1-dim. real-valued inputs x_i in x=(x_1,x_2,...)
def sin_mean_mol_bnd_1d(x, b=-1.25, c=1.25, d=2):
    return 3*np.sin(x)*np.exp(-1/d*(np.abs(x-b) + np.abs(x-c)))


#EOF