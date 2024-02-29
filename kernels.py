"""Collection of prior kernel / covariance functions for Gaussian process regression.
Add more kernels if needed.

Args:
    x1 (np.ndarray): Input data of shape (n, n_features).
    x2 (np.ndarray): Input data of shape (m, n_features).
    theta (float or np.ndarray): Hyperparameters of the kernel function.
    args (float or tuple or np.ndarray): Arguments (like for for non-stationary features, e.g., boundaries).
    diag (bool): If True, return only the diagonal of the kernel matrix. Default is False.
    
Returns:
    K (np.ndarray): Kernel matrix of the kernel function evaluated at x1 and x2 of shape (n, m).
    grad_K (list of np.ndarray): List of elementwise derivatives of the kernel matrix w.r.t. hyperparameters.
"""

import numpy as np


def is_scalar_or_length_one(var):
    """Check if a variable is a scalar or has length one."""
    if np.isscalar(var):
        return True
    elif hasattr(var, '__len__') and len(var) == 1:
        return True
    else:
        return False


### (Proper, real-valued) RBF kernels for real-valued inputs x1, x2 ###
def RBF_kernel(x1, x2, theta, diag=False):
    one_dim = False # Deal with multi-dimensional inputs.
    if np.isscalar(x1[0]):
        one_dim = True # Deal with 1-dim. inputs.
    
    if not np.isscalar(theta):
        if 1 < len(theta) != len(x1[0]):
            raise ValueError("Number of length-scales must equal 1 or input dimension.")
        theta = np.asarray(theta)
    
    if diag:
        dx = x1 - x2
        if one_dim:
            sqdist = dx*dx / theta**2
        else:
            sqdist = np.sum(dx*dx / theta**2, axis=-1)
        K = np.exp(-0.5 * sqdist)
        return K
    else:
        n = len(x1)
        m = len(x2)
        
        if one_dim:
            X1 = np.tile(x1,(m,1))
            X2 = np.tile(x2.reshape(-1,1),(1,n))
            sqdist = (X1 - X2)**2
            K = np.exp(-0.5 / theta**2 * sqdist)
            grad_K = [(K * sqdist / theta**3).T]
        else:
            X1 = np.tile(x1,(m,1,1))
            X2 = np.array(np.split(np.repeat(x2,n,axis=0),m))
            if is_scalar_or_length_one(theta): # spherical RBF kernel in n-dim.
                sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
                K = np.exp(-0.5 / theta**2 * sqdist)
                grad_K = [(K * sqdist / theta**3).T]
            else:
                dX = X1 - X2
                # ls = np.diag(1/theta**2)
                # sqdist = np.einsum('ijk,kij->ij', dX, np.einsum('ij,klj', ls, dX))
                sqdist = dX[:,:,0]**2 / theta[0]**2
                for i in range(1,len(theta)):
                    sqdist += dX[:,:,i]**2 / theta[i]**2
                K = np.exp(-0.5 * sqdist)
                grad_K = [(K * dX[:,:,i]**2 / theta[i]**3).T for i in range(len(theta))]
        return K.T, grad_K


### Rational quadratic kernel function ###
def RQ_kernel(x1, x2, theta, diag=False):
    one_dim = False # Deal with multi-dimensional inputs.
    if np.isscalar(x1[0]):
        one_dim = True # Deal with 1-dim. inputs.
    
    ls, alpha = theta
    
    if not np.isscalar(ls):
        if 1 < len(ls) != len(x1[0]):
            raise ValueError("Number of length-scales must equal 1 or input dimension.")
        ls = np.asarray(ls)
    
    if diag:
        dx = x1 - x2
        if one_dim:
            sqdist = dx*dx
        else:
            sqdist = np.sum(dx*dx, axis=-1)
        K = (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha)
        return K
    else:
        n = len(x1)
        m = len(x2)
        
        if one_dim:
            X1 = np.tile(x1,(m,1))
            X2 = np.tile(x2.reshape(-1,1),(1,n))
            sqdist = (X1 - X2)**2
            K = (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha)
            grad_K = [((1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha-1) * sqdist / ls**3).T, 
                      (K*(sqdist / (sqdist + 2*ls**2*alpha) - np.log(1 + 0.5 / (ls**2 * alpha) * sqdist))).T]
        else:
            X1 = np.tile(x1,(m,1,1))
            X2 = np.array(np.split(np.repeat(x2,n,axis=0),m))
            if is_scalar_or_length_one(ls): # spherical length-scale in n-dim.
                sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
                K = (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha)
                grad_K = [((1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha-1) * sqdist / ls**3).T, 
                            (K*(sqdist / (sqdist + 2*ls**2*alpha) - np.log(1 + 0.5 / (ls**2 * alpha) * sqdist))).T]
    return K.T, grad_K


### Experimental RBF kernels with various non-stationary features. ###

# RBF kernel in 1-dim. with homogeneous (Dirichlet) boundary conditions: y(x=args[0]) = 0 = y(x=args[1])
def RBF_kernel_bnd_1d(x1, x2, theta, args=(-2, 2), diag=False):
    B   = np.abs(x1 - args[0]) * np.abs(x1 - args[1])
    B_p = np.abs(x2 - args[0]) * np.abs(x2 - args[1])
    
    if diag:
        dx = x1 - x2
        sqdist = dx*dx / theta**2
        # sqdist = np.sum(dx*dx / theta**2, axis=-1)
        K = B * np.exp(-0.5 * sqdist) * B_p
        return K
    else:
        n = len(x1)
        m = len(x2)
        X1 = np.tile(x1,(m,1))
        X2 = np.tile(x2.reshape(-1,1),(1,n))
        sqdist = (X1 - X2)**2
        K = B * np.exp(-0.5 / theta**2 * sqdist) * B_p.reshape(-1,1)
        grad_K = [(K * sqdist / theta**3).T]
        return K.T, grad_K

# RBF kernel in 1-dim. with molecular boundary conditions
def RBF_kernel_mol_1d(x1, x2, theta, args=(-1.25, 1.25), diag=False):
    ls, c = theta
    B   = np.exp(-1/c * (np.abs(x1 - args[0]) + np.abs(x1 - args[1])))
    B_p = np.exp(-1/c * (np.abs(x2 - args[0]) + np.abs(x2 - args[1])))
    
    if diag:
        dx = x1 - x2
        sqdist = dx*dx / ls**2
        # sqdist = np.sum(dx*dx / ls**2, axis=-1)
        K = B * np.exp(-0.5 * sqdist) * B_p
        return K
    else:
        n = len(x1)
        m = len(x2)
        X1 = np.tile(x1,(m,1))
        X2 = np.tile(x2.reshape(-1,1),(1,n))
        sqdist = (X1 - X2)**2
        K = B * np.exp(-0.5 / ls**2 * sqdist) * B_p.reshape(-1,1)
        grad_K = [(K * sqdist / ls**3).T, 
                (K * ((np.abs(x1 - args[0]) + np.abs(x1 - args[1])) + (np.abs(x2 - args[0]) + np.abs(x2 - args[1])).reshape(-1,1)) / c**2).T]
        return K.T, grad_K

# RBF kernel in n-dim. with homogeneous boundary conditions (cuboid box potential)
def RBF_kernel_bnd(x1, x2, theta, args=np.array([[0,1],[0,1]]), diag=False):
    if not np.isscalar(theta):
        if 1 < len(theta) != len(x1[0]):
            raise ValueError("Number of length-scales must equal 1 or input dimension.")
        theta = np.asarray(theta)
        
    B   = np.prod(np.abs(x1-args[:,0])*np.abs(x1-args[:,1]), axis=-1)
    B_p = np.prod(np.abs(x2-args[:,0])*np.abs(x2-args[:,1]), axis=-1)
    
    if diag:
        dx = x1 - x2
        sqdist = np.sum(dx*dx / theta**2, axis=-1)
        K = B * np.exp(-0.5 * sqdist) * B_p
        return K
    else:
        n = len(x1)
        m = len(x2)
        X1 = np.tile(x1,(m,1,1))
        X2 = np.array(np.split(np.repeat(x2,n,axis=0),m))
        if np.isscalar(theta) or len(theta) == 1: # spherical RBF kernel in n-dim.
            sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
            K = B * np.exp(-0.5 / theta**2 * sqdist) * B_p.reshape(-1,1)
            grad_K = [(K * sqdist / theta**3).T]
        else:
            dX = X1 - X2
            # ls = np.diag(1/theta**2)
            # sqdist = np.einsum('ijk,kij->ij', dX, np.einsum('ij,klj', ls, dX))
            sqdist = dX[:,:,0]**2 / theta[0]**2
            for i in range(1,len(theta)):
                sqdist += dX[:,:,i]**2 / theta[i]**2
            K = B * np.exp(-0.5 * sqdist) * B_p.reshape(-1,1)
            grad_K = [(K * dX[:,:,i]**2 / theta[i]**3).T for i in range(len(theta))]
        return K.T, grad_K

# RBF kernel in n-dim. with one-sided homogeneous boundary conditions (for EBK of hydrogen in B-field)
def RBF_kernel_bnd_HinB(x1, x2, theta, diag=False):
    if not np.isscalar(theta):
        if 1 < len(theta) != len(x1[0]):
            raise ValueError("Number of length-scales must equal 1 or input dimension.")
        theta = np.asarray(theta)
        
    B   = x1[:,0]
    B_p = x2[:,0]
    # B = np.prod(np.abs(x1), axis=-1); B_p = np.prod(np.abs(x2), axis=-1)
    
    if diag:
        dx = x1 - x2
        sqdist = np.sum(dx*dx / theta**2, axis=-1)
        K = B * np.exp(-0.5 * sqdist) * B_p
        return K
    else:
        n = len(x1)
        m = len(x2)
        X1 = np.tile(x1,(m,1,1))
        X2 = np.array(np.split(np.repeat(x2,n,axis=0),m))
        if np.isscalar(theta) or len(theta) == 1: # spherical RBF kernel in n-dim.
            sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
            K = B * np.exp(-0.5 / theta**2 * sqdist) * B_p.reshape(-1,1)
            grad_K = [(K * sqdist / theta**3).T]
        else:
            dX = X1 - X2
            # ls = np.diag(1/theta**2)
            # sqdist = np.einsum('ijk,kij->ij', dX, np.einsum('ij,klj', ls, dX))
            sqdist = dX[:,:,0]**2 / theta[0]**2
            for i in range(1,len(theta)):
                sqdist += dX[:,:,i]**2 / theta[i]**2
            K = B * np.exp(-0.5 * sqdist) * B_p.reshape(-1,1)
            grad_K = [(K * dX[:,:,i]**2 / theta[i]**3).T for i in range(len(theta))]
        return K.T, grad_K


#EOF