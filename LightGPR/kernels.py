"""Collection of prior kernel / covariance functions for Gaussian process regression.
Add more kernels if needed.
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


class RBF_kernel:
    """RBF covariance function"""
    
    def __init__(self, Xtrain):
        """Initialize RBF kernel hyperparameters.

        Args:
            Xtrain (np.ndarray): 2D array of training data with shape (n_samples, n_features).
        """

        self.Xtrain = Xtrain

        if len(self.Xtrain.shape) == 1:
            self.x_dim = 1
        else:
            self.x_dim = len(self.Xtrain[0])

        # Set hyperparameters and bounds for optimization
        lengthscale = 1.0
        bnds_lengthscale = [(1e-5, 1e2)]
        self.hyperparams = lengthscale * np.ones(self.x_dim)
        self.bnds_hyperparams = bnds_lengthscale * self.x_dim

    
    def matrix(self, Xtest=None, gradient=True, diag=False):
        """Compute the kernel matrix of the RBF kernel function.

        Args:
            Xtest (np.ndarray, optional): 2D array of test data with shape (n_samples, n_features). 
                                          If provided, the kernel of the training and test data K(Xtrain, Xtest) is computed.
                                          If None, the kernel of the training data K(Xtrain, Xtrain) is computed. Defaults to None.
            gradient (bool, optional): If True, return grad_K. Default is True.
            diag (bool, optional): If True, return only the diagonal of the kernel matrix K(Xtest, Xtest). Default is False.
        
        Returns:
            K (np.ndarray): Kernel matrix of the kernel function evaluated at Xtrain (and Xtest) of shape (n_samples, n_samples).
            grad_K (list of np.ndarray): List of elementwise derivatives of the kernel matrix w.r.t. hyperparameters.
        """
        
        if Xtest is None:
            Xtest = self.Xtrain
        
        if diag:
            # dx = Xtest - Xtest
            # if self.x_dim == 1:
            #     sqdist = dx*dx / self.hyperparams**2
            # else:
            #     sqdist = np.sum(dx*dx / self.hyperparams**2, axis=-1)
            # K = np.exp(-0.5 * sqdist)
            
            # Diagonals of RBF kernel matrix K(Xtest, Xtest) are ones
            K = np.ones(len(Xtest))
            return K
        else:
            n = len(Xtest)
            m = len(self.Xtrain)
            
            if self.x_dim == 1:
                X1 = np.tile(Xtest,(m,1))
                X2 = np.tile(self.Xtrain.reshape(-1,1),(1,n))
                sqdist = (X1 - X2)**2
                K = np.exp(-0.5 / self.hyperparams**2 * sqdist)
                if gradient:
                    grad_K = [K * sqdist / self.hyperparams**3]
            else:
                X1 = np.tile(Xtest,(m,1,1))
                X2 = np.array(np.split(np.repeat(self.Xtrain,n,axis=0),m))
                if is_scalar_or_length_one(self.hyperparams): # spherical RBF kernel in n-dim.
                    sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
                    K = np.exp(-0.5 / self.hyperparams**2 * sqdist)
                    if gradient:
                        grad_K = [K * sqdist / self.hyperparams**3]
                else:
                    dX = X1 - X2
                    # ls = np.diag(1/self.hyperparams**2)
                    # sqdist = np.einsum('ijk,kij->ij', dX, np.einsum('ij,klj', ls, dX))
                    sqdist = dX[:,:,0]**2 / self.hyperparams[0]**2
                    for i in range(1,len(self.hyperparams)):
                        sqdist += dX[:,:,i]**2 / self.hyperparams[i]**2
                    K = np.exp(-0.5 * sqdist)
                    if gradient:
                        grad_K = [K * dX[:,:,i]**2 / self.hyperparams[i]**3 for i in range(len(self.hyperparams))]

            if gradient:
                return K, grad_K
            else:
                return K


class RQ_kernel:
    """RQ covariance function"""
    
    def __init__(self, Xtrain):
        """Initialize RQ kernel hyperparameters.

        Args:
            Xtrain (np.ndarray): 2D array of training data with shape (n_samples, n_features).
        """

        self.Xtrain = Xtrain

        if len(self.Xtrain.shape) == 1:
            self.x_dim = 1
        else:
            self.x_dim = len(self.Xtrain[0])

        # Set hyperparameters and bounds for optimization
        lengthscale = 1.0
        bnds_lengthscale = [(1e-5, 1e2)]
        alphascale = 1.0
        bnds_alphascale = [(1e-5, 1e2)]
        self.hyperparams = np.array([lengthscale, alphascale])
        self.bnds_hyperparams = bnds_lengthscale + bnds_alphascale

    
    def matrix(self, Xtest=None, gradient=True, diag=False):
        """Compute the kernel matrix of the RQ kernel function.

        Args:
            Xtest (np.ndarray, optional): 2D array of test data with shape (n_samples, n_features). 
                                          If provided, the kernel of the training and test data K(Xtrain, Xtest) is computed.
                                          If None, the kernel of the training data K(Xtrain, Xtrain) is computed. Defaults to None.
            gradient (bool, optional): If True, return grad_K. Default is True.
            diag (bool, optional): If True, return only the diagonal of the kernel matrix K(Xtest, Xtest). Default is False.
        
        Returns:
            K (np.ndarray): Kernel matrix of the kernel function evaluated at Xtrain (and Xtest) of shape (n_samples, n_samples).
            grad_K (list of np.ndarray): List of elementwise derivatives of the kernel matrix w.r.t. hyperparameters.
        """
        
        if Xtest is None:
            Xtest = self.Xtrain
            
        ls, alpha = self.hyperparams
        
        if diag:
            # Diagonals of RQ kernel matrix K(Xtest, Xtest) are ones
            K = np.ones(len(Xtest))
            return K
        else:
            n = len(Xtest)
            m = len(self.Xtrain)
            
            if self.x_dim == 1:
                X1 = np.tile(Xtest,(m,1))
                X2 = np.tile(self.Xtrain.reshape(-1,1),(1,n))
                sqdist = (X1 - X2)**2
                K = (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha)
                if gradient:
                    grad_K = [(1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha-1) * sqdist / ls**3, 
                              K*(sqdist / (sqdist + 2*ls**2*alpha) - np.log(1 + 0.5 / (ls**2 * alpha) * sqdist))]
            else:
                X1 = np.tile(Xtest,(m,1,1))
                X2 = np.array(np.split(np.repeat(self.Xtrain,n,axis=0),m))
                if is_scalar_or_length_one(ls): # spherical length-scale in n-dim.
                    sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
                    K = (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha)
                    if gradient:
                        grad_K = [(1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha-1) * sqdist / ls**3, 
                                  K*(sqdist / (sqdist + 2*ls**2*alpha) - np.log(1 + 0.5 / (ls**2 * alpha) * sqdist))]
                        
            if gradient:
                return K, grad_K
            else:
                return K



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
        grad_K = [K * sqdist / theta**3]
        return K, grad_K

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
        grad_K = [K * sqdist / ls**3, 
                  K * ((np.abs(x1 - args[0]) + np.abs(x1 - args[1])) + (np.abs(x2 - args[0]) + np.abs(x2 - args[1])).reshape(-1,1)) / c**2]
        return K, grad_K

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
            grad_K = [K * sqdist / theta**3]
        else:
            dX = X1 - X2
            # ls = np.diag(1/theta**2)
            # sqdist = np.einsum('ijk,kij->ij', dX, np.einsum('ij,klj', ls, dX))
            sqdist = dX[:,:,0]**2 / theta[0]**2
            for i in range(1,len(theta)):
                sqdist += dX[:,:,i]**2 / theta[i]**2
            K = B * np.exp(-0.5 * sqdist) * B_p.reshape(-1,1)
            grad_K = [K * dX[:,:,i]**2 / theta[i]**3 for i in range(len(theta))]
        return K, grad_K

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
            grad_K = [K * sqdist / theta**3]
        else:
            dX = X1 - X2
            # ls = np.diag(1/theta**2)
            # sqdist = np.einsum('ijk,kij->ij', dX, np.einsum('ij,klj', ls, dX))
            sqdist = dX[:,:,0]**2 / theta[0]**2
            for i in range(1,len(theta)):
                sqdist += dX[:,:,i]**2 / theta[i]**2
            K = B * np.exp(-0.5 * sqdist) * B_p.reshape(-1,1)
            grad_K = [K * dX[:,:,i]**2 / theta[i]**3 for i in range(len(theta))]
        return K, grad_K


#EOF