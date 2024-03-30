"""Collection of prior kernel / covariance functions for Gaussian 
process regression.

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
            Xtrain (np.ndarray): 
                2D array of training data with shape 
                (n_samples, n_features).
        """

        self.Xtrain = Xtrain

        if len(self.Xtrain.shape) == 1:
            self.Xtrain = self.Xtrain.reshape(-1, 1)
        self.x_dim = len(self.Xtrain[0])

        # Set hyperparameters and bounds for optimization
        lengthscale = 1.0
        bnds_lengthscale = [(1e-3, 1e2)]
        self.hyperparams = lengthscale * np.ones(self.x_dim)
        self.bnds_hyperparams = bnds_lengthscale * self.x_dim

        # Set non-stationary kernel features with fixed parameters 
        # (optional)
        self.nonstat_fct = None

    
    def matrix(self, Xtest=None, gradient=True, diag=False):
        """Compute the kernel matrix of the RBF kernel function.

        Args:
            Xtest (np.ndarray, optional): 
                2D array of test data with shape 
                (n_samples, n_features). If provided, the kernel of the 
                training and test data K(Xtrain, Xtest) is computed. 
                If None, the kernel of the training data 
                K(Xtrain, Xtrain) is computed. Defaults to None.
            gradient (bool, optional): 
                If True, return grad_K. Default is True.
            diag (bool, optional): 
                If True, return only the diagonal of the kernel matrix 
                K(Xtest, Xtest). Default is False.
        
        Returns:
            K (np.ndarray): 
                Kernel matrix of the kernel function evaluated at Xtrain
                (and Xtest) of shape (n_samples, n_samples).
            grad_K (list of np.ndarray): 
                List of elementwise derivatives of the kernel matrix 
                w.r.t. hyperparameters.
        """
        
        if Xtest is None:
            # Compute kernel matrix K(Xtrain, Xtrain)
            Xtest = self.Xtrain
        else:
            if len(Xtest.shape) == 1:
                Xtest = Xtest.reshape(-1, 1)

        if self.nonstat_fct is not None:
            nonstat_1 = self.nonstat_fct(Xtest)
            nonstat_2 = self.nonstat_fct(self.Xtrain)
        
        if diag:
            if self.nonstat_fct is None:
                # Diagonal of RBF kernel matrix K(Xtest, Xtest) are ones
                K = np.ones(len(Xtest))
            else:
                K = nonstat_1 * nonstat_1
            return K
        else:
            n = len(Xtest)
            m = len(self.Xtrain)
            
            # if self.x_dim == 1:
            #     X1 = np.tile(Xtest,(m,1))
            #     X2 = np.tile(self.Xtrain.reshape(-1,1),(1,n))
            #     sqdist = (X1 - X2)**2
            #     K = np.exp(-0.5 / self.hyperparams**2 * sqdist)
            #     if self.nonstat_fct is not None:
            #         K = nonstat_1 * K * nonstat_2.reshape(-1,1)
            #     if gradient:
            #         grad_K = [K * sqdist / self.hyperparams**3]
            # else:
            X1 = np.tile(Xtest,(m,1,1))
            X2 = np.array(np.split(np.repeat(self.Xtrain,n,axis=0),m))
            if is_scalar_or_length_one(self.hyperparams):
                # Spherical RBF kernel in n-dim.
                sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
                K = np.exp(-0.5 / self.hyperparams**2 * sqdist)
                if self.nonstat_fct is not None:
                    K = nonstat_1 * K * nonstat_2.reshape(-1,1)
                if gradient:
                    grad_K = [K * sqdist / self.hyperparams**3]
            else:
                # Non-spherical RBF kernel in n-dim.
                dX = X1 - X2
                sqdist = dX[:,:,0]**2 / self.hyperparams[0]**2
                for i in range(1,len(self.hyperparams)):
                    sqdist += dX[:,:,i]**2 / self.hyperparams[i]**2
                K = np.exp(-0.5 * sqdist)
                if self.nonstat_fct is not None:
                    K = nonstat_1 * K * nonstat_2.reshape(-1,1)
                if gradient:
                    grad_K = [
                        K * dX[:,:,i]**2 / self.hyperparams[i]**3 
                        for i in range(len(self.hyperparams))
                    ]

            if gradient:
                return K, grad_K
            else:
                return K


class RQ_kernel:
    """RQ covariance function"""
    
    def __init__(self, Xtrain):
        """Initialize RQ kernel hyperparameters.

        Args:
            Xtrain (np.ndarray): 
                2D array of training data with shape 
                (n_samples, n_features).
        """

        self.Xtrain = Xtrain

        if len(self.Xtrain.shape) == 1:
            self.Xtrain = self.Xtrain.reshape(-1, 1)
        self.x_dim = len(self.Xtrain[0])

        # Set hyperparameters and bounds for optimization
        lengthscale = 1.0
        bnds_lengthscale = [(1e-5, 1e2)]
        alphascale = 1.0
        bnds_alphascale = [(1e-5, 1e2)]
        self.hyperparams = np.array([lengthscale, alphascale])
        self.bnds_hyperparams = bnds_lengthscale + bnds_alphascale

        # Set non-stationary kernel features with fixed parameters 
        # (optional)
        self.nonstat_fct = None

    
    def matrix(self, Xtest=None, gradient=True, diag=False):
        """Compute the kernel matrix of the RQ kernel function.

        Args:
            Xtest (np.ndarray, optional): 
                2D array of test data with shape 
                (n_samples, n_features). If provided, the kernel of the 
                training and test data K(Xtrain, Xtest) is computed. 
                If None, the kernel of the training data 
                K(Xtrain, Xtrain) is computed. Defaults to None.
            gradient (bool, optional): 
                If True, return grad_K. Default is True.
            diag (bool, optional): 
                If True, return only the diagonal of the kernel matrix 
                K(Xtest, Xtest). Default is False.
        
        Returns:
            K (np.ndarray): 
                Kernel matrix of the kernel function evaluated at Xtrain
                (and Xtest) of shape (n_samples, n_samples).
            grad_K (list of np.ndarray): 
                List of elementwise derivatives of the kernel matrix 
                w.r.t. hyperparameters.
        """

        if Xtest is None:
            Xtest = self.Xtrain
        else:
            if len(Xtest.shape) == 1:
                Xtest = Xtest.reshape(-1, 1)
            
        ls, alpha = self.hyperparams
        
        if self.nonstat_fct is not None:
            nonstat_1 = self.nonstat_fct(Xtest)
            nonstat_2 = self.nonstat_fct(self.Xtrain)
        
        if diag:
            if self.nonstat_fct is None:
                # Diagonal of RQ kernel matrix K(Xtest, Xtest) are ones
                K = np.ones(len(Xtest))
            else:
                K = nonstat_1 * nonstat_1
            return K
        else:
            n = len(Xtest)
            m = len(self.Xtrain)
            
            # if self.x_dim == 1:
            #     X1 = np.tile(Xtest,(m,1))
            #     X2 = np.tile(self.Xtrain.reshape(-1,1),(1,n))
            #     sqdist = (X1 - X2)**2
            #     K = (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha)
            #     if self.nonstat_fct is not None:
            #         K = nonstat_1 * K * nonstat_2.reshape(-1,1)
            #     if gradient:
            #         grad_K = [
            #             (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha-1)
            #             * sqdist / ls**3, 
            #             K * (sqdist / (sqdist + 2*ls**2*alpha) 
            #                  - np.log(1 + 0.5 / (ls**2 * alpha) * sqdist))
            #         ]
            # else:
            X1 = np.tile(Xtest,(m,1,1))
            X2 = np.array(np.split(np.repeat(self.Xtrain,n,axis=0),m))
            if is_scalar_or_length_one(ls):
                # Spherical length-scale in n-dim.
                sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
                K = (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha)
                if self.nonstat_fct is not None:
                    K = nonstat_1 * K * nonstat_2.reshape(-1,1)
                if gradient:
                    grad_K = [
                        (1 + 0.5 / (ls**2 * alpha) * sqdist)**(-alpha-1) 
                        * sqdist / ls**3, 
                        K*(sqdist / (sqdist + 2*ls**2*alpha) 
                            - np.log(1 + 0.5 / (ls**2 * alpha) * sqdist))
                    ]
                        
            if gradient:
                return K, grad_K
            else:
                return K



### Non-stationary RBF kernels ###

class RBF_exp_decay_kernel:
    """RBF covariance function with exponential decay away from centers.
    """
    
    def __init__(self, Xtrain, centers=None):
        """Initialize RBF kernel hyperparameters.

        Args:
            Xtrain (np.ndarray): 
                2D array of training data with shape 
                (n_samples, n_features).
            centers (np.ndarray): 
                2D array of centers with shape (n_centers, n_features). 
                Defaults to None, in which case a single center is 
                placed in the origin.
        """

        self.Xtrain = Xtrain

        if len(self.Xtrain.shape) == 1:
            self.Xtrain = self.Xtrain.reshape(-1, 1)
        self.x_dim = len(self.Xtrain[0])

        if centers is None:
            self.centers = np.zeros((1, self.x_dim))

        # Set hyperparameters and bounds for optimization
        lengthscale = 1.0
        bnds_lengthscale = [(1e-3, 1e2)]
        decay_rate = 1.0
        bnds_decay_rate = [(1e-3, 1e2)]
        self.hyperparams = np.concatenate((lengthscale * np.ones(self.x_dim), 
                                           [decay_rate]))
        self.bnds_hyperparams = bnds_lengthscale * self.x_dim + bnds_decay_rate

    
    def matrix(self, Xtest=None, gradient=True, diag=False):
        """Compute the kernel matrix.

        Args:
            Xtest (np.ndarray, optional): 
                2D array of test data with shape 
                (n_samples, n_features). If provided, the kernel of the 
                training and test data K(Xtrain, Xtest) is computed.
                If None, the kernel of the training data 
                K(Xtrain, Xtrain) is computed. Defaults to None.
            gradient (bool, optional): 
                If True, return grad_K. Default is True.
            diag (bool, optional): 
                If True, return only the diagonal of the kernel matrix 
                K(Xtest, Xtest). Default is False.
        
        Returns:
            K (np.ndarray): 
                Kernel matrix of the kernel function evaluated at Xtrain
                (and Xtest) of shape (n_samples, n_samples).
            grad_K (list of np.ndarray): 
                List of elementwise derivatives of the kernel matrix 
                w.r.t. hyperparameters.
        """
        
        if Xtest is None:
            # Compute kernel matrix K(Xtrain, Xtrain)
            Xtest = self.Xtrain
        else:
            if len(Xtest.shape) == 1:
                Xtest = Xtest.reshape(-1, 1)
            
        ls, c = self.hyperparams[:-1], self.hyperparams[-1]

        centers_reshaped = self.centers.reshape(len(self.centers), 
                                                1, self.x_dim)
        sum_norm_1 = np.sum(np.linalg.norm(Xtest - centers_reshaped, axis=-1), 
                            axis=0)
        sum_norm_2 = np.sum(np.linalg.norm(self.Xtrain - centers_reshaped, 
                                           axis=-1), axis=0)
        nonstat_1 = np.exp(-c * sum_norm_1)
        nonstat_2 = np.exp(-c * sum_norm_2)

        if diag:
            K = nonstat_1 * nonstat_1
            return K
        else:
            n = len(Xtest)
            m = len(self.Xtrain)
            
            # if self.x_dim == 1:
            #     X1 = np.tile(Xtest, (m,1))
            #     X2 = np.tile(self.Xtrain.reshape(-1,1), (1,n))
            #     sqdist = (X1 - X2)**2
            #     K = (nonstat_1 * np.exp(-0.5 / ls**2 * sqdist) 
            #          * nonstat_2.reshape(-1,1))
            #     if gradient:
            #         grad_K = [K * sqdist / ls**3, 
            #                   -K * (sum_norm_1 + sum_norm_2.reshape(-1,1))]
            # else:
            X1 = np.tile(Xtest,(m,1,1))
            X2 = np.array(np.split(np.repeat(self.Xtrain,n,axis=0),m))
            if is_scalar_or_length_one(ls):
                # Spherical RBF kernel in n-dim.
                sqdist = np.linalg.norm(X1 - X2, axis=-1)**2
                K = (nonstat_1 * np.exp(-0.5 / ls**2 * sqdist) 
                        * nonstat_2.reshape(-1,1))
                if gradient:
                    grad_K = [K * sqdist / ls**3,
                                -K * (sum_norm_1 + sum_norm_2.reshape(-1,1))]
            else:
                dX = X1 - X2
                sqdist = dX[:,:,0]**2 / ls[0]**2
                for i in range(1,len(ls)):
                    sqdist += dX[:,:,i]**2 / ls[i]**2
                K = (nonstat_1 * np.exp(-0.5 * sqdist) 
                        * nonstat_2.reshape(-1,1))
                if gradient:
                    grad_K = [(K * dX[:,:,i]**2 / ls[i]**3 
                                for i in range(len(ls))), 
                                -K * (sum_norm_1 + sum_norm_2.reshape(-1,1))]

            if gradient:
                return K, grad_K
            else:
                return K


#EOF