import numpy as np
from scipy import optimize
from .kernels import *



class gp_reg:
    """Gaussian Process Regression (GPR) class"""
    
    def __init__(self, Xtrain, ytrain, kernel=RBF_kernel, prior_mean=None, ynoise=None):
        """Initialize GPR model by providing training data (Xtrain, ytrain) and setting prior kernel / covariance function and mean function.

        Args:
            Xtrain (np.ndarray): 2D array of training data with shape (n_samples, n_features).
            ytrain (np.ndarray): 1D array of training labels with shape (n_samples,).
            kernel (class, optional): Choice of prior kernel / covariance function from LightGPR.kernels. Defaults to RBF.
            prior_mean (func, optional): Prior mean function. Defaults to None, corresponding to zero mean.
            ynoise (float, optional): Fixed noise on training data. If set to None, it is treated as a variable hyperparameter. Defaults to None.
        """
        
        self.Xtrain = np.asarray(Xtrain)
        self.ytrain = np.asarray(ytrain)
        self.kernel = kernel(self.Xtrain)

        # Set prior mean function
        if prior_mean:
            self.prior_mean = prior_mean
        else:
            self.prior_mean = lambda x: np.zeros(len(x))
        
        # Set outputscale and noise hyperparameters
        self.outputscale = 1.0
        self.bnds_outputscale = [(1e-5, 1e4)]
        
        if ynoise is None:
            self.ynoise = 1e-8
            self.flag_train_noise = True
            self.bnds_ynoise = [(1e-10, 1.0)]
        else:
            self.ynoise = ynoise
            self.flag_train_noise = False
        
        self.bias_ynoise = 0.0 # Additional fixed (non-spherical) noise on training data (optional)

        self.loss = None # Negative log-likelihood


    def train(self):
        """Find optimal hyperparameters via maximum log-likelihood estimation method."""
        
        theta0 = np.concatenate(([self.outputscale], self.kernel.hyperparams)) # Initial hyperparameters
        bnds = self.bnds_outputscale + self.kernel.bnds_hyperparams
        if self.flag_train_noise:
            theta0 = np.concatenate((theta0, [self.ynoise]))
            bnds = bnds + self.bnds_ynoise
        
        sol = optimize.minimize(self.log_p, theta0, bounds=bnds, jac=True)
        theta_max = sol.x # Optimal hyperparameters: min(-log_p)
        # if not sol.success:
        #     print('Loss:', sol.fun, '; Failed;', sol.message, 'theta_max:', theta_max, '#iterations:', sol.nit)
        # else:
        #     print('Loss:', sol.fun, '; Success; theta_max:', theta_max, '#iterations:', sol.nit)
    
        # Set optimal hyperparameters
        self.outputscale = theta_max[0]
        if self.flag_train_noise:
            self.kernel.hyperparams = theta_max[1:-1]
            self.ynoise = theta_max[-1]
        else:
            self.kernel.hyperparams = theta_max[1:]
        self.loss = sol.fun
    
    
    def log_p(self, theta):
        """Compute the negative log-likelihood -log p(ytrain|Xtrain,theta) and its gradient w.r.t. theta (for minimalization).

        Args:
            theta (np.ndarray): 1D array of hyperparameters.

        Returns:
            -log_p (float): negative log-likelihood.
            -grad_log_p (np.ndarray): 1D array of negative gradient of log-likelihood.
        """
        
        len_y = len(self.ytrain)
        if self.flag_train_noise:
            self.kernel.hyperparams = theta[1:-1]
            K, grad_K = self.kernel.matrix()
            C = theta[0]**2 * K + (theta[-1]**2 + self.bias_ynoise**2)*np.eye(len_y)
        else:
            self.kernel.hyperparams = theta[1:]
            K, grad_K = self.kernel.matrix()
            C = theta[0]**2 * K + (self.ynoise**2 + self.bias_ynoise**2)*np.eye(len_y)
        L = np.linalg.cholesky(C)
        Ly = np.linalg.solve(L, self.ytrain - self.prior_mean(self.Xtrain))
        log_p = -0.5*(Ly @ Ly) - np.sum(np.log(np.diag(L))) - 0.5*len_y*np.log(2*np.pi)
        
        ### Taken from GPML code by Rasmussen & Williams & Nickisch ###
        # if self.flag_train_noise:
        #     K = self.kernel(self.Xtrain, self.Xtrain, theta[1:-1])[0]
        #     if theta[-1] < 1e-6:
        #         C = theta[0]**2 * K + theta[-1]**2 * np.eye(len_y)
        #         s1 = 1
        #     else:
        #         C = theta[0]**2 * K/theta[-1]**2 + np.eye(len_y)
        #         s1 = theta[-1]**2
        # else:
        #     K = self.kernel(self.Xtrain, self.Xtrain, theta[1:])[0]
        #     if theta[-1] < 1e-6:
        #         C = theta[0]**2 * K + self.ynoise**2 * np.eye(len_y)
        #         s1 = 1
        #     else:
        #         C = theta[0]**2 * K/self.ynoise**2 + np.eye(len_y)
        #         s1 = self.ynoise**2
        # L = np.linalg.cholesky(C)
        # Ly = np.linalg.solve(L, self.ytrain - self.prior_mean(self.Xtrain))
        # log_p = -0.5*(Ly @ Ly)/s1 - np.sum(np.log(np.diag(L))) - 0.5*len_y*np.log(2*np.pi*s1)
        
        LLy = np.linalg.solve(L.T, Ly)
        
        grad_log_p = []
        grad_log_p.append(theta[0]*(LLy.T @ (K @ LLy) - np.trace(np.linalg.solve(L.T, np.linalg.solve(L, K))))) # Derivative w.r.t. outputscale
        for i in range(len(grad_K)):
            grad_log_p.append(0.5*theta[0]**2*(LLy.T @ (grad_K[i] @ LLy) - np.trace(np.linalg.solve(L.T, np.linalg.solve(L, grad_K[i]))))) # Derivatives w.r.t. kernel hyperparameters
        if self.flag_train_noise:
            grad_log_p.append(theta[-1]*(LLy @ LLy - np.trace(np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len_y)))))) # Derivative w.r.t. ynoise
        grad_log_p = np.asarray(grad_log_p)
        
        return -log_p, -grad_log_p
    

    def predict(self, Xtest):
        """Make predictions for test locations Xtest using the trained GPR model.

        Args:
            Xtest (np.ndarray): 2D array of test locations with shape (n_samples, n_features).

        Returns:
            mean_post (np.ndarray): 1D array of posterior mean predictions for test locations with shape (n_samples,).
            var_post (np.ndarray): 1D array of posterior variance predictions for test locations with shape (n_samples,).
        """
        
        K = self.outputscale**2 * self.kernel.matrix(gradient=False) # K(X,X)
        K_s = self.outputscale**2 * self.kernel.matrix(Xtest=Xtest, gradient=False) # K(X,X*)
        K_ss_diag = self.outputscale**2 * self.kernel.matrix(Xtest=Xtest, diag=True) # diag(K(X*,X*))

        # Get cholesky decomposition (square root) of the covariance matrix / kernel
        L = np.linalg.cholesky(K + (self.ynoise**2 + self.bias_ynoise**2)*np.eye(len(self.ytrain))) # K = L L^T

        Ly = np.linalg.solve(L, self.ytrain - self.prior_mean(self.Xtrain))
        Lk = np.linalg.solve(L, K_s) # <=> L^(-1) @ K_s

        var_post = K_ss_diag - np.sum(Lk**2, axis=0)
        mean_post = Lk.T @ Ly + self.prior_mean(Xtest)
        return mean_post, var_post


#EOF