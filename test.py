"""Visualization of Gaussian process regression on 1D and 2D data."""

import numpy as np
import matplotlib.pyplot as plt
from gp_reg import gp_reg


### 1D Gaussian process regression ###
# Generate some 1D data
Xtrain_1D = np.random.uniform(0, 10, size=8)
ytrain_1D = np.sin(Xtrain_1D)
Xtest_1D = np.linspace(0, 10, 1000)

# Create Gaussian process regression model
model_1D = gp_reg(Xtrain_1D, ytrain_1D)#, kernel='RQ')
model_1D.train() # Learn hyperparameters
print(f"Loss: {model_1D.loss}, Outputscale: {model_1D.outputscale}, Hyperparams: {model_1D.kernel_hyperparams}, Noise: {model_1D.ynoise}")
mean_post_1D, var_post_1D = model_1D.predict(Xtest_1D)

# Plot results
plt.plot(Xtrain_1D, ytrain_1D, 'ro', label='Training data')
plt.plot(Xtest_1D, mean_post_1D, 'b-', label='Regression')
plt.fill_between(Xtest_1D, mean_post_1D - 2*np.sqrt(var_post_1D), mean_post_1D + 2*np.sqrt(var_post_1D), color='b', alpha=0.2, label='95% confidence interval')
plt.xlabel(r'$x$')
plt.ylabel(r'$f \, (x)$')
plt.legend()
plt.show()


### 2D Gaussian process regression ###
# Generate some 2D data
Xtrain = np.random.rand(10,2)
ytrain = np.sin(Xtrain[:,0]) + np.cos(Xtrain[:,1])
test_grid = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
Xtest = np.array(list(zip(test_grid[0].ravel(), test_grid[1].ravel())))

# Create Gaussian process regression model
model = gp_reg(Xtrain, ytrain)#, kernel='RQ')
model.train() # Learn hyperparameters
print(f"Loss: {model.loss}, Outputscale: {model.outputscale}, Hyperparams: {model.kernel_hyperparams}, Noise: {model.ynoise}")
mean_post, var_post = model.predict(Xtest)

# Plot results
fig, ax = plt.subplots(1, 3)
ax[0].imshow(np.sin(test_grid[0]) + np.cos(test_grid[1]), cmap='jet', extent=[0,1,0,1], origin='lower')
ax[0].set_title('True function')
ax[0].set_ylabel(r'$x_2$')
ax[0].set_xlabel(r'$x_1$')

im2 = ax[1].imshow(mean_post.reshape(test_grid[0].shape), cmap='jet', extent=[0,1,0,1], origin='lower')
ax[1].scatter(Xtrain[:,0], Xtrain[:,1], c='k', marker='x', label='Training data')
ax[1].set_title('Regression')
ax[1].set_ylabel(r'$x_2$')
ax[1].set_xlabel(r'$x_1$')
ax[1].legend()

im3 = ax[2].imshow(np.sqrt(var_post).reshape(test_grid[0].shape), cmap='jet', extent=[0,1,0,1], origin='lower')
ax[2].scatter(Xtrain[:,0], Xtrain[:,1], c='w', marker='x', label='Training data')
fig.colorbar(im3, ax=ax[2], label=r'$\sigma$', shrink=0.5, pad=0.1)
ax[2].set_title('Standard deviation')
ax[2].set_ylabel(r'$x_2$')
ax[2].set_xlabel(r'$x_1$')
fig.tight_layout()
plt.show()


#EOF
