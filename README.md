Disclaimer: This package is still under development!

# LightGPR
LightGPR is a minimalist Python repository designed for effortless creation of Gaussian Process Regression (GPR) models in an object-oriented manner. With a focus on simplicity, this repository offers a streamlined approach to implement GPR, allowing users to quickly prototype and experiment with Gaussian processes without unnecessary complexity.

Key Features:

**Object-Oriented Design:** Structured around intuitive object-oriented principles, making it easy to understand and extend GPR models.

**Minimalist Implementation:** Aims to provide a lightweight and concise codebase for Gaussian Process Regression, ensuring simplicity without sacrificing functionality.

**Customizable:** Offers flexibility for customization, allowing users to tailor GPR models to specific use cases and datasets by simply adding custom kernel and mean functions.

Whether you're a seasoned practitioner or just getting started with Gaussian Process Regression, LightGPR offers a straightforward solution for implementing and experimenting with GPR models in Python.

# Usage
For a quickstart to LightGPR, check out the [Jupyter notebook `test.ipynb`](./test.ipynb).

For generic use, import the Python class with `from LightGPR.gp_reg import gp_reg`. Then create a GP model with `model = gp_reg(Xtrain, ytrain)`, where `Xtrain` and `ytrain` are your training data inputs and outputs, respectively. Call `model.train()` to optimize the hyperparameters. `mean_post, var_post = model.predict(Xtest)` then returns the mean and variance of the regression function at the test locations `Xtest`.

Additional custom kernel functions can be added in the file `kernels.py`; additional custom prior mean functions can be added in the file `prior_means.py`