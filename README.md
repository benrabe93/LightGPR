**Disclaimer:** This package is fully functional. However, it is still under further development.

# :zap: LightGPR
LightGPR is a *minimalist* Python repository designed for effortless creation of Gaussian Process Regression (GPR) models in an object-oriented manner. With a focus on simplicity, this repository offers a streamlined approach to implement GPR, allowing users to quickly prototype and experiment with Gaussian processes without unnecessary complexity.

🎯 **Special use cases** include custom kernel functions, such as non-stationary kernels, for enhanced predictive modeling.

Key Features:

**Customizable:** Offers flexibility for customization, allowing users to tailor GPR models to specific use cases and datasets by simply adding custom kernel functions.

**Minimalist Implementation:** Aims to provide a lightweight and concise codebase for Gaussian Process Regression, ensuring simplicity without sacrificing functionality.

**Object-Oriented Design:** Structured around intuitive object-oriented principles, making it easy to understand and extend GPR models.

Whether you're a seasoned practitioner or just getting started with Gaussian Process Regression, LightGPR offers a straightforward solution for implementing and experimenting with GPR models in Python.

# 📖 Usage
To **download**, either clone the repository
```
git clone https://github.com/benrabe93/LightGPR.git
```
or download the inner folder `LightGPR` into your local python project.

For a **quickstart guide** to LightGPR, check out the Jupyter notebook [`test.ipynb`](./test.ipynb).

For **generic use**, import the `gp_reg` class with
```
from LightGPR.gp_reg import gp_reg
```
Then, create a GP model with
```
model = gp_reg(Xtrain, ytrain)
```
where `Xtrain` and `ytrain` are your training data inputs and outputs, respectively. Call
```
model.train()
```
to optimize the hyperparameters. To make predictions, call
```
mean_post, var_post = model.predict(Xtest)
```
which returns the mean and variance of the regression function at the test locations `Xtest`.

**Additional custom kernel** functions can be added in the file [`kernels.py`](./LightGPR/kernels.py); custom prior mean functions are directly given when initializing the model using the `prior_mean` argument.
