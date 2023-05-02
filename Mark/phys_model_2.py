import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import pandas as pd
from sklearn.model_selection import train_test_split


k = 200
test_size = 0.5
# Bat = pd.read_excel(r"./data/processed/B0018_1.xlsx").to_numpy()
Bat_1 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:] #pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:]
Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()[:k,:]
Bat_3 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0007').to_numpy()[:k,:]
Bat_4 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0006').to_numpy()[:k,:]

Bat = Bat_1

x_columns = [0,1,2,3,4,7] # voltage, temp, capacity

x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=test_size, random_state=None, shuffle=False)


# Define the physical model of battery discharge
def battery_model(x, k, Q):
    return Q / (k * x + Q)

# Define the negative log marginal likelihood function for the GPR model
def nll_fn(theta, X, y):
    kernel = C(theta[0], (1e-3, 1e3)) * RBF(theta[1], (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=theta[2]**2, n_restarts_optimizer=10)
    gp.fit(X, y)
    return gp.log_marginal_likelihood()

# Define the objective function for the optimization of the negative log marginal likelihood
def objective_fn(theta, X, y):
    return nll_fn(theta, X, y)

# Generate some synthetic data

# Optimize the hyperparameters of the GPR model
res = minimize(objective_fn, np.array([1, 1, 1]), args=(x_train, y_train), bounds=((1e-3, 1e3), (1e-2, 1e2), (1e-3, 1e3)))
theta_opt = res.x

# Fit the GPR model to the data
kernel = C(theta_opt[0], (1e-3, 1e3)) * RBF(theta_opt[1], (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=theta_opt[2]**2)
gp.fit(x_train, y_train)

# Define a function to predict battery time to discharge using model fusion
def predict_battery_time(x):
    y_pred_gpr, y_std_gpr = gp.predict([[x]], return_std=True)
    y_pred_phys = battery_model(x, 1, 5)
    y_pred_fusion = (y_pred_gpr + y_pred_phys) / 2
    y_std_fusion = np.sqrt(y_std_gpr**2 + 0.01**2) # combine the GPR and physical model uncertainties
    return y_pred_fusion, y_std_fusion

# Plot the results
x = np.linspace(0, 10, 100)
y_pred_fusion, y_std_fusion = predict_battery_time(x)
y_pred_gpr, y_std_gpr = gp.predict(x.reshape(-1, 1), return_std=True)
y_pred_phys = battery_model(x, 1, 5)
plt.plot(x, y_pred_fusion, label='Model Fusion')
plt.fill_between(x, y_pred_fusion - 2*y_std_fusion, y_pred_fusion + 2*y_std_fusion, alpha=0.1)
plt.plot(x, y_pred_gpr, label='GPR')
