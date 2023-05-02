from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct, ExpSineSquared, RationalQuadratic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import SVR_tests as svt
from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle
import sys
from sklearn.metrics import mean_squared_error
'''
x = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(x * np.sin(x))
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
x_train, y_train = x[training_indices], y[training_indices]
#v_input = Bat['time to discharge ']
#y = Bat['time to discharge ']

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
gp.fit(x_train, y_train)
gp.kernel_

mean_prediction, std_prediction = gp.predict(x, return_std=True)
'''

'''KERNEL FUNCTIONS'''
# Matern:
matern_lenght_scale = 4.35
matern_nu = 1.5


# Constant:
constant_value = 100000.00000000001

# ExpSinSquared:
expsin_length_scale = 0.001441839076703889
expsin_periodicity = 29.339



'''
k = 1000

# Bat = pd.read_excel(r"./data/processed/B0018_1.xlsx").to_numpy()
Bat = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()
Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()

Bat = Bat[:k,:]
x_columns = [0,1,2,3,4,7] # voltage, temp, capacity


x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=0.1, random_state=None, shuffle=False)
'''

k = 600
test_size = 0.5
# Bat = pd.read_excel(r"./data/processed/B0018_1.xlsx").to_numpy()
Bat_1 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:] #pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:]
Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()[:k,:]
Bat_3 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0007').to_numpy()[:k,:]
Bat_4 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0006').to_numpy()[:k,:]

Bat = Bat_1

x_columns = [0,1,2,3,4,7] # voltage, temp, capacity

x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=test_size, random_state=None, shuffle=False)
kernel_1 = Matern(length_scale= matern_lenght_scale, nu = matern_nu)

trend_kernel = Matern()
variational_kernel = Matern()* ExpSineSquared()
irregular_kernel = RationalQuadratic()
noise_kernel = RBF() #+ WhiteKernel(    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5))
kernel = Matern()  #variational_kernel + irregular_kernel + noise_kernel

y_train_mean = y_train.mean()
gp_1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
gp_1.fit(x_train, y_train)
gp_1.kernel_


#svr = svt.SVR_gen(100000)

'''DYNAMIC MODEL'''
pred =[]
K = 100
print(y_test.shape)
for data_counter in range(x_test.shape[0]):
    y_pred_1, std = gp_1.predict(x_test[data_counter,:].reshape(1,-1), return_std=True)
    #y_pred_2 = svr.predict(x_test[data_counter,:].reshape(1,-1))
    pred.append(1*y_pred_1[0])#+0.01*y_pred_2)

    kernel.set_params(**(gp_1.kernel_.get_params()))
    x_train = np.vstack([x_train, x_test[data_counter,:]])
    y_train = np.append(y_train, y_test[data_counter])# np.vstack([y_train, y_test[data_counter]])

    gp_1.fit(x_train, y_train)
    print(data_counter)
print(len(pred), y_test.shape)
plt.subplot(2,1,1)
plt.plot(np.linspace(0,1,len(pred)), pred)
plt.plot(np.linspace(0,1,len(y_test)), y_test)
plt.subplot(2,1,2)
plt.plot(np.linspace(0,1,len(pred)), (y_test-np.array(pred))/y_test*100)
plt.show()
print(mean_squared_error(y_test, pred))







