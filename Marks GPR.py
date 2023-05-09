from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct, ExpSineSquared
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys

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
length_scale = 200
nu = 0
length_scale_bounds = 0


'''
k = 1000

# Bat = pd.read_excel(r"./data/processed/B0018_1.xlsx").to_numpy()
Bat = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()
Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()

Bat = Bat[:k,:]
x_columns = [0,1,2,3,4,7] # voltage, temp, capacity


x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=0.1, random_state=None, shuffle=False)
'''
k = 8000
#k /= 4
k = int(k)
# Bat = pd.read_excel(r"./data/processed/B0018_1.xlsx").to_numpy()
Bat_1 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:]
Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()[:k,:]
Bat_3 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0007').to_numpy()[:k,:]
Bat_4 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0006').to_numpy()[:k,:]

Bat = Bat_1
#Bat = np.block([[Bat_1], [Bat_2], [Bat_3]])

x_columns = [0,1,2,3,4,7] # voltage, temp, capacity

x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=0.4, random_state=None, shuffle=False)
#print(x_train.shape, y_train.shape)
#x_train, x_test, y_train, y_test = x_train[:k,:], x_test[:k,:], y_train[:k], y_test[:k]
'''
x_train = Bat[:n_test,x_columns]
y_train = Bat[:n_test,5]

x_test = Bat[n_test:,x_columns]
y_test = Bat[n_test:,5]
'''

kernel_1 = Matern() + ExpSineSquared() + WhiteKernel()  # Matern() + ConstantKernel() + ExpSineSquared()
gp_1 = GaussianProcessRegressor(kernel=kernel_1,n_restarts_optimizer=8)
gp_1.fit(x_train, y_train)
gp_1.kernel_

#print(gp.predict(np.array([4, 25, 1.804298052]).reshape(1,-1), return_std=False))
'''
diff_plot = []
for i in range(len(y_test)):
    y_pred = gp_1.predict(x_test[i,:].reshape(1,-1), return_std=False)
    diff_plot.append((y_pred - y_test[i])/y_pred *100)
'''
y_pred_1 = gp_1.predict(x_test, return_std=False)
differences_1 = (y_pred_1[y_pred_1!=0] - y_test[y_pred_1!=0])/y_pred_1[y_pred_1!=0]*100


#plt.plot(np.linspace(0,1,len(differences_1)), differences_1)
plt.plot(np.linspace(0,1,len(y_pred_1)), y_pred_1)
plt.plot(np.linspace(0,1,len(y_test)), y_test)

#plt.plot(np.linspace(0,1,len(diff_plot)), diff_plot)
plt.xlabel('test instance')
plt.ylabel('test-truth % error ')
#plt.ylim((-100,100))
plt.show()

#plt.plot(x, mean_prediction)
#plt.plot(vec, y)
#plt.show()