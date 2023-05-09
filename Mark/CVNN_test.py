from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct, ExpSineSquared
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys


'''KERNEL FUNCTIONS'''
length_scale = 200
nu = 0
length_scale_bounds = 0


k = 3500

# Bat = pd.read_excel(r"./data/processed/B0018_1.xlsx").to_numpy()
Bat_1 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:]
Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()[:k,:]
Bat_3 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0007').to_numpy()[:k,:]
Bat_4 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0006').to_numpy()[:k,:]

#Bat = np.block([[Bat_1], [Bat_2], [Bat_3], [Bat_4]])
Bat = Bat_1
x_columns = [0,1,2,3,4,7] # voltage, temp, capacity

x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=0.05, random_state=None, shuffle=False)

kernel_1 = ExpSineSquared() #+ ConstantKernel()

'''KERNEL PARAMETERS'''

# kernel_1 = Matern() + ExpSineSquared() + WhiteKernel()  # Matern() + ConstantKernel() + ExpSineSquared()
gp_1 = GaussianProcessRegressor(kernel=kernel_1,n_restarts_optimizer=8)
gp_1.fit(x_train, y_train)
gp_1.kernel_

'''
diff_plot = []
for i in range(len(y_test)):
    y_pred = gp_1.predict(x_test[i,:].reshape(1,-1), return_std=False)
    diff_plot.append((y_pred - y_test[i])/y_pred *100)
'''

y_pred_1 = gp_1.predict(x_test, return_std=False)
# differences_1 = (y_pred_1[y_pred_1!=0] - y_test[y_pred_1!=0])/y_pred_1[y_pred_1!=0]*100
differences_1 = (y_pred_1 - y_test)/y_pred_1*100

plt.plot(np.linspace(0,1,len(differences_1)), differences_1)


#plt.plot(np.linspace(0,1,len(y_pred_1)), y_pred_1)
#plt.plot(np.linspace(0,1,len(y_test)), y_test)

#plt.plot(np.linspace(0,1,len(diff_plot)), diff_plot)
plt.xlabel('test instance')
plt.ylabel('test-truth % error ')
plt.show()