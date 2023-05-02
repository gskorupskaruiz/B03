#from data_processing import *
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from torch.nn import MSELoss
import torch
from sklearn.model_selection import train_test_split

#import tensorflow as tf

#Loading the data
k = 5000
test_size = 0.5


def SVR_gen(k):

    # Bat = pd.read_excel(r"./data/processed/B0018_1.xlsx").to_numpy()
    Bat_1 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:] #pd.read_excel(r"./data/processed/B0005.xlsx", 'B0005').to_numpy()[:k,:]
    Bat_2 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0018').to_numpy()[:k,:]
    Bat_3 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0007').to_numpy()[:k,:]
    Bat_4 = pd.read_excel(r"./data/processed/B0005.xlsx", 'B0006').to_numpy()[:k,:]

    Bat = Bat_1

    x_columns = [0,1,2,3,4,7] # voltage, temp, capacity

    x_train, x_test, y_train, y_test = train_test_split(Bat[:,x_columns], Bat[:,8], test_size=test_size, random_state=None, shuffle=True)
    x,y = Bat[:,x_columns], Bat[:,8]

    #Setting up the SVR
    best_svr = SVR(C=100, epsilon=0.0001, gamma='scale', cache_size=100,
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

    #Training the SVR
    best_svr.fit(x_train, y_train)


    return best_svr

'''
#Predicting the results
y_pred = best_svr.predict(x_test)#.values)
#Computing the MSE from normalized data
loss = MSELoss()
print(y_pred)

y_tens = torch.tensor((y_test-y_test.mean())/y_test.std())
y_pred_tens = torch.tensor((y_pred-y_pred.mean())/y_pred.std())

mse = loss(y_pred_tens, y_tens)
#mse = mse.detach().numpy() * 100
#print("MSE: ", round(mse, 3), "%")
'''
#time = x["Time"][:len(y_pred)]

#Plotting the results
#time = X["Time"][:len(y_pred)]

# plt.scatter(time[:197], y_test[:197], color = 'red', label = 'Real')
# plt.scatter(time[:197], y_pred[:197], color = 'blue', label = 'Predicted')
# plt.title('SVR')
# plt.xlabel('Time')
# plt.ylabel('Time to Discharge')
# plt.legend()
# plt.show()