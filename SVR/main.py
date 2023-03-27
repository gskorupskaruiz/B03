from data_processing import *
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
from pandas import *
from torch.nn import MSELoss
import torch
import tensorflow as tf

#Loading the data
data = load_data("B0006")
data = data.iloc[:10000]

#Splitting the data into training and testing data (not normalized)
#best split is 0.2; MSE = 5.79% for B0005
X_train, X_test, y_train, y_test, X, y = split_data(data, 0.2)
y_test = y_test.to_numpy()
#Setting up the SVR

best_svr = SVR(C=100, epsilon=0.0001, gamma='scale', cache_size=100,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#Training the SVR
best_svr.fit(X_train, y_train)

#Predicting the results
y_pred = best_svr.predict(X_test.values)

#Computing the MSE from normalized data
loss = MSELoss()


y_tens = torch.tensor((y_test-y_test.mean())/y_test.std())
y_pred_tens = torch.tensor((y_pred-y_pred.mean())/y_pred.std())

mse = loss(y_pred_tens, y_tens)
mse = mse.detach().numpy() * 100
print("MSE: ", round(mse, 3), "%")

#Plotting the results
#time = X["Time"][:len(y_pred)]

# plt.scatter(time[:197], y_test[:197], color = 'red', label = 'Real')
# plt.scatter(time[:197], y_pred[:197], color = 'blue', label = 'Predicted')
# plt.title('SVR')
# plt.xlabel('Time')
# plt.ylabel('Time to Discharge')
# plt.legend()
# plt.show()