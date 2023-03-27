from data_processing import *
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
from pandas import *
from torch.nn import MSELoss

X_train, X_test, y_train, y_test, X, y = split_data(load_data("B0005"), 0.2)

best_svr = SVR(C=10, epsilon=0.0001, gamma=0.00001, cache_size=200,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

best_svr.fit(X_train, y_train)

y_pred = best_svr.predict(X.values)

loss = MSELoss()

#print("MSE: ", loss(y, y_pred))

time = X["Time"]

plt.plot(time[:198], y[:198], color = 'red', label = 'Real')
plt.plot(time[:198], y_pred[:198], color = 'blue', label = 'Predicted')
plt.title('SVR')
plt.xlabel('Time')
plt.ylabel('Time to Discharge')
plt.legend()
plt.show()