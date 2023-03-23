from data_processing import *
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
from pandas import *

X_train, X_test, y_train, y_test, X, y = split_data(load_data("B0005"), 0.2)

best_svr = SVR(C=20, epsilon=0.0001, gamma=0.00001, cache_size=200,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

best_svr.fit(X_train, y_train)

y_pred = best_svr.predict(X.values)



plt.scatter(y, y_pred)
plt.show()