from data_processing import *
import matplotlib.pyplot as plt
from sklearn.svm import SVR

X_train, X_test, y_train, y_test = split_data(load_data("B0005"), 0.2)

X_train = X_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)

best_svr = SVR(C=20, epsilon=0.0001, gamma=0.00001, cache_size=200,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

best_svr.fit(X_train, y_train)

y_pred = best_svr.predict(X.values.reshape(-1, 1))

plt.plot(y_train, y_test)
plt.show()