from data_processing import *
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

def objective(params):
    kfold = KFold(n_splits=3, shuffle=True, random_state=0)
    svr = SVR(**params)
    scores = cross_val_score(svr, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)

    # Extract the best score
    best_score = np.mean(scores)

    # Loss must be minimized
    loss = - best_score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

def hyperparameter_search():
    # List of C values
    C_range = np.logspace(-10, 10, 21)
    # List of gamma values
    gamma_range = np.logspace(-10, 10, 21)
    space = {
    'C' : hp.choice('C', C_range),
    'gamma' : hp.choice('gamma', gamma_range.tolist()+['scale', 'auto']),
    'kernel' : hp.choice('kernel', ['rbf', 'poly'])
    }

    kfold = KFold(n_splits=3, shuffle=True, random_state=0)
    bayes_trials = Trials()

    # Optimize
    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = bayes_trials)
    print(space_eval(best))

X_train, X_test, y_train, y_test, X, y = split_data(load_data("B0005"), 0.2)

hyperparameter_search()



best_svr = SVR(C=20, epsilon=0.0001, gamma=0.00001, cache_size=200,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

best_svr.fit(X_train, y_train)

y_pred = best_svr.predict(X.values)

plt.scatter(y, y_pred)
plt.show()