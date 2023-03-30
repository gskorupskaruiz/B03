import torch
from data_processing import *
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval
from backprop_nn import NeuralNetwork
from sklearn.model_selection import train_test_split
from DNN_RUL_EKF import split_scale
from activations import *
from losses import *
def objective(params):
    rng = np.random.RandomState(123)
    state = rng.__getstate__()
    df_ekf = pd.read_csv("data/B0005_TTD.csv")
    v_ekf = df_ekf["Voltage_measured"][:100].values
    c_ekf = df_ekf["Current_measured"][:100].values
    t_ekf = df_ekf["Temperature_measured"][:100].values
    X_ekf = np.hstack((v_ekf, c_ekf))
    X_ekf = np.hstack((X_ekf, t_ekf))
    X_ekf = X_ekf.reshape(len(v_ekf), 3)
    X_ekf = (X_ekf-X_ekf.mean())/X_ekf.std()
    #print(X.shape)
    y_ekf = df_ekf["TTD"][:100].values
    y_ekf = np.array(y_ekf)
    y_ekf = y_ekf.reshape(len(y_ekf), 1)
    y_ekf = (y_ekf-y_ekf.mean())/y_ekf.std()
    X_train_ekf, X_test_ekf, y_train_ekf, y_test_ekf = split_scale(X_ekf, y_ekf, test_size=.1, scale=False, verbose=True)
    X_train_ekf, X_val_ekf, y_train_ekf, y_val_ekf = split_scale(X_train_ekf, y_train_ekf, test_size=.1, scale=False, verbose=True)
    kfold = KFold(n_splits=3, shuffle=True, random_state=0)
    rng.__setstate__(state)
    nn_l = NeuralNetwork(layers=[3, 40, 1], activations=[ReLU(), Linear()], loss=Unity(), rng=rng)
    print(X_train_ekf.shape())
    train = nn_l.train_ekf(X_train_ekf.T, y_train_ekf.reshape(1, -1), **params, epochs=10, val=(X_val_ekf.T, y_val_ekf.reshape(1, -1)), eta=1e-2)
    print(nn_l.feedforward(X_train_ekf))
    scores = cross_val_score(nn_l.feedforward, X_val_ekf, y_val_ekf, cv=kfold, scoring='accuracy', n_jobs=-1)

    # Extract the best score
    best_score = np.mean(scores)

    # Loss must be minimized
    loss = - best_score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

def hyperparameter_search():
    P_range = stats.loguniform.rvs(1e-1, 1e8, size=5)
    Q_range = stats.loguniform.rvs(1e-1, 1e8, size=5)
    R_range = stats.loguniform.rvs(1e-1, 1e8, size=5)
    space = {
    'P' : hp.choice('P', P_range),
    'Q' : hp.choice('Q', Q_range),
    'R' : hp.choice('R', R_range)
    }

    kfold = KFold(n_splits=3, shuffle=True, random_state=0)
    bayes_trials = Trials()

    # Optimize
    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = bayes_trials)
    print(space_eval(best))

hyperparameter_search()














#X_train, y_train, X_test, y_test, X_val, y_val = train_test_validation_split(X, y, 0.1, 0.1)
X_train, X_val, y_train, y_val = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val)

def gbm_cl_bo(p, q, r):
    params_gbm = {}
    params_gbm['p'] = p
    params_gbm['q'] = q
    params_gbm['r'] = r
    scores = cross_val_score(GradientBoostingClassifier(random_state=123, **params_gbm),
                             X_train, y_train, scoring=acc_score, cv=5).mean()
    score = scores.mean()
    return score
# Run Bayesian Optimization
start = time.time()
params_gbm ={
    'p':stats.loguniform.rvs(1e-1, 1e8, size=5),
    'q':stats.loguniform.rvs(1e-1, 1e8, size=5),
    'r':stats.loguniform.rvs(1e-1, 1e8, size=5)
}
gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=111)
gbm_bo.maximize(init_points=20, n_iter=4)
print('It takes %s minutes' % ((time.time() - start)/60))
def optimize():
    p_val = stats.loguniform.rvs(1e-1, 1e8, size=5)
    q_val = stats.loguniform.rvs(1e-1, 1e8, size=5)
    r_val = stats.loguniform.rvs(1e-1, 1e8, size=5)
    p_list = []
    q_list = []
    r_list = []
    loss_p = []
    for p in p_val:
        for q in q_val:
            for r in r_val:
                rng.__setstate__(state)
                nn_l = NeuralNetwork(layers=[3, 40, 1], activations=[ReLU(), Linear()], loss=Unity(), rng=rng)
                train_loss_p, val_loss_p = nn_l.train_ekf(X_train_ekf.T, y_train_ekf.reshape(1, -1), P=p, R=r, Q=q, epochs=10, val=(X_val_ekf.T, y_val_ekf.reshape(1, -1)), eta=1e-2)
                train_loss_vals = train_loss_p.values()
                print(type(train_loss_vals))
                loss_p.append(list(train_loss_vals)[-1])
                p_list.append(p)
                q_list.append(q)
                r_list.append(r)
    
    loss_arr = np.array(loss_p)
    idx = np.where(loss_arr == np.amin(loss_arr))
    idxs = (idx[0])
    p_best = p_list[int(idxs)]
    q_best = q_list[int(idxs)]
    r_best = r_list[int(idxs)]