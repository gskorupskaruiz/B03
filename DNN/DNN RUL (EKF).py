import torch 
import numpy as np
import scipy as s
import sklearn as sk
import pandas as pd
import tensorflow as tf
import backprop_nn 
from losses import *
from activations import *
from backprop_nn import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from scipy import stats

"""
def splitting_data(X, y, test_size, cv_size):
    from sklearn.model_selection import train_test_split

    x_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=0)

    ratio_remaining = 1 - test_size
    ratio_val_adjusted = cv_size / ratio_remaining

    X_train, X_cv, y_train, y_cv = train_test_split(
    x_remaining, y_remaining, test_size=ratio_val_adjusted, shuffle=False, random_state=0)
    
    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn


def split_scale(X, y, test_size=.1, scale=True, verbose=False):
    original_data_type = type(X)

    # convert to numpy array, ravel labels 
    if original_data_type == torch.Tensor: 
        X = X.detach().numpy()
        y = y.detach().numpy()

    # split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    if verbose:
        # report
        print("Training set size: ", y_train.size)
        print("Test set size: ", y_test.size)

    # scale the data linearly between 0 and 1. Fit only on train dataset
    if scale:
        X_scaler = MinMaxScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        # scale the labels by the max of the training set
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

        if original_data_type == torch.Tensor:
            X_train = torch.tensor(X_train)
            y_train = torch.tensor(y_train)
            X_test = torch.tensor(X_test)
            y_test = torch.tensor(y_test)

        # return scaled and split data and the scaler 
        return (X_train, X_test, y_train, y_test, X_scaler, y_scaler)

    if original_data_type == torch.Tensor:
            X_train = torch.tensor(X_train)
            y_train = torch.tensor(y_train)
            X_test = torch.tensor(X_test)
            y_test = torch.tensor(y_test)

    return (X_train, X_test, y_train, y_test)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

def predict(model, X):
    with torch.no_grad():
        preds = model(X)
    return preds

def nn_model(dim_input, dim_hidden, dim_output, act, n_hidden):
    layers = []
    layers.append(torch.nn.Linear(dim_input, dim_hidden))
    layers.append(act)
    for i in range(n_hidden):
        layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
        layers.append(act)

    layers.append(torch.nn.Linear(dim_hidden, dim_output))
    model = torch.nn.Sequential(*layers)
    
    return model
 
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Implement the early stopping criterion.
        The function has to return 'True' if the current validation loss (in the arguments) has increased
        with respect to the minimum value of more than 'min_delta' and for more than 'patience' steps.
        Otherwise the function returns 'False'."""
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            return False
        else:
            if validation_loss - self.min_validation_loss > self.min_delta: #add code here
                self. counter += 1
                if self.counter > self.patience:
                    return True 
                else:
                    return False

            else:
                return False            

def train_model(X_train, y_train, X_val, y_val, n_epochs, model, loss_function, optimizer, es_patience, es_delta, verbose=True):
    train_loss_history = []
    val_loss_history = []
    epoch = []
    # instantiate early stopper
    es = EarlyStopper(patience=es_patience, min_delta=es_delta)

    for i in range(n_epochs):
        X_train = X_train.float()
        y_train = y_train.float()
        X_val = X_val.float()
        y_val = y_val.float()
        target_train = model(X_train)
        target_val = model(X_val)
        loss_train = loss_function(target_train, y_train)

        loss_val = loss_function(target_val, y_val)

        train_loss_history.append(loss_train)

        val_loss_history.append(loss_val)
        
        epoch.append(i)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if verbose:
            print("Epoch: %d, loss train: %1.5f, loss val: %1.5f" % (i, loss_train , loss_val))

        #if es.early_stop(loss_val):
            #break
        #else:
            #continue

    return train_loss_history, val_loss_history, epoch

def lr_random_search(model, X_train, y_train, X_val, y_val, reps:int=15):
    es_patience = 4
    es_delta = 1e-4
    
    learning_rate = stats.loguniform.rvs(1e-6, 1e1, size=reps)
    
    loss = []
    value = []
    for i in learning_rate:

        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        #torch.nn.init.xavier_uniform(model.weight)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=i)
        
        train_loss_history, val_loss_history, epoch = train_model(X_train, y_train, X_val, y_val, 10, model, loss_function, optimizer, es_patience, es_delta, verbose=False)
        final_loss = train_loss_history[-1]
        loss.append(final_loss.item())
        value.append(i)
    
    #print(loss, type(loss))
    loss_arr = np.array(loss)
    idx = np.where(loss_arr == np.amin(loss_arr))
    idxs = (idx[0])
    lr_best = value[int(idxs)]
    
    return lr_best

df = pd.read_csv("data/B0005_TTD.csv")
v = df["Voltage_measured"].values
c = df["Current_measured"].values
t = df["Temperature_measured"].values
X = np.hstack((v, c))
X = np.hstack((X, t))
X = X.reshape(len(v), 3)
X = (X-X.mean())/X.std()
#print(X.shape)
y = df["TTD"].values
y = np.array(y)

y = y.reshape(len(y), 1)
y = (y-y.mean())/y.std()


X_trainEKF, X_testEKF, y_trainEKF, y_testEKF = split_scale(X, y, test_size=.1, scale=False, verbose=True)
X_trainEKF, X_valEKF, y_trainEKF, y_valEKF = split_scale(X_trainEKF, y_trainEKF, test_size=.1, scale=False, verbose=True)
X_train, X_val, y_train, y_val = torch.from_numpy(X_trainEKF), torch.from_numpy(X_valEKF), torch.from_numpy(y_trainEKF), torch.from_numpy(y_valEKF)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
act = torch.nn.Sigmoid()

model = nn_model(3, 20, 1, act, 10) #20 #50

loss_fn = torch.nn.MSELoss()

lr_best = lr_random_search(model, X_train, y_train, X_val, y_val)

opt = torch.optim.Adam(params=model.parameters(), lr=lr_best)

#[train_loss_history, val_loss_history, epoch] = train_model(X_train, y_train, X_val, y_val, 100, model, loss_fn, opt, 1e-6, 1e-4, verbose=True)

### ekf stuff

if __name__ == "__main__":

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
    #X_train_ekf, X_val_ekf, y_train_ekf, y_val_ekf = torch.from_numpy(X_train_ekf), torch.from_numpy(X_val_ekf), torch.from_numpy(y_train_ekf), torch.from_numpy(y_val_ekf)

    np.random.seed(1234)
    
    # make sure runs have same initialized weights
    rng = np.random.RandomState(123)
    state = rng.__getstate__()

    nn = NeuralNetwork(layers=[3, 10, 10, 1], activations=[ReLU(), ReLU(), Sigmoid()], loss=QuadraticLoss(), rng=rng)

    # reset state
    rng.__setstate__(state)

    # train with ekf
    nn = NeuralNetwork(layers=[3, 40, 1], activations=[ReLU(), Linear()], loss=Unity(), rng=rng)
    train_loss, val_loss = nn.train_ekf(X_train_ekf.T, y_train_ekf.reshape(1, -1), P=1, R=1e4, Q=1e8, epochs=3, val=(X_val_ekf.T, y_val_ekf.reshape(1, -1)), eta=1e1)

    plt.plot(train_loss.keys(), train_loss.values(), label="train")
    plt.plot(val_loss.keys(), val_loss.values(), label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss for EKF-Algorithm")
    plt.legend()
    plt.show()
