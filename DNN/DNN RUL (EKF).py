import torch 
import numpy as np
import scipy as s
import sklearn as sk
import pandas as pd
import tensorflow as tf
import backprop_nn
import matplotlib.pyplot as plt

from scipy import stats
#splitting test, train, validation 

#from exe_2_utils import split_scale, count_parameters, predict, weights_init
"""
def splitting_data(X, y, test_size, cv_size):
    from sklearn.model_selection import train_test_split

    x_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=0)

    ratio_remaining = 1 - test_size
    ratio_val_adjusted = cv_size / ratio_remaining

    X_train, X_cv, y_train, y_cv = train_test_split(
    x_remaining, y_remaining, test_size=ratio_val_adjusted, shuffle=False, random_state=0)
    
    # to gpu



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
            

#training model with backpropagation
def train_model(X_train, y_train, X_val, y_val, n_epochs, model, loss_function, optimizer, es_patience, es_delta):
    train_loss_history = []
    val_loss_history = []

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
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        #if es.early_stop(loss_val):
            #break
        #else:
            #continue
        print("Epoch: %d, loss train: %1.5f, loss val: %1.5f" % (i, loss_train , loss_val))

    
    return train_loss_history, val_loss_history

def lr_random_search(model, X_train, y_train, X_val, y_val, reps:int=15):
    es_patience = 4
    es_delta = 1e-4
    
    learning_rate = stats.loguniform.rvs(1e-4, 1e-1, size=reps)
    
    loss = []
    value = []
    for i in learning_rate:
        torch.nn.init.xavier_uniform(model.weight)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=i)
        
        train_loss_history, val_loss_history = model(
        model, X_train, y_train, X_val, y_val, loss_function,
        optimizer, n_epochs=1_000, tol_train=1e-3, es_patience=es_patience, es_delta=es_delta, verbose=False
        )
        
        final_loss = train_loss_history[-4]
        loss.append(final_loss)
        value.append(i)
    
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
#print(y)

#device for gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print(f"Test split:")
X_train, X_test, y_train, y_test = split_scale(X, y, test_size=.1, scale=False, verbose=True)
    
print(f"\nValidation split:")
X_train, X_val, y_train, y_val = split_scale(X_train, y_train, test_size=.1, scale=False, verbose=True)
X_train, X_val, y_train, y_val = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
act = torch.nn.Sigmoid()
#[X_train, y_train, X_test, y_test, X_cv, y_cv] = splitting_data(X, y, 0.1, 0.1)

model = nn_model(3, 100, 1, act, 50)

loss_fn = torch.nn.MSELoss()
#lr_best = lr_random_search(model, X_train, y_train, X_val, y_val)
opt = torch.optim.Adam(params=model.parameters(), lr=1e-2)
model = model.to(device)
X_train = torch.tensor(X_train).to(device)
y_train = torch.tensor(y_train).to(device)
X_test = torch.tensor(X_test).to(device)
y_test = torch.tensor(y_test).to(device)
X_val = torch.tensor(X_val).to(device)
y_val = torch.tensor(y_val).to(device)
#[train_loss_history, val_loss_history] = train_model(X_train, y_train, X_val, y_val, 300, model, loss_fn, opt, 1e-6, 1e-4)

[train_loss_history_ekf, val_loss_history_ekf] = backprop_nn.NeuralNetwork.train_ekf(self=model, x=X_train, y=y_train)

print(train_loss_history_ekf)
#print(train_loss_history_ekf)
