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
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

def train_test_validation_split(X, y, test_size, val_size):
    from sklearn.model_selection import train_test_split

    x_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0) #the input here has to be a numpy array

    ratio_remaining = 1 - test_size
    ratio_val_adjusted = val_size / ratio_remaining

    X_train, X_val, y_train, y_val = train_test_split(x_remaining, y_remaining, test_size=ratio_val_adjusted, shuffle=True, random_state=0) #the input here has to be a numpy array
    
    return [X_train, y_train, X_test, y_test, X_val, y_val]

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
    epoch = [0]
    # instantiate early stopper
    es = EarlyStopper(patience=es_patience, min_delta=es_delta)

    X_train = X_train.float()
    y_train = y_train.float()
    X_val = X_val.float()
    y_val = y_val.float()

    with torch.no_grad():
        train_loss_history = [loss_function(model(X_train), y_train).item()]
        val_loss_history = [loss_function(model(X_val), y_val).item()]

    for i in range(n_epochs):
        X_train = X_train.float()
        y_train = y_train.float()
        X_val = X_val.float()
        y_val = y_val.float()
        target_train = model(X_train)
        target_val = model(X_val)

        loss_train = loss_function(target_train, y_train)
        loss_val = loss_function(target_val, y_val)
        
        train_loss_history.append(loss_train.item())

        val_loss_history.append(loss_val.item())
        
        epoch.append(i+1)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if verbose:
            print("Epoch: %d, loss train: %1.5f, loss val: %1.5f" % (i, loss_train , loss_val))

        if es.early_stop(loss_val):
            break
        else:
            continue

    return train_loss_history, val_loss_history, epoch

def lr_random_search(model, X_train, y_train, X_val, y_val, reps:int=15):
    es_patience = 4
    es_delta = 1e-6
    
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# device = torch.device("cpu")
### constan model params
act = torch.nn.ReLU()
model = nn_model(6, 100, 1, act, 100) #20 #50
loss_fn = torch.nn.MSELoss()
n_epoch = 150

#model to device
model.to(device)

### DNN with backpropagration 
df = pd.read_csv("data/B0005_TTD.csv")
v = df["Voltage_measured"].values
c = df["Current_measured"].values
t = df["Temperature_measured"].values
v_charge = df["Voltage_charge"].values
c_charge = df["Current_charge"].values
capacity = df["Capacity"].values
X = np.hstack((v, c))
X = np.hstack((X, t))
X = np.hstack((X, v_charge))
X = np.hstack((X, c_charge))
X = np.hstack((X, capacity))
X = X.reshape(len(v), 6)
X = (X-X.mean())/X.std()

y = df["TTD"].values
y = np.array(y)

y = y.reshape(len(y), 1)
y = (y-y.mean())/y.std()

X_train, X_test, y_train, y_test = split_scale(X, y, test_size=.1, scale=False, verbose=True)
X_train, X_val, y_train, y_val = split_scale(X_train, y_train, test_size=.1, scale=False, verbose=True)
X_train, X_val, y_train, y_val = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val)

# data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
# X_test = torch.from_numpy(X_test).to(device)
# y_test = torch.from_numpy(y_test).to(device)





#lr_best = lr_random_search(model, X_train, y_train, X_val, y_val) #0.00012
opt = torch.optim.Adam(params=model.parameters(), lr=0.00012)

[train_loss_history, val_loss_history, epoch] = train_model(X_train, y_train, X_val, y_val, n_epoch, model, loss_fn, opt, 1e-6, 1e-4, verbose=True)


# print("DONE TRAINING NOW KFOLD")
# ### DNN with k fold cross validation 
# k = 10 
# kf = KFold(n_splits=k, random_state=None)
# loss_train_k_fold = []
# for train_index, val_index in kf.split(X):
#     X_train_k, y_train_k = torch.from_numpy(X[train_index]), torch.from_numpy(y[train_index])
#     X_val_k, y_val_k = torch.Tensor(X[val_index]), torch.Tensor(y[val_index])

#     # data to device
#     X_train_k = X_train_k.to(device)
#     y_train_k = y_train_k.to(device)
#     X_val_k = X_val_k.to(device)
#     y_val_k = y_val_k.to(device)
#     for module in model.modules():
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#     [train_loss_k, val_loss_k, epoch_k] = train_model(X_train_k, y_train_k, X_val_k, y_val_k, n_epoch, model, loss_fn, opt, 1e-6, 1e-4, verbose=True)
#     final_train_loss_k = train_loss_k[-1]
#     final_val_loss_k = val_loss_k[-1]
#     loss_train_k_fold.append(final_train_loss_k)

# avg_error = np.mean(np.array(loss_train_k_fold))
# print("DONE KFOLD") 
# print("Average error: ", avg_error)
print("EKF TIME YAAAy")

### ekf stuff
if __name__ == "__main__":
    # check if gpu
    
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
    
    #data to device
    # X_train_ekf, X_val_ekf, y_train_ekf, y_val_ekf = torch.from_numpy(X_train_ekf), torch.from_numpy(X_val_ekf), torch.from_numpy(y_train_ekf), torch.from_numpy(y_val_ekf)
    # X_train_ekf, X_val_ekf, y_train_ekf, y_val_ekf = X_train_ekf.to(device), X_val_ekf.to(device), y_train_ekf.to(device), y_val_ekf.to(device)
    
    np.random.seed(1234)
    
    # make sure runs have same initialized weights
    rng = np.random.RandomState(123)
    state = rng.__getstate__()

    # nn = NeuralNetwork(layers=[3, 10, 10, 1], activations=[ReLU(), ReLU(), Sigmoid()], loss=QuadraticLoss(), rng=rng)

    # reset state
    rng.__setstate__(state)

    # train with ekf
    nn = NeuralNetwork(layers=[3, 40, 1], activations=[ReLU(), Linear()], loss=Unity(), rng=rng)



    # p_val = stats.loguniform.rvs(1e-1, 1e8, size=5)
    # q_val = stats.loguniform.rvs(1e-1, 1e8, size=5)
    # r_val = stats.loguniform.rvs(1e-1, 1e8, size=5)
    # p_list = []
    # q_list = []
    # r_list = []
    # loss_p = []
    # for p in p_val:
    #     for q in q_val:
    #         for r in r_val:
    #             rng.__setstate__(state)
    #             nn_l = NeuralNetwork(layers=[3, 40, 1], activations=[ReLU(), Linear()], loss=Unity(), rng=rng)
    #             train_loss_p, val_loss_p = nn_l.train_ekf(X_train_ekf.T, y_train_ekf.reshape(1, -1), P=p, R=r, Q=q, epochs=10, val=(X_val_ekf.T, y_val_ekf.reshape(1, -1)), eta=1e-2)
    #             train_loss_vals = train_loss_p.values()
    #             loss_p.append(list(train_loss_vals)[-1])
    #             p_list.append(p)
    #             q_list.append(q)
    #             r_list.append(r)
    
    # loss_arr = np.array(loss_p)
    # idx = np.where(loss_arr == np.amin(loss_arr))
    # idxs = (idx[0])
    # p_best = p_list[int(idxs)]
    # q_best = q_list[int(idxs)]
    # r_best = r_list[int(idxs)]
  

    rng.__setstate__(state)
    train_loss, val_loss = nn.train_ekf(X_train_ekf.T, y_train_ekf.reshape(1, -1), P=5.3, R=1e-2, Q=214.9, epochs=70, val=(X_val_ekf.T, y_val_ekf.reshape(1, -1)), eta=1e-2)
    # print("p_best, q_best, r_best")
    # print(p_best, q_best, r_best)
    # 1046603, 3370, 0.47
    # 5.2926 214.91008 12968.44
    # 1046603.717654118 42000473.17406705 214.910083604359
    # 5.292672888299462 30.773599420974 214.910083604359
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].plot(train_loss.keys(), train_loss.values(), label="train ekf")
    axs[0].plot(val_loss.keys(), val_loss.values(), label="validation ekf")
    axs[1].plot(epoch, train_loss_history, label='train DNN')
    axs[1].plot(epoch, val_loss_history, label='val DNN')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    #plt.title("Loss on training and validation sets comparison")
    axs[0].legend()
    axs[1].legend()
    plt.show()

    
