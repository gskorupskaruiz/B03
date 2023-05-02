import numpy as np 
import torch 
from scipy import stats
from sklearn.model_selection import KFold

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

def train(model, X_train, y_train, X_val, y_val, n_epochs, lf, optimizer, es_patience, es_delta, verbose=True,):

        epoch = [0]
        
        # instantiate early stopper
        es = EarlyStopper(patience=es_patience, min_delta=es_delta)

        with torch.no_grad():
            train_loss_history = [lf(model(X_train), y_train).item()]
            val_loss_history = [lf(model(X_val), y_val).item()]

        for i in range(n_epochs):
            target_train = model(X_train)
            target_val = model(X_val)

            loss_train = lf(target_train, y_train)
            loss_val = lf(target_val, y_val)
            
            train_loss_history.append(loss_train.item())

            val_loss_history.append(loss_val.item())
            
            epoch.append(i+1)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if verbose:
                print("Epoch: %d, loss train: %1.5f, loss val: %1.5f" % (i, loss_train , loss_val))

            # if es.early_stop(loss_val):
            #     break
            # else:
            #     continue

        return train_loss_history, val_loss_history, epoch

def lr_opt(model, X_train, y_train, X_val, y_val, n_epochs, minv, maxv, time=False):

        es_patience = 4
        es_delta = 1e-4
        
        learning_rate = stats.loguniform.rvs(minv, maxv, size=15)
        
        loss = []
        value = []
        for i in learning_rate:

            for module in model():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)

            #torch.nn.init.xavier_uniform(model.weight)
            lf = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=i)

            train_loss_history, val_loss_history, epoch = train(model, X_train, y_train, X_val, y_val, n_epochs, lf, optimizer, es_patience, es_delta, verbose=True,)
            
            final_loss = train_loss_history[-1]
            loss.append(final_loss.item())
            value.append(i)
        
        #print(loss, type(loss))
        loss_arr = np.array(loss)
        idx = np.where(loss_arr == np.amin(loss_arr))
        idxs = (idx[0])
        lr_best = value[int(idxs)]
        
        return lr_best

def kfold_validation(model, X, y, k, n_epochs, lf, optimizer, es_patience, es_delta, time=False,):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        k = 10 
        kf = KFold(n_splits=k, random_state=None)
        loss_train_k_fold = []
        for train_index, val_index in kf.split(X):

            for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight)

            X_train_k, y_train_k = torch.from_numpy(X[train_index]), torch.from_numpy(y[train_index])
            X_val_k, y_val_k = torch.Tensor(X[val_index]), torch.Tensor(y[val_index])

            model.to(device)
            X_train_k, y_train_k = X_train_k.to(device), y_train_k.to(device)
            X_val_k, y_val_k = X_val_k.to(device), y_val_k.to(device)
            
            [train_loss_k, val_loss_k, epoch_k] = train(model, X_train_k, y_train_k, X_val_k, y_val_k, n_epochs, lf, optimizer, es_patience, es_delta, verbose=False)
            
            final_train_loss_k = train_loss_k[-1]
            final_val_loss_k = val_loss_k[-1]
            loss_train_k_fold.append(final_train_loss_k)

        
        avg_error = np.mean(np.array(loss_train_k_fold))
        return avg_error

class NN(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, n_hidden): #use this code first!!!
        super(NN, self).__init__() #also alwasy include this!!!

        self.layers = []
        self.layers.append(torch.nn.Linear(dim_input, dim_hidden))
        self.layers.append(torch.nn.ReLU())
        for i in range(n_hidden):
            self.layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(dim_hidden, dim_output))
        # self.layers.append(self.layer1)
        # self.layers.append(self.layer2)
        # self.layers.append(self.layer3)
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], torch.nn.Linear()):
                out = self.layers[i](x)
                out = torch.cos(out)
        out = self.layers[-1](out)
        return out 
    
    def train(self, X_train, y_train, X_val, y_val, n_epochs, lf, optimizer, es_patience, es_delta, verbose=True,):
        epoch = [0]
        # instantiate early stopper
        es = EarlyStopper(patience=es_patience, min_delta=es_delta)

        with torch.no_grad():
            train_loss_history = [lf(self.model.forward(X_train), y_train).item()]
            val_loss_history = [lf(self.model.forward(X_val), y_val).item()]

        for i in range(n_epochs):

            target_train = self.model.forward(X_train)
            #print(target_train)
            target_val = self.model.forward(X_val)

            loss_train = lf(target_train, y_train)
            loss_val = lf(target_val, y_val)
            
            train_loss_history.append(loss_train.item())

            val_loss_history.append(loss_val.item())
            
            epoch.append(i+1)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if verbose:
                print("Epoch: %d, loss train: %1.5f, loss val: %1.5f" % (i, loss_train , loss_val))

            # if es.early_stop(loss_val):
            #     break
            # else:
            #     continue

        return train_loss_history, val_loss_history, epoch
