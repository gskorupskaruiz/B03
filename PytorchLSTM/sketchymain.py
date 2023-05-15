import pandas as pd
from model import *
import numpy as np
from data_processing import load_gpu_data, load_gpu_data_with_batches
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset
import math 
# import plot
import matplotlib.pyplot as plt
def load_data_normalise(battery):
    data = []
    # for all battery files combine them into one dataframe
    for i in battery:
        data.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
    data = pd.concat(data)
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
    return normalized_data

def testing_func(X_test, y_test):
    rmse_test, result_test = 0, list()

    # one interation 
    test_predict = model.forward(X_test)

    y_pred = model(X_test)
    rmse_test = np.sqrt(criterion(y_pred, y_test).item())
    # for ite in range(1, num + 1):
    #     X_test = group_for_test.get_group(ite).iloc[:, 2:]
    #     X_test_tensors = torch.Tensor(X_test.to_numpy())
    #     X_test_tensors = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    #     test_predict = model.forward(X_test_tensors)
    #     data_predict = max(test_predict[-1].detach().numpy(), 0)
    #     result_test.append(data_predict)
    #     rmse_test = np.add(np.power((data_predict - y_test.to_numpy()[ite - 1]), 2), rmse_test)

    # rmse_test = (np.sqrt(rmse_test / num)).item()
    return result_test, rmse_test

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

def train(model, X_train, y_train, X_val, y_val, n_epoch, lf, optimizer, verbose = True):
    epoch = []
    model.to(device) # set model to GPU
    #intiate early stopper
    early_stopper = EarlyStopper(patience=1e-16, min_delta=1e-6)
    # X_train = X_train.double()
    # y_train = y_train.double()
    # X_val = X_val.double()
    # y_val = y_val.double()

    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        target_train = model(X_train).unsqueeze(2) # i changed this 

        target_val = model(X_val).unsqueeze(2) # i changed this - added the unsqueeze thing 
        # print(f'size of target_train {target_train.shape} and size of y_train {y_train.shape}')
        # print(f"x_train {X_train.shape} and y_train {y_train.shape}")
        loss_train = lf(target_train, y_train)
        loss_val = lf(target_val, y_val)
        train_loss_history.append(loss_train.item())
        val_loss_history.append(loss_val.item())

        epoch.append(i+1)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print(loss_train)

        # if verbose:
        print(f"Epoch {i+1}: train loss = {loss_train.item():.4f}, val loss = {loss_val.item():.4f}")

        # if early_stopper.early_stop(loss_val.item()):
        #      print(f"Early stopping at epoch {i+1}")
        #      break
    return train_loss_history, val_loss_history, epoch

def trainbatch(model, train_dataloader, val_dataloader, n_epoch, lf, optimizer, verbose = True):
    epoch = []
  #  model.to(device) # set model to GPU
    #intiate early stopper
    early_stopper = EarlyStopper(patience=1e-16, min_delta=1e-6)

    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        loss_v = 0
        loss = 0
        for l, (x, y) in enumerate(train_dataloader):
            target_train = model(x) # .unsqueeze(2) #uncomment this for simple lstm
            loss_train = lf(target_train, y)
            loss += loss_train.item()
            #train_loss_history.append(loss_train.item())
            epoch.append(i+1)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        for k, (xv, yv) in enumerate(val_dataloader):
            
            target_val = model(xv) #.unsqueeze(2) uncomment this for simple lstm
            loss_val = lf(target_val, yv)
            loss_v += loss_val.item()

        train_loss = loss/len(train_dataloader)
        val_loss = loss_v/len(val_dataloader)
        
        
        if i == 0 and float(train_loss)>1:
            print('Loss is too high')
            break
        if i == 2 and float(train_loss)>0.5:
            print('Loss is too high')
            break
        if epoch == 4 and float(train_loss)>0.3:
            print('Loss is too high')
            break
        
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        #epoch.append(i+1)
        # if verbose:
        print(f"Epoch {i+1}: train loss = {train_loss:.10f}, val loss = {val_loss:.10f}")
    return train_loss_history, val_loss_history

def plot_loss(train_loss, val_loss, epoch):
    plt.plot(epoch, train_loss, label='train loss')
    plt.plot(epoch, val_loss, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def lr_opt(model, X_train, y_train, X_val, y_val, n_epochs,time=False):
        from scipy import stats
        
        learning_rate = stats.loguniform.rvs(10e-8, 1e0, size=15)
        
        loss = []
        value = []
        for i in learning_rate:

            for module in model():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)

            torch.nn.init.xavier_uniform(model.weight)
            print(f"current lrq 1qqqqqq is {i}")
            lf = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(params = model.parameters(), lr=i)

            train_loss_history, val_loss_history, epoch = train(model, X_train, y_train, X_val, y_val, n_epochs, lf, optimizer, verbose=True,)
            
            final_loss = train_loss_history[-1]
            loss.append(final_loss)
            value.append(i)
        
        #print(loss, type(loss))
        loss_arr = np.array(loss)
        idx = np.where(loss_arr == np.amin(loss_arr))
        idxs = (idx[0])
        lr_best = value[int(idxs)]
        
        return lr_best

class SeqDataset(Dataset):
    def __init__(self, x_data, y_data, seq_len, batch):
        self.x_data = x_data
        self.y_data = y_data
        self.seq_len = seq_len
        self.batch = batch

    def __len__(self):
        return math.ceil((len(self.x_data) / self.batch))

    def __getitem__(self, idx):
        start_idx = idx * self.batch
        end_idx = start_idx + self.batch

        x = self.x_data[start_idx:end_idx]
        y = self.y_data[start_idx:end_idx]

        if end_idx > len(self.x_data):
            x = self.x_data[start_idx:]
            y = self.y_data[start_idx:]
    
        if x.shape[0] == 0:
            raise StopIteration
        
        return x, y

def run_model(hyperparams):
    # import data
    battery = ['B0005', 'B0006', 'B0007', 'B0018'] # no clue why but battery 5 just doesnt work - even though it has the same format and i use the same code :(
    data = load_data_normalise(battery)
    input_size = data.shape[1] - 1 #len(data.columns) - 1
    print(f'size of input is {input_size}')
    print(hyperparams)
    n_hidden, n_layer, lr, seq, batch_size, num_layers_conv, output_channels_val, kernel_sizes_val, stride_sizes_val, padding_sizes_val, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense_val = hyperparams
    # n_hidden = 40 #input_size
    # n_layer = 2
    lr = lr/1000
    n_epoch = 4
    #lr = 0.005
    test_size = 0.1
    cv_size = 0.1
   # seq = 20
    #batch_size = 1000
    
    # gpu?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #data initialization
    X_train, y_train, X_test, y_test, X_val, y_val = load_gpu_data_with_batches(data, test_size=test_size, cv_size=cv_size, seq_length=seq)  
    
    X_train, y_train, X_test, y_test, X_val, y_val = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device), X_val.to(device), y_val.to(device)
    
    dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq, batch=batch_size)
    datasetv = SeqDataset(x_data=X_val, y_data=y_val, seq_len=seq, batch=batch_size)
    # LsTM Model initialization
    
    #num_layers_conv = 3
    
    #output_channels_val = 7
    output_channels = [output_channels_val] * num_layers_conv
    
    #kernel_sizes_val = 2
    kernel_sizes = [kernel_sizes_val] * num_layers_conv
    
    #stride_sizes_val = 1
    stride_sizes = [stride_sizes_val] * num_layers_conv
    
    #padding_sizes_val = 1
    padding_sizes = [padding_sizes_val] * num_layers_conv
    
    #hidden_size_lstm = 40
    
    #num_layers_lstm = 2
    
    #hidden_neurons_dense_val = 10
    hidden_neurons_dense = [seq, hidden_neurons_dense_val, 1]

    model = ParametricCNNLSTM(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq).double()
    model.to(device)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training and evaltuation
    train_hist, val_hist = trainbatch(model, dataset, datasetv, n_epoch, criterion, optimizer, verbose = True)
    #train_hist, val_hist, epoch = train(model, X_train, y_train, X_val, y_val, n_epoch, criterion, optimizer, verbose = True)

    predictions = model(X_test).to('cpu').detach().numpy()
    
    # WHYYYYYYY NO PREDICT GOWRIIIIII HELPPPPPPP
    
    #print(predictions)
    epoch = np.linspace(1, n_epoch+1, n_epoch)
    # plt.plot(predictions.squeeze(2), label='pred', linewidth=2, color='red')
    # plt.plot(y_test.squeeze(2).to('cpu').detach().numpy()) 
    # plt.legend()
    # plt.show()

    
    loss = ((predictions.squeeze(2) - y_test.squeeze(2).to('cpu').detach().numpy()) ** 2).mean()

    if loss != 'nan':
    #    print(f'no wayy sooo cooool the model predicts! :)')
        print(f'btw the current loss is {loss.round(5)}')
    
    if loss < 0.3:
        print('yes')
        
        with open('PytorchLSTM/final_runs.txt', 'a') as f:
            print('yess')
            f.write(str(hyperparams))
            f.write('\t')
            f.write(str(loss))
            f.write('\n')
            
    
    #plt.plot(epoch, train_hist)
    #plt.plot(epoch, val_hist)
    #plt.show()
    
   # print(model)

    return loss

