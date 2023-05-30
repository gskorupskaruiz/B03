import pandas as pd
from model import *
import numpy as np
from data_processing import *
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset
import math 
from bitstring import BitArray  
# import plot
import matplotlib.pyplot as plt
def load_data_normalise_hybrid(battery):
    data = []
    # for all battery files combine them into one dataframe
    for i in battery:
        data.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
    data = pd.concat(data)
    # print(data)
    time = data["Time"]
    time_mean = data["Time"].mean(axis=0)
    time_std = data["Time"].std(axis=0)
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
    return normalized_data, time_mean, time_std

def load_data_normalise(battery):
    data = []
    # for all battery files combine them into one dataframe
    for i in battery:
        data.append(pd.read_csv("data/" + i + "_TTD.csv"))
    data = pd.concat(data)
    # print(data)
    time = data["Time"]
    time_mean = data["Time"].mean(axis=0)
    time_std = data["Time"].std(axis=0)
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
    return normalized_data, time_mean, time_std

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

def trainbatch(model, train_dataloader, val_dataloader, n_epoch, lf, optimizer, verbose = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epoch = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # set model to GPU
    #intiate early stopper
    early_stopper = EarlyStopper(patience=1e-16, min_delta=1e-6)
    model.train()
    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        loss_v = 0
        loss = 0
        for l, (x, y) in enumerate(train_dataloader):
            #print(f'shape of y and x are {y.shape}, {x.shape}')
            target_train = model(x) #.unsqueeze(2) uncomment this for simple lstm
            #print(target_train.shape, y.shape, x.shape)
            loss_train = lf(target_train, y)
            loss += loss_train.item()
            #train_loss_history.append(loss_train.item())
            epoch.append(i+1)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        for k, (xv, yv) in enumerate(val_dataloader):
            
            target_val = model(xv) #.unsqueeze(2) uncomment this for simple lstm 
            #print(target_val.shape, yv.shape, xv.shape)
            loss_val = lf(target_val, yv)
            loss_v += loss_val.item()

        train_loss = loss/len(train_dataloader)
        val_loss = loss_v/len(val_dataloader)
        
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

if __name__ == '__main__': 
	# import data
    battery = ['B0006', 'B0007', 'B0018']
    # battery = ['B0018']
    data, time_mean, time_std = load_data_normalise_hybrid(battery)
    input_size = data.shape[1] - 1
    n_hidden = 20 #input_size
    n_layer = 2
    n_epoch = 25
    lr = 60/1000
    test_size = 0.1
    cv_size = 0.1
    seq = 28
    batch_size = 393
    
    # gpu?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #data initialization
    X_train, y_train, X_test, y_test, X_val, y_val = load_gpu_data_with_batches_hybrid(data, test_size=test_size, cv_size=cv_size, seq_length=seq)  
    X_train, y_train, X_test, y_test, X_val, y_val = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device), X_val.to(device), y_val.to(device)
    # print(f'shape of xtrain is {X_train.shape}')
    dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq, batch=batch_size)
    datasetv = SeqDataset(x_data=X_val, y_data=y_val, seq_len=seq, batch=batch_size)

    # LsTM Model initialization
    # num_layers_conv = 3
    # output_channels = [32, 10, 1]
    # kernel_sizes = [2, 1, 2]
    # stride_sizes = [1, 1, 1]
    # padding_sizes = [0, 2, 2]
    # hidden_size_lstm = 40
    # num_layers_lstm = 2
    # hidden_neurons_dense = [30, 10, 1]

    input_lstm = X_train.shape[2]
    num_layers_conv = 2
    output_channels = [5, 5]
    kernel_sizes = [6, 6]
    stride_sizes = [5, 5]
    padding_sizes = [3,3]
    hidden_size_lstm = 10
    num_layers_lstm = 1
    hidden_neurons_dense = [28, 41,  1]
    ga = True
    if ga:
        print('running ga individual')
        gene_length = 3
        ga_individual_solution =  [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1] 


        lstm_layers_bit = BitArray(ga_individual_solution[0:gene_length]) # don't understand the bitarray stuff yet or the length given per hyperparameter
        lstm_neurons_bit = BitArray(ga_individual_solution[gene_length:2*gene_length])
        lstm_sequential_length_bit = BitArray(ga_individual_solution[2*gene_length:3*gene_length])
        learning_rate_bit = BitArray(ga_individual_solution[3*gene_length:4*gene_length])
        cnn_layers_bit = BitArray(ga_individual_solution[4*gene_length:5*gene_length])
        cnn_kernel_size_bit = BitArray(ga_individual_solution[5*gene_length:6*gene_length])
        cnn_stride_bit = BitArray(ga_individual_solution[6*gene_length:7*gene_length])
        cnn_padding_bit = BitArray(ga_individual_solution[7*gene_length:8*gene_length])
        cnn_output_size_bit = BitArray(ga_individual_solution[8*gene_length:9*gene_length])
        hidden_neurons_dense_bit = BitArray(ga_individual_solution[9*gene_length:10*gene_length])
        batch_size_bit = BitArray(ga_individual_solution[10*gene_length:11*gene_length])

        lstm_layers = lstm_layers_bit.uint
        lstm_sequential_length = lstm_sequential_length_bit.uint
        lstm_neurons = lstm_neurons_bit.uint
        learning_rate = learning_rate_bit.uint
        cnn_layers = cnn_layers_bit.uint
        cnn_kernel_size = cnn_kernel_size_bit.uint
        cnn_stride = cnn_stride_bit.uint
        cnn_padding = cnn_padding_bit.uint
        cnn_output_size = cnn_output_size_bit.uint
        hidden_neurons_dense = hidden_neurons_dense_bit.uint

        batch_size = batch_size_bit.uint

        # resize hyperparameters
        lstm_layers += 1
        lstm_sequential_length += 1
        lstm_neurons += 1
        learning_rate += 1
        cnn_layers += 1
        cnn_kernel_size += 1
        cnn_stride += 1
        cnn_padding += 1
        cnn_output_size += 1
        hidden_neurons_dense += 1
        batch_size += 1
        learning_rate = learning_rate/100
        batch_size = batch_size * 100
        lstm_neurons *= 10 

        # get rid of possibility of Kernel size being bigger than input size
        if cnn_kernel_size > cnn_output_size + 2* cnn_padding:
            cnn_kernel_size = cnn_output_size + 2* cnn_padding 
            print(f'cnn kernel size changed to {cnn_kernel_size} as it was bigger than the input size')


        # ensure lists are the correct length
        cnn_output_size = [cnn_output_size] * cnn_layers
        cnn_kernel_size = [cnn_kernel_size] * cnn_layers
        cnn_stride = [cnn_stride] * cnn_layers
        cnn_padding = [cnn_padding] * cnn_layers
        hidden_neurons_dense = [hidden_neurons_dense] * (cnn_layers)
        hidden_neurons_dense.append(1)
        hidden_neurons_dense[0] = lstm_sequential_length

        # print(f"lstm Layers =  {lstm_layers}")
        # print(f"lstm Sequential Length =  {lstm_sequential_length}")
        # print(f"lstm Neurons =  {lstm_neurons}")
        # print(f"learning rate =  {learning_rate}")
        # print(f"cnn layers =  {cnn_layers}")
        # print(f"cnn kernel size =  {cnn_kernel_size}")
        # print(f"cnn stride =  {cnn_stride}")
        # print(f"cnn padding =  {cnn_padding}")
        # print(f"cnn neurons =  {cnn_output_size}")
        # print(f"hidden neurons =  {hidden_neurons_dense}")
        # print(f"batch size =  {batch_size}")
        
        # print('Gretas sketchy parameters:', [120, 2, learning_rate*1000, lstm_sequential_length, batch_size, cnn_layers, cnn_output_size[0], cnn_kernel_size[0], cnn_stride[0], cnn_padding[0], lstm_neurons, lstm_layers, hidden_neurons_dense[1] ])

    model = ParametricCNNLSTM(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq, input_lstm).double()
    model.to(device)

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training and evaltuation
    train_hist, val_hist = trainbatch(model, dataset, datasetv, n_epoch, criterion, optimizer, verbose = True)
    model.eval()
    predictions = model(X_test).to('cpu').detach().numpy()
    predictions_real = predictions.squeeze(2) * time_std + time_mean
    y_test_real = y_test.squeeze(2).to('cpu').detach().numpy() * time_std + time_mean
    epoch = np.linspace(1, n_epoch+1, n_epoch)
    plt.plot(predictions_real, linewidth=2, color='red', label = 'Predicted')
    plt.plot(y_test_real, label = 'Actual') 
    plt.xlabel('Instance (-)')
    plt.ylabel('Time to Discharge (seconds)')
    plt.legend()
    plt.show()

    loss = ((predictions.squeeze(2) - y_test.squeeze(2).to('cpu').detach().numpy()) ** 2).mean()
    print(loss)
    
    import matplotlib.pyplot as plt
    plt.plot(epoch, train_hist, label = 'Train Loss')
    plt.plot(epoch, val_hist, label = 'Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE seconds))')
    plt.legend()
    plt.show()
    
    print(model)

# torch.save(model.state_dict(), 'PytorchLSTM/Data_for_first_draft/data_driven_model.pt')
# pd.DataFrame(predictions.squeeze(2)).to_csv('PytorchLSTM/Data_for_first_draft/predictions_data_driven.csv')
# pd.DataFrame(y_test.squeeze(2).to('cpu').detach().numpy()).to_csv('PytorchLSTM/Data_for_first_draft/y_test_data_driven.csv')
# pd.DataFrame(time).to_csv('PytorchLSTM/Data_for_first_draft/time_data_driven.csv')
# torch.save(model, 'PytorchLSTM/Data_for_first_draft/model_data_driven.pt')
