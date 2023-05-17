import pandas as pd
from model import *
import numpy as np
from data_processing import load_gpu_data, load_gpu_data_with_batches
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset
import math 
from bitstring import BitArray  
# import plot
import matplotlib.pyplot as plt
def load_data_normalise(battery):
    data = []
    # for all battery files combine them into one dataframe
    for i in battery:
        data.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
    data = pd.concat(data)
    # print(data)
    time = data["Time"]
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)


    # for i in battery:
    #     data += pd.read_csv("data/" + i + "_TTD.csv").to_numpy().tolist()
    # print(f"data size {len(data)}")
    # data = pd.DataFrame(data)
    # normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
    return normalized_data, time

def check_nan(battery):
    data = []
    data.append(pd.read_csv("data/" + 'B0005' + "_TTD - with SOC.csv"))
    data = pd.concat(data)
    return data.dropna()
    

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

    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        target_train = model(X_train).unsqueeze(2) # i changed this 
        print(target_train.shape, y_train.shape)

        target_val = model(X_val).unsqueeze(2) # i changed this - added the unsqueeze thing 
        loss_train = lf(target_train, y_train)
        loss_val = lf(target_val, y_val)
        train_loss_history.append(loss_train.item())
        val_loss_history.append(loss_val.item())

        epoch.append(i+1)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        print(f"Epoch {i+1}: train loss = {loss_train.item():.4f}, val loss = {loss_val.item():.4f}")
    return train_loss_history, val_loss_history, epoch

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


# def train(model, battery):
#     model.to(device) # set model to GPU
#      # number of samples
#      # grouped data per sample

#     norm_data = load_data_normalise(battery)
    
#     X_train, y_train, X_test, y_test, X_cv, y_cv = load_gpu_data(norm_data, test_size=test_size, cv_size=cv_size)
#     train_dataset = TensorDataset(X_train, y_train)
#     print(y_train.shape)
#     train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
#     val_dataset = TensorDataset(X_cv, y_cv)
#     val_loader = DataLoader(val_dataset, batch_size=10)
#     test_dataset = TensorDataset(X_test, y_test)
#     test_loader = DataLoader(test_dataset, batch_size=10)
#     # X = norm_data.drop("TTD", axis = 1)
#     # y = norm_data["TTD"]
#     # X_train, y_train, X_test, y_test, X_cv, y_cv  = train_test_validation_split(X, y, test_size, cv_size)
#     num_train = len(X_train)
#     rmse_temp = 1000
#     epoch_loss = 0
#     for epoch in range(n_epoch):
#         # print(X_train[0]) this highlighets the issue that the data doesn't change so need to fix data loaders
#         model.train() # set model to training mode
#         optimizer.zero_grad() # calc and set grad = 0
#         outputs = model(X_train) # forward pass
#         print(f"size of output {outputs.shape}, size of y_train {y_train.shape}")
#         loss = criterion(outputs, y_train) # calc loss for current pass
#         loss = torch.unsqueeze(loss, 0) # add dimension to loss
#         epoch_loss += loss.item() # add loss to epoch loss
#         loss.backward() # update model parameters
#         optimizer.step() # update loss func   
        
#         model.eval() # evaluate mode model (ie no drop out)
#         result, rmse = testing_func(X_test, y_test)  #run test through model

#         if rmse_temp < rmse and rmse_temp <0.5:
#             result, rmse = result_temp, rmse_temp
#             print("Early stopping ")
#             break
        
#         rmse_temp, result_temp = rmse, result #store lst rmse
#         print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, loss , rmse))

#     return rmse, result
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

        #print(f'size of xdata from the dataset is {x_data.shape}')

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
    battery = ['B0018']
    data, time = load_data_normalise(battery)
    input_size = data.shape[1] - 2
    print(f'input_size of data is {input_size}') 
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
    X_train, y_train, X_test, y_test, X_val, y_val = load_gpu_data_with_batches(data, test_size=test_size, cv_size=cv_size, seq_length=seq)  
    X_train, y_train, X_test, y_test, X_val, y_val = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device), X_val.to(device), y_val.to(device)
    print(y_test)
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
        ga_individual_solution =  [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]

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

        print(f"lstm Layers =  {lstm_layers}")
        print(f"lstm Sequential Length =  {lstm_sequential_length}")
        print(f"lstm Neurons =  {lstm_neurons}")
        print(f"learning rate =  {learning_rate}")
        print(f"cnn layers =  {cnn_layers}")
        print(f"cnn kernel size =  {cnn_kernel_size}")
        print(f"cnn stride =  {cnn_stride}")
        print(f"cnn padding =  {cnn_padding}")
        print(f"cnn neurons =  {cnn_output_size}")
        print(f"hidden neurons =  {hidden_neurons_dense}")
        print(f"batch size =  {batch_size}")

    model = ParametricCNNLSTM(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq).double()
    model.to(device)

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training and evaltuation
    train_hist, val_hist = trainbatch(model, dataset, datasetv, n_epoch, criterion, optimizer, verbose = True)
    model.eval()
    predictions = model(X_test).to('cpu').detach().numpy()
    epoch = np.linspace(1, n_epoch+1, n_epoch)
    plt.plot(predictions.squeeze(2), linewidth=2, color='red', label = 'Predicted')
    plt.plot(y_test.squeeze(2).to('cpu').detach().numpy(), label = 'Actual') 
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time to Discharge (seconds))')
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
