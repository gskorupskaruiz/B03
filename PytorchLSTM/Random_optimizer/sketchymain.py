import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from model import *
from data_processing import load_gpu_data, load_gpu_data_with_batches
from main import trainbatch, load_data_normalise, SeqDataset

"""
THIS MAIN TRAINS FOR CROSS VALIDATION

"""
                
def load_data_normalise(battery):
    data = []
    # for all battery files combine them into one dataframe
    for i in battery:
        data.append(pd.read_csv("data/" + i + "_TTD.csv"))
    data = pd.concat(data)
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
    return normalized_data


def run_model_cv(hyperparams):
    # import data
    all_losses = []
    kfold_test = [['B0032'],['B0031'], ['B0029'], ['B0018'], ['B0007'],['B0006'],['B0005']]
    kfold_train = [['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0031'], ['B0005', 'B0006', 'B0007', 'B0018','B0029', 'B0032'], ['B0005', 'B0006', 'B0007', 'B0018', 'B0031', 'B0032'],['B0005', 'B0006', 'B0007', 'B0029', 'B0031', 'B0032'], ['B0005', 'B0006', 'B0018', 'B0029', 'B0031', 'B0032'], ['B0005', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032'], ['B0006', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032']]
    battery = ['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0030', 'B0031', 'B0032'] # no clue why but battery 5 just doesnt work - even though it has the same format and i use the same code :(

    for i in range(7):
        battery = kfold_train[i]
        test_battery = load_data_normalise(kfold_test[i])
        
        print(kfold_test[i])
        data = load_data_normalise(battery)
        input_size = data.shape[1] - 1 #len(data.columns) - 1
        print(f'size of input is {input_size}')
        print(hyperparams)
        n_hidden, n_layer, lr, seq, batch_size, num_layers_conv, output_channels_val, kernel_sizes_val, stride_sizes_val, padding_sizes_val, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense_val = hyperparams

        lr = lr/1000
        n_epoch = 3
        test_size = 0.1
        cv_size = 0.1
        
        # gpu?
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #data initialization
        X_train, y_train, X_test, y_test, X_val, y_val = load_gpu_data_with_batches(data, test_size=test_size, cv_size=cv_size, seq_length=seq)          
        X_train, y_train, X_test, y_test, X_val, y_val = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device), X_val.to(device), y_val.to(device)
        X_kfold, y_kfold, X_testkf, y_testkf, X_valkf, y_valkf = load_gpu_data_with_batches(test_battery, test_size=0.1, cv_size=0.1, seq_length=seq)
        
        
        dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq, batch=batch_size)
        datasetv = SeqDataset(x_data=X_valkf, y_data=y_valkf, seq_len=seq, batch=batch_size)
        # LsTM Model initialization
    
        output_channels = [output_channels_val] * num_layers_conv
        kernel_sizes = [kernel_sizes_val] * num_layers_conv
        stride_sizes = [stride_sizes_val] * num_layers_conv
        padding_sizes = [padding_sizes_val] * num_layers_conv
        
        hidden_neurons_dense = [seq, hidden_neurons_dense_val, 1]
        
        model = ParametricCNNLSTM(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq).double()
        model.train()
        model.to(device)
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        # training and evaltuation
        train_hist, val_hist = trainbatch(model, dataset, datasetv, n_epoch, criterion, optimizer, verbose = True)
        model.eval()
        predictions = model(X_kfold).to('cpu').detach().numpy()
    
        loss = ((predictions.squeeze(2) - y_kfold.squeeze(2).to('cpu').detach().numpy()) ** 2).mean()

        all_losses.append(loss)
        if loss > 0.5:
            print('Loss at', i+1, 'th cross validation is to high, breaking')
            break
        
        print(f'Loss at {i+1}th cross validation', loss)
        
        # PRINT PREDICTIONS AT EACH FOLD
        
        # plt.style.use('seaborn-dark')
        # plt.plot(predictions.squeeze(2), label='pred', linewidth=2, color='red')
        # plt.plot(y_kfold.squeeze(2).to('cpu').detach().numpy()) 
        
        # plt.legend()
        # plt.show()
        
    loss = np.mean(all_losses)


    if loss != 'nan':

        print(f'btw the current loss is {loss.round(5)}')
    
    if loss < 0.3:
        print('loss is good, saving hyperparams')
        
        with open('PytorchLSTM/final_runs_Run1.txt', 'a') as f:
            print('Writing to file')
            f.write(str(hyperparams))
            f.write('\t')
            f.write(str(loss))
            f.write('\n')
            
    
    # plt.plot(epoch, train_hist)
    # plt.plot(epoch, val_hist)
    # plt.show()
    
   # print(model)

    return loss