
import pandas as pd
from model import *
import numpy as np

import torch
from torch.utils.data import Dataset
import math 
from main import trainbatch, SeqDataset
import matplotlib.pyplot as plt


def load_data_normalise_cv(battery, which_model):
    data = []
    # for all battery files combine them into one dataframe
    
    if which_model == "LSTM-CNN":
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD.csv"))
        
    elif which_model == "hybrid":
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
            
    data = pd.concat(data)   
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
    return normalized_data

def load_gpu_data_with_batches_cv(data, seq_length, which_model):
    
    y = data["TTD"]
    if which_model == "LSTM-CNN":
        X = data.drop(["TTD"], axis=1)
        input_lstm = 7
    elif which_model == "hybrid":
        X = data.drop(["TTD"], axis=1).drop(["Voltage_measured"], axis=1)
        input_lstm = 8
    x_tr = []
    y_tr = []
    for i in range(seq_length, len(X)):
        x_tr.append(X.values[i-seq_length:i])
        y_tr.append(y.values[i])
		
    x_tr = torch.tensor(np.array(x_tr))
    y_tr = torch.tensor(y_tr).unsqueeze(1).unsqueeze(2)

    if torch.cuda.is_available() == True:
        print('Running on GPU')
        X = x_tr.to('cuda').double()
        y = y_tr.to('cuda').double()

    else:
        X = x_tr.clone().detach().double()
        y = y_tr.clone().detach().double()

    return X, y, input_lstm



def run_model_cv(hyperparams, which_model, k_fold):
    
    all_losses = []
    
    all_batteries = ['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032']
    
    k_fold_batteries = all_batteries[:k_fold]
   
    kfold_test = []
    kfold_train = []
    
    for i in range(k_fold):
        kfold_test.append([k_fold_batteries[i]])
        other_batteries = k_fold_batteries.copy()
        other_batteries.remove(k_fold_batteries[i])
        kfold_train.append(other_batteries)
                
        battery = kfold_train[i]
        test_battery = load_data_normalise_cv(kfold_test[i], which_model)
        data = load_data_normalise_cv(battery, which_model)
        print('Test battery:', kfold_test[i])
        

        print(hyperparams)
        n_hidden, n_layer, lr, seq, batch_size, num_layers_conv, output_channels_val, kernel_sizes_val, stride_sizes_val, padding_sizes_val, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense_val = hyperparams

        lr = lr/1000
        n_epoch = 3

        test_size = 0.1
        cv_size = 0.1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        #data initialization
        X_train, y_train, input_lstm = load_gpu_data_with_batches_cv(data, seq_length=seq, which_model=which_model)          
        
        #X_train, y_train = X_train.to(device), y_train.to(device)
        
        X_test, y_test, _ = load_gpu_data_with_batches_cv(test_battery, seq_length=seq, which_model=which_model)

        #X_test, y_test = X_test.to(device), y_test.to(device)
        
        dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq, batch=batch_size)
        datasetv = SeqDataset(x_data=X_test, y_data=y_test, seq_len=seq, batch=batch_size)
        # LsTM Model initialization
        
        output_channels = [output_channels_val] * num_layers_conv
        kernel_sizes = [kernel_sizes_val] * num_layers_conv
        stride_sizes = [stride_sizes_val] * num_layers_conv
        padding_sizes = [padding_sizes_val] * num_layers_conv
        hidden_neurons_dense = [seq, hidden_neurons_dense_val, 1]
        
        model = ParametricCNNLSTM(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq, input_lstm).double()
        model.train()
        
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        train_hist, val_hist = trainbatch(model, dataset, datasetv, n_epoch, criterion, optimizer, verbose = True)
        model.to(device)

        # training and evaltuation
        model.eval()
        
        predictions = model(X_test).to('cpu').detach().numpy()
    
        loss = ((predictions.squeeze(2) - y_test.squeeze(2).to('cpu').detach().numpy()) ** 2).mean()

        # if loss > 0.5:
        #     print(f'Loss is greater than 0.5 at {i+1}th cross validation, stopping iteration')
        #     break
        
        print(f'Loss at {i+1}th cross validation', loss)
        all_losses.append(loss)
        
        # PLOT THE PREDICTIONS FOR EACH FOLD
        
        # plt.plot(predictions.squeeze(2), label='pred', linewidth=2, color='red')
        # plt.plot(y_kfold.squeeze(2).to('cpu').detach().numpy()) 
        # plt.legend()
        # plt.show()
    
    
    loss = np.mean(all_losses)
    
    epoch = np.linspace(1, n_epoch+1, n_epoch)
    



    if loss != 'nan':
    #    print(f'no wayy sooo cooool the model predicts! :)')
        print(f'btw the current loss is {loss.round(5)}')
    
    
    # UNCOMMENT IF YOU WANT TO SAVE THE LOSSES
    if loss < 0.5:
        print('Loss is less than 0.5')
        
        with open('PytorchLSTM/Random_optimizer/ga_runs1.txt', 'a') as f:
            print('Writing to file')
            f.write(str(hyperparams))
            f.write('\t')
            f.write(str(loss))
            f.write('\n')

    return loss

"""
Define the hyperparameters to be tested

(comment this out if you're optimising the hyperparameters)

"""


# testing_hyperparameters = [120, 2, 30, 8, 800, 1, 7, 1, 2, 1, 50, 7, 1]
# print(run_model_cv(testing_hyperparameters, 'LSTM-CNN', 4))