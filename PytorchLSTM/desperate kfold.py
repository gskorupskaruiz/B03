from model import *
import numpy as np
from data_processing import load_gpu_data, load_gpu_data_with_batches
import torch
from torch.utils.data import Dataset
import math 
from main import trainbatch, SeqDataset
import matplotlib.pyplot as plt


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
    
    all_losses = []
    
    # SPECIFY K
    k_fold = 4
    
    #IMPORTANT - IF YOU'RE RUNNING THIS FOR LSTM ONLY CHANGE THE Number OF LSTM INPUTS IN THE  model.py (LINE 243 AFAIK)
    
    if k_fold == 7:
    # for 7-fold cross validation
        kfold_test = [['B0032'],['B0031'], ['B0029'], ['B0018'], ['B0007'],['B0006'],['B0005']]
        kfold_train = [['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0031'], ['B0005', 'B0006', 'B0007', 'B0018','B0029', 'B0032'], ['B0005', 'B0006', 'B0007', 'B0018', 'B0031', 'B0032'],['B0005', 'B0006', 'B0007', 'B0029', 'B0031', 'B0032'], ['B0005', 'B0006', 'B0018', 'B0029', 'B0031', 'B0032'], ['B0005', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032'], ['B0006', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032']]
    elif k_fold == 4:
    # for 4-fold cross validation
        kfold_test = [['B0005'],['B0006'], ['B0007'], ['B0018']]
        kfold_train = [['B0006', 'B0007', 'B0018'], ['B0005', 'B0007', 'B0018'], ['B0005', 'B0006', 'B0018'], ['B0005', 'B0006', 'B0007']]
    
    
    battery = ['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032']

    for i in range(k_fold):
        battery = kfold_train[i]
        test_battery = load_data_normalise(kfold_test[i])
        
        print('Test battery:', kfold_test[i])
        data = load_data_normalise(battery)

        print(hyperparams)
        n_hidden, n_layer, lr, seq, batch_size, num_layers_conv, output_channels_val, kernel_sizes_val, stride_sizes_val, padding_sizes_val, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense_val = hyperparams

        lr = lr/1000
        n_epoch = 3

        test_size = 0.1
        cv_size = 0.1

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

        if loss > 0.5:
            print(f'Loss is greater than 0.5 at {i+1}th cross validation, stopping iteration')
            break
        
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
    # if loss < 0.3:
    #     print('Loss is less than 0.3')
        
    #     with open('PytorchLSTM/Random_optimizer/final_runs_run1.txt', 'a') as f:
    #         print('Writing to file')
    #         f.write(str(hyperparams))
    #         f.write('\t')
    #         f.write(str(loss))
    #         f.write('\n')

    return loss



testing_hyperparameters = [120, 2, 30, 8, 800, 1, 7, 1, 2, 1, 50, 7, 1]
print(run_model_cv(testing_hyperparameters))