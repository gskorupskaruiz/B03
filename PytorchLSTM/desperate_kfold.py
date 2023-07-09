
import pandas as pd
from model import *
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import math 
from main import trainbatch, SeqDataset
import matplotlib.pyplot as plt
from GeneticCNNLSTMOptimiser_kfold import basis_func

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
    time_mean, time_std = data["TTD"].mean(axis=0), data["TTD"].std(axis=0)
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)

    return normalized_data, time_mean, time_std

def load_gpu_data_with_batches_cv(data, seq_length, which_model):
    
    y = data["TTD"]
    if which_model == "LSTM-CNN":
        X = data.drop(["TTD"], axis=1)
        input_lstm = 7
    elif which_model == "hybrid":
        X = data.drop(['TTD'], axis=1).drop(["Voltage_measured"], axis=1)
        input_lstm = 8
    
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    x_tr = []
    y_tr = []
    for i in range(seq_length, len(x_train)):
        x_tr.append(x_train.values[i-seq_length:i])
        y_tr.append(y_train.values[i])
		
    x_tr = torch.tensor(np.array(x_tr))
    y_tr = torch.tensor(y_tr).unsqueeze(1).unsqueeze(2)

    x_v = []
    y_v = []
    for i in range(seq_length, len(x_val)):
        x_v.append(x_val.values[i-seq_length:i])
        y_v.append(y_val.values[i])

    x_v = torch.tensor(np.array(x_v))
    y_v = torch.tensor(y_v).unsqueeze(1).unsqueeze(2)

    if torch.cuda.is_available() == True:
        # print('Running on GPU')
        x_training = x_tr.to('cuda').double()
        y_training = y_tr.to('cuda').double()
        x_validation = x_v.to('cuda').double()
        y_validation = y_v.to('cuda').double()

    else:
        x_training = x_tr.clone().detach().double()
        y_training = y_tr.clone().detach().double()
        x_validation = x_v.clone().detach().double()
        y_validation = y_v.clone().detach().double()
    

    return x_training, y_training, x_validation, y_validation, input_lstm

# def basis_func(scaling_factor, hidden_layers):
    
#     scaling_factor = scaling_factor + 2
#     basis = np.cos(np.linspace(-np.pi/2, np.pi/2, hidden_layers)) * scaling_factor
#     basis = (basis).astype(int)
#     for i in range(hidden_layers): 
#         if basis[i] == 0: basis[i] = 1
#     return basis

def run_model_cv(hyperparams, which_model, k_fold, save_for_plots):
    print(f'why are you running')
    save_for_plots = True
    torch.manual_seed(124)
    
    all_losses = []
    print('STARTING')
    
    all_batteries = ['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032']
    
    k_fold_batteries = all_batteries[:k_fold]
   
    kfold_test = []
    kfold_train = []
    if save_for_plots:
        # init variables for plots
        kthlostperIndivudual = np.zeros(k_fold)
        kth_predictions = []
        kth_actual = []
    for i in range(k_fold):
        kfold_test.append([k_fold_batteries[i]])
        other_batteries = k_fold_batteries.copy()
        other_batteries.remove(k_fold_batteries[i])
        kfold_train.append(other_batteries)
                
        battery = kfold_train[i]
        test_battery, time_mean, time_std = load_data_normalise_cv(kfold_test[i], which_model)
        data, time_mean_d, time_std_m = load_data_normalise_cv(battery, which_model)
        print('Test battery:', kfold_test[i])
        
        test_battery = test_battery[:340]
        print(f'hyperparameters = {hyperparams}')
        lr, seq, batch_size, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense = hyperparams
        
        n_epoch = 25
        test_size = 0.1
        cv_size = 0.1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        #data initialization
        X_train, y_train, X_val, y_val, input_lstm = load_gpu_data_with_batches_cv(data, seq_length=seq, which_model=which_model)          
        
        #X_train, y_train = X_train.to(device), y_train.to(device)
        
        X_test, y_test, _, _, _ = load_gpu_data_with_batches_cv(test_battery, seq_length=seq, which_model=which_model)

        #X_test, y_test = X_test.to(device), y_test.to(device)
        
        dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq, batch=batch_size)
        datasetv = SeqDataset(x_data=X_val, y_data=y_val, seq_len=seq, batch=batch_size)
        # LsTM Model initialization
        
        # output_channels = [output_channels_val] * num_layers_conv
        # kernel_sizes = [kernel_sizes_val] * num_layers_conv
        # stride_sizes = [stride_sizes_val] * num_layers_conv
        # padding_sizes = [padding_sizes_val] * num_layers_conv
        # hidden_neurons_dense = [seq, hidden_neurons_dense_val, 1]
        
        
        # output_channels = basis_func(output_channels_val, num_layers_conv)
        # kernel_sizes = basis_func(kernel_sizes_val, num_layers_conv)
        # stride_sizes = basis_func(stride_sizes_val, num_layers_conv)
        # padding_sizes =  basis_func(padding_sizes_val, num_layers_conv)
        # hidden_neurons_dense = basis_func(hidden_neurons_dense[1], num_layers_conv)
        # hidden_neurons_dense[0] = seq
        # hidden_neurons_dense[-1] = 1
        
        # print(f"lstm Layers =  {num_layers_lstm}")
        # print(f"lstm Sequential Length =  {seq}")
        # print(f"lstm Neurons =  {hidden_size_lstm}")
        # print(f"learning rate =  {lr}")
        # print(f"cnn layers =  {num_layers_conv}")
        # print(f"cnn kernel size =  {kernel_sizes}")
        # print(f"cnn stride =  {stride_sizes}")
        # print(f"cnn padding =  {padding_sizes}")
        # print(f"cnn neurons =  {output_channels}")
        # print(f"hidden neurons =  {hidden_neurons_dense}")
        # print(f"batch size =  {batch_size}")
        
        torch.manual_seed(100)
        torch.manual_seed(100)
        model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq, input_lstm).double()
        if not model.hyperparameter_check():
            loss = 100
            all_losses.append(loss)
            print(f'skip k_fold becasue bad set of hyperparameters')
            break
        
        model.to(device)
        model.weights_init
        
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        train_hist, val_hist = trainbatch(model, dataset, datasetv, n_epoch, criterion, optimizer, verbose = True)
        model.to(device)

        # training and evaltuation
        model.eval()
        
        predictions = model(X_test).to('cpu').detach().numpy()
        if save_for_plots:
            kth_predictions.append(predictions)
            kth_actual.append(y_test.squeeze(2).to('cpu').detach().numpy())
        
        loss = criterion(model(X_test), y_test).item()
        # if loss > 0.5:
        #     print(f'Loss is greater than 0.5 at {i+1}th cross validation, stopping iteration')
        #     break
        all_losses_arr = np.array(all_losses)
        print(f'Loss at {i+1}th cross validation', loss)
        if all_losses_arr[all_losses_arr >= 1].size >= 1:
            loss = 1000
            all_losses.append(loss)
            print(f'skip k_fold due to bad loss')
            break
            
        all_losses.append(loss)
        if save_for_plots:
            kthlostperIndivudual[i] += loss
             # PLOT THE PREDICTIONS FOR EACH FOLD
            predictions_plot = predictions.squeeze(2) * time_std + time_mean
            y_kfold = y_test.squeeze(2).to('cpu').detach().numpy() * time_std + time_mean
            plt.plot(predictions_plot, label='pred', linewidth=2, color='red')
            plt.plot(y_kfold, label='actual', linewidth=2, color='blue')
            
            plt.legend()
            plt.show()

            
        
        if i==2:
            save_model = True
            if save_model:
                print(f'this model will now be saved')
                torch.save(model.state_dict(), f'{which_model}model_B0007.pt')
                print('Model saved')
    
    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.plot(kth_predictions[i].squeeze(2) * time_std + time_mean, label='pred', linewidth=2, color='red')
    #     plt.plot(kth_actual[i].squeeze(2) * time_std + time_mean, label='actual', linewidth=2, color='blue')
    #     plt.legend()
    #     plt.title(f'kth fold {i+1}')
    # plt.show()
    
    
    loss = np.mean(all_losses)
    
    epoch = np.linspace(1, n_epoch+1, n_epoch)

    if loss != 'nan':
    #    print(f'no wayy sooo cooool the model predicts! :)')
        print(f'btw the mean of all losses is {loss.round(5)}')
    # print(f'the values of i is {i}')
    # if i==2:
    #     save_model = True
    #     if save_model:
    #         print(f'this model will now be saved')
    #         torch.save(model.state_dict(), f'{which_model}model_B0007.pt')
    #         print('Model saved')

    
    # UNCOMMENT IF YOU WANT TO SAVE THE LOSSES
    if loss < 0.5:
        print('Loss is less than 0.5')
        
        with open('PytorchLSTM/Random_optimizer/ga_runs_greta.txt', 'a') as f:
            print('Writing to file')
            f.write(str(hyperparams))
            f.write('\t')
            f.write(str(loss))
            f.write('\n')

    return loss




"""
Define the hyperparameters to be tested

(comment this out if you're optimising the hyperparameters)

# """


#testing_hyperparameters = [120, 60, 50.0, 3, 200, 2, [3, 3], [7, 7], [3, 3], [7, 7], 60, 1, [2, 1]]
# testing_hyperparameters = [0.050, 20, 600, 1, [8], [4], [2], [4], 10, 3, [4, 1]] # trained lstmcnn (overnight run)
# testing_hyperparameters = [0.02282, 13, 1120, 1, [1], [1], [1], [1],14,1,[1, 1]] #alexis best ones yet 0.09 cross validation 
# testing_hyperparameters = [0.00167, 8, 2000, 5, [1, 9, 18, 27, 36], [1, 5, 2.0, 7.0, 9.0], [1, 1, 1, 1, 1], [1, 1, 2, 3, 4], 14, 3, [1, 6, 12, 18, 24, 1]] # 0.06 kfold loss 
testing_hyperparameters = [0.00167, 8, 2000, 5, [1, 9, 18, 27, 36], [1, 5, 2.0, 7.0, 9.0], [1, 1, 1, 1, 1], [1, 1, 2, 3, 4], 14, 3, [24, 18, 12, 6, 1]] # 0.06 kfold loss 
# testing_hyperparameters = [0.024, 14, 1612, 4, [1, 1, 2, 3], [1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 3, 4], 22, 1, [1, 2, 4, 6, 1]]
# testing_hyperparameters = [0.01851, 12, 723, 1, [1], [1], [1], [1], 3, 1, [1, 1]] # 0.08

# testing_hyperparameters = [0.01264, 12, 425, 1, [1], [1], [1], [1], 22, 1, [1, 1]] 
# amount of training data not sufficient to learn rul? multiple neural networks for rul? - different models have different accuracy at different parts of the graph 
print(run_model_cv(testing_hyperparameters, 'hybrid', 4, True))
