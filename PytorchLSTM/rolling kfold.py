import numpy as np 
from main import *
from model import *
from sklearn.model_selection import TimeSeriesSplit
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
    time_mean, time_std = data["TTD"].mean(axis=0), data["TTD"].std(axis=0)
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)

    return normalized_data, time_mean, time_std

def data_selection(data, which_model):
    y = data["TTD"]
    if which_model == "LSTM-CNN":
        X = data.drop(["TTD"], axis=1)
        input_lstm = 7
    elif which_model == "hybrid":
        X = data.drop(['TTD'], axis=1).drop(["Voltage_measured"], axis=1)
        input_lstm = 8

    return X, y

def load_gpu_data_with_batches_cv(x_train, y_train, x_val, y_val, seq_length, which_model):
    
    # y = data["TTD"]
    if which_model == "LSTM-CNN":
    #     X = data.drop(["TTD"], axis=1)
        input_lstm = 7
    elif which_model == "hybrid":
    #     X = data.drop(['TTD'], axis=1).drop(["Voltage_measured"], axis=1)
        input_lstm = 8
    
    # x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
    
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

def training(model, train_dataloader, n_epoch, lf, optimizer, verbose = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epoch = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # set model to GPU
    #intiate early stopper
    early_stopper = EarlyStopper(patience=1e-16, min_delta=1e-6)
    model.train()
    with torch.no_grad():
        train_loss_history = []
        # val_loss_history = []

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

        # for k, (xv, yv) in enumerate(val_dataloader):
            
        #     target_val = model(xv) #.unsqueeze(2) uncomment this for simple lstm 
        #     #print(target_val.shape, yv.shape, xv.shape)
        #     loss_val = lf(target_val, yv)
        #     loss_v += loss_val.item()

        train_loss = loss/len(train_dataloader)
        # val_loss = loss_v/len(val_dataloader)
        
        train_loss_history.append(train_loss)
        # val_loss_history.append(val_loss)

        # val_loss_history_arr = np.array(val_loss_history)
        print(f"Epoch {i+1}: train loss = {train_loss:.10f}")
        # # if len(val_loss_history) > 5:
        #     if (val_loss_history[-1] > val_loss_history[-2] ) or (np.abs(val_loss_history[-1] - val_loss_history[-2]) < 1e-3) or (val_loss_history_arr[val_loss_history_arr > 1].size > 4):
        #         print(f'early stopper has been activated')
        #         break 
       
    return train_loss_history

def master(hyperparams, which_model, kfold):
    torch.manual_seed(124)
    battery = ['B0005']

    bat_norm, time_mean, time_std = load_data_normalise_cv(battery, which_model)
    X, y = data_selection(bat_norm, which_model)

    tscv = TimeSeriesSplit(max_train_size=None, n_splits=kfold)
    all_loss = []

    for train_index, test_index in tscv.split(X):
        print(f'test idx {test_index}')
        lr, seq, batch_size, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense = hyperparams
        
        n_epoch = 25
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        x_train, x_test = X.head(train_index[-1]), X.iloc[train_index[-1]:test_index[-1]]
        y_train, y_test = y[train_index], y[test_index]

        x_tr, y_tr, x_v, y_v, input_lstm = load_gpu_data_with_batches_cv(x_train, y_train, x_test, y_test, seq, which_model)

        dataset = SeqDataset(x_data=x_tr, y_data=y_tr, seq_len=seq, batch=batch_size)

        model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq, input_lstm).double()
        model.to(device)
        model.weights_init
        
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        train_hist = training(model, dataset, n_epoch, criterion, optimizer, verbose = True)
        model.to(device)

        model.eval()

        loss = criterion(model(x_v), y_v).item()
        all_loss.append(loss)
        predictions = model(x_v).to('cpu').detach().numpy()
        predictions_plot = predictions.squeeze(2) * time_std + time_mean
        y_kfold = y_v.squeeze(2).to('cpu').detach().numpy() * time_std + time_mean
        # plt.plot(predictions_plot, label='pred', linewidth=2, color='red')
        # plt.plot(y_kfold, label='actual', linewidth=2, color='blue')
        
        # plt.legend()
        # plt.show()
        
    
    return all_loss

testing_hyperparameters = [0.00167, 8, 2000, 5, [1, 9, 18, 27, 36], [1, 5, 2.0, 7.0, 9.0], [1, 1, 1, 1, 1], [1, 1, 2, 3, 4], 14, 3, [24, 18, 12, 6, 1]] # 0.06 kfold loss 

print(master(testing_hyperparameters, 'hybrid', 10))
    

