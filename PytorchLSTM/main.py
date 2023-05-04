import pandas as pd
from sklearn.model_selection import train_test_split
from model import *
import numpy as np
from data_processing import load_gpu_data
from torch.utils.data import DataLoader, TensorDataset
# import plot
import matplotlib.pyplot as plt
def load_data_normalise(battery):
    data = []
    # for all battery files combine them into one dataframe
    for i in battery:
        data.append(pd.read_csv("data/" + i + "_TTD.csv"))
    data = pd.concat(data)
    # print(data)
    # normalize the data
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)


    # for i in battery:
    #     data += pd.read_csv("data/" + i + "_TTD.csv").to_numpy().tolist()
    # print(f"data size {len(data)}")
    # data = pd.DataFrame(data)
    # normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
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
    #early_stopper = EarlyStopper(patience=es_patience, min_delta=es_delta)
    # X_train = X_train.double()
    # y_train = y_train.double()
    # X_val = X_val.double()
    # y_val = y_val.double()

    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        target_train = model(X_train).unsqueeze(2) # i changed this 
        #target_train.reshape(4000, 10, 1)
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

        # if verbose:
        print(f"Epoch {i+1}: train loss = {loss_train.item():.4f}, val loss = {loss_val.item():.4f}")

        # if early_stopper.early_stop(loss_val.item()):
        #     #print(f"Early stopping at epoch {i+1}")
        #     break
    return train_loss_history, val_loss_history, epoch

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
            print(f"current lr is {i}")
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

if __name__ == '__main__': 
	# import data
    battery = ['B0005', 'B0006', 'B0007', 'B0018']
    data = load_data_normalise(battery)
    input_size = data.shape[1] -1 #len(data.columns) - 1
    n_hidden = 20 #input_size
    n_layer = 2
    n_epoch = 200
    lr = 0.004
    test_size = 0.1
    cv_size = 0.1
    seq = 20
    # gpu?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #data initialization
    X_train, y_train, X_val, y_val, X_test, y_test = load_gpu_data(data, test_size=test_size, cv_size=cv_size, seq_length=seq)  
    print(X_train.dtype)
    #where is X_train
    print(f"x_train is on {X_train.device}, y_train is on {y_train.device}")


    # LsTM Model initialization
    # modellstm = LSTM1(input_size, n_hidden, n_layer, seq).double() # ahh i changed the seq len thing too 
    model = CNNLSTM(input_size, seq, n_hidden, n_layer).double() 

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training and evaltuation
    train_hist, val_hist, epoch = train(model, X_train, y_train, X_val, y_val, n_epoch, criterion, optimizer, verbose = True)
    predictions = model(X_test).to('cpu').detach().numpy()
    
    plt.plot(predictions)
    plt.plot(y_test.squeeze(2).to('cpu').detach().numpy())
    plt.show()

    loss = ((predictions - y_test.squeeze(2).to('cpu').detach().numpy()) ** 2).mean()
    print(loss)
    
    import matplotlib.pyplot as plt
    plt.plot(epoch, train_hist)
    plt.plot(epoch, val_hist)
    plt.show()
    
    print(model)
