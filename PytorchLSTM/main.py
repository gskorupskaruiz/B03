import pandas as pd
from sklearn.model_selection import train_test_split
from model import *
import numpy as np
from data_processing import load_gpu_data

def load_data_normalise(battery):

    data = pd.read_csv("data/" + battery + "_TTD.csv")
    normalized_data = (data-data.mean())/data.std()
    return normalized_data
    

print(load_data_normalise("B0005"))

def train_test_validation_split(X, y, test_size, cv_size):
    """
    TODO:
    Part 0, Step 3: 
        - Use the sklearn {train_test_split} function to split the dataset (and the labels) into
            train, test and cross-validation sets
    """
    X_train, X_test_cv, y_train, y_test_cv = train_test_split(
        X, y, test_size=test_size+cv_size, shuffle=True, random_state=0)

    test_size = test_size/(test_size+cv_size)

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test_cv, y_test_cv, test_size=test_size, shuffle=True, random_state=0)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]

def testing_func(num, group_for_test):
    rmse_test, result_test = 0, list()

    for ite in range(1, num + 1):
        X_test = group_for_test.get_group(ite).iloc[:, 2:]
        X_test_tensors = torch.Tensor(X_test.to_numpy())
        X_test_tensors = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        test_predict = model.forward(X_test_tensors)
        data_predict = max(test_predict[-1].detach().numpy(), 0)
        result_test.append(data_predict)
        rmse_test = np.add(np.power((data_predict - y_test.to_numpy()[ite - 1]), 2), rmse_test)

    rmse_test = (np.sqrt(rmse_test / num)).item()
    return result_test, rmse_test

def train(model, battery):

     # number of samples
     # grouped data per sample

    norm_data = load_data_normalise(battery)
    X_train, y_train, X_test, y_test, X_cv, y_cv = load_gpu_data(norm_data, test_size=test_size, cv_size=cv_size)


    # X = norm_data.drop("TTD", axis = 1)
    # y = norm_data["TTD"]
    # X_train, y_train, X_test, y_test, X_cv, y_cv  = train_test_validation_split(X, y, test_size, cv_size)
    num_train = len(X_train)
    print("num_train: ", num_train)
    rmse_temp = 1000

    for epoch in range(1, n_epoch +1):

        #model.train()
        #epoch_loss = 0

        outputs = model(X_train) # forward pass
        optimizer.zero_grad() # calc and set grad = 0
        loss = criterion(outputs, y_train) # calc loss for current pass
        epoch_loss += loss.item()
        loss.backward() # update model parameters
        optimizer.step # update loss func
        
        if epoch % 1 == 0:
             
            model.eval() # evaluate model
            result, rmse = testing_func(num_test, group_test)  #run test through model

            if rmse_temp < rmse and rmse_temp <5:
                result, rmse = result_temp, rmse_temp
                break
            
            rmse_temp, result_temp = rmse, result #store last rmse
            print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, epoch_loss / num_train, rmse))
   
	
if __name__ == '__main__': 
	# import data
 
    battery = 'B0005'
    data = load_data_normalise(battery)
    input_size = len(data.columns) - 1

    input_size = 8 # shouldn't be hard coded
    n_hidden = input_size
    n_layer = 2
    n_epoch = 150
    lr = 0.01
    test_size = 0.2
    cv_size = 0.2
    # gpu?


    # LSTM Model initialization
    model = LSTM1(input_size, n_hidden, n_layer)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training and evaltuation
    result, rmse = train(model, battery )
