import torch 
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_validation_split(X, y, test_size, val_size):

    x_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0) #the input here has to be a numpy array

    ratio_remaining = 1 - test_size
    ratio_val_adjusted = val_size / ratio_remaining

    X_train, X_val, y_train, y_val = train_test_split(x_remaining, y_remaining, test_size=ratio_val_adjusted, shuffle=True, random_state=0) #the input here has to be a numpy array
    
    return [X_train, y_train, X_test, y_test, X_val, y_val]

def preprocessing_dnn(num, test_size, val_size, k, cutoff=False,):
    df = pd.read_csv(f"data/B000" + str(num) +"_TTD.csv")
    if cutoff:
        v = df["Voltage_measured"][:k].values
        c = df["Current_measured"][:k].values
        t = df["Temperature_measured"][:k].values
        t2 = df["Current_charge"][:k].values
        t3 = df["Voltage_charge"][:k].values
        t4 = df["Capacity"][:k].values
        X = np.hstack((v, c))
        X = np.hstack((X, t))
        X = np.hstack((X, t2))
        X = np.hstack((X, t3))
        X = np.hstack((X, t4))
        X = X.reshape(len(v), 6)
        X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

        y = df["TTD"][:k].values
        y = np.array(y)  
        y = y.reshape(len(y), 1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)

        X_train, y_train, X_test, y_test, X_val, y_val = train_test_validation_split(X, y, test_size, val_size)

        return [X, y, X_train, y_train, X_test, y_test, X_val, y_val]
    
    else:
        v = df["Voltage_measured"].values
        c = df["Current_measured"].values
        t = df["Temperature_measured"].values
        t2 = df["Current_charge"].values
        t3 = df["Voltage_charge"].values
        t4 = df["Capacity"].values
        X = np.hstack((v, c))
        X = np.hstack((X, t))
        X = np.hstack((X, t2))
        X = np.hstack((X, t3))
        X = np.hstack((X, t4))
        X = X.reshape(len(v), 6)
        X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

        y = df["TTD"].values
        y = np.array(y)  
        y = y.reshape(len(y), 1)
        values = [np.mean(y), np.std(y)]
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)

        X_train, y_train, X_test, y_test, X_val, y_val = train_test_validation_split(X, y, test_size, val_size)
        X_train, X_val, y_train, y_val= torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)

        return [X, y, X_train, y_train, X_test, y_test, X_val, y_val, values]


def load_data_normalise(battery):
    data = pd.read_csv("data/" + battery + "_TTD.csv")
    print(f"data shape {data.shape}")
    normalized_data = (data-data.mean(axis=0))/data.std(axis=0)
    return normalized_data

def load_data(battery):
	data = pd.read_csv("data/" + battery + "_TTD.csv")
	print("Battery: ", battery)
	print(data)
	return data

data_fields = {
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Current_charge', 'Voltage_charge', 'Time', 'Capacity'}

def train_test_validation_split(X, y, test_size, cv_size):
    X_train, X_test_cv, y_train, y_test_cv = train_test_split(
        X, y, test_size=test_size+cv_size, shuffle=True, random_state=0)

    test_size = test_size/(test_size+cv_size)

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test_cv, y_test_cv, test_size=test_size, shuffle=True, random_state=0)

    return [X_train, y_train, X_test, y_test, X_cv, y_cv]

def load_gpu_data(data, test_size, cv_size):
    y = data["TTD"][:50000]
    X = data[:50000].drop(["TTD"], axis=1)
    X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(X, y, test_size, cv_size)
    print(X_train.shape, X_test.shape, X_cv.shape)
    lex1 = len(X_train)
    X_train = torch.tensor(X_train.values).reshape(int(lex1), len(data_fields)) # changed the reshaping of this 
    y_train = torch.tensor(y_train.values).view(int(lex1), 1)
    lexxx1 = len(X_test)
    X_test = torch.tensor(X_test.values).reshape(int(lexxx1), len(data_fields))
    y_test = torch.tensor(y_test.values).reshape(int(lexxx1), 1)
    lexx1 = len(X_cv)
    X_cv = torch.tensor(X_cv.values).reshape(int(lexx1), len(data_fields))
    y_cv = torch.tensor(y_cv.values).view(int(lexx1), 1)

    print("GPU is availible: ", torch.cuda.is_available())
    if torch.cuda.is_available() == True:
        print('Running on GPU')
        X_train = X_train.to('cuda').float()
        y_train = y_train.to('cuda').float()
        X_test = X_test.to('cuda').float()
        y_test = y_test.to('cuda').float()
        X_cv = X_cv.to('cuda').float()
        y_cv = y_cv.to('cuda').float()

        print("X_train and y_train are on GPU: ", X_train.is_cuda, y_train.is_cuda)
        print("X_test and y_test are on GPU: ", X_test.is_cuda, y_test.is_cuda)
        print("X_cv and y_cv are on GPU: ", X_cv.is_cuda, y_cv.is_cuda)
        print(f"size of X_train: {X_train.size()} and y_train: {y_train.size()}")
    
    return X_train, y_train, X_test, y_test, X_cv, y_cv

