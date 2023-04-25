import torch 
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def train_test_validation_split(X, y, test_size, val_size):
    from sklearn.model_selection import train_test_split

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
        X = np.hstack((v, c))
        X = np.hstack((X, t))
        X = X.reshape(len(v), 3)
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
        X = np.hstack((v, c))
        X = np.hstack((X, t))
        X = X.reshape(len(v), 3)
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

