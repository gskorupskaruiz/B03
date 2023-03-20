import pandas as pd
from sklearn.model_selection import train_test_split
from model import *
from sklearn import preprocessing

def load_data_split_normalise(battery):
	data = pd.read_csv("data/" + battery + "_TTUD.csv")
    data = preprocessing.fit_transform(data)
    return data

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


	
if __name__ == '__main__': 
	# import data
    
    input_size = 8 # shouldn't be hard coded
    n_hidden = input_size
    n_layer = 2
    # gpu?

    # LSTM Model initialization
    model = LSTM1(input_size, n_hidden, n_layer)

