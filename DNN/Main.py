import numpy as np 
import scipy as s
import sklearn as sk 
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt

import dnn_py
from Data_processing import *
from backprop_nn import NeuralNetwork
from losses import *
from activations import *


if __name__ == "__main__":
    # data pre-processing DNN
    X, y, X_train, y_train, X_test, y_test, X_val, y_val, values = preprocessing_dnn(5, 0.1, 0.1, 100, cutoff=False)
   
    # setting the DNN model
    act = torch.nn.Sigmoid()
    model = dnn_py.nn_model(3, 2, 1, act, 1)
    loss_fn = torch.nn.MSELoss()
    n_epoch = 150
    opt = torch.optim.Adam(params=model.parameters(), lr=0.001)
    [train_loss_history_dnn, val_loss_history_dnn, epoch] = dnn_py.train(model, X_train, y_train, X_val, y_val, n_epoch, loss_fn, opt, 1e-6, 1e-4, verbose=False)

    # to know the kfold error uncomment the following line - it will return one number:
    # kf_error = dnn_py.kfold_validation(model, X, y, 10, n_epoch, loss_fn, opt, 1e-6, 1e-4, time=True)
    # print(kf_error)



    # data pre-processing NN EKF
    X_ekf, y_ekf, X_train_ekf, y_train_ekf, X_test_ekf, y_test_ekf, X_val_ekf, y_val_ekf = preprocessing_dnn(5, 0.1, 0.1, 100, cutoff=True)
    # setting the NN-EKF model
    np.random.seed(1234)
    rng = np.random.RandomState(123)
    state = rng.__getstate__()
    nn = NeuralNetwork(layers=[3, 40, 1], activations=[ReLU(), Linear()], loss=Unity(), rng=rng)
    train_loss, val_loss = nn.train_ekf(X_train_ekf.T, y_train_ekf.reshape(1, -1), P=5.3, R=1e-2, Q=214.9, epochs=70, val=(X_val_ekf.T, y_val_ekf.reshape(1, -1)), eta=1e-2)
    


    # making predictions using both methods 
    # using the DNN model 
    predictions = model(X_test.float()).detach().numpy()
    predictions = predictions * values[1] + values[0]

    # using the EKF hybrid thing 
    _, predictions_ekf = nn.feedforward(X_test.detach().numpy().T)
    predictions_ekf = np.array(predictions_ekf[-1][0])
    predictions_ekf = np.abs(predictions_ekf * values[1] + values[0])

    # taking an average of these predictions
    pred = np.zeros((X_test.shape[0], 2))
    pred[:, 0] = predictions.flatten()
    pred[:, 1] = predictions_ekf.flatten()
    pred = np.mean(pred, axis=1)
    true = y_test.detach().numpy() * values[1] + values[0]



    # plotting error on training and validation 
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].plot(train_loss.keys(), train_loss.values(), label="train ekf")
    axs[0].plot(val_loss.keys(), val_loss.values(), label="validation ekf")
    axs[1].plot(epoch, train_loss_history_dnn, label='train DNN')
    axs[1].plot(epoch, val_loss_history_dnn, label='val DNN')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    #plt.title("Loss on training and validation sets comparison")
    axs[0].legend()
    axs[1].legend()
    plt.show()

    

    

