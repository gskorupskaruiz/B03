import matplotlib.pyplot as plt
from model import ParametricCNNLSTM
import torch

def read_model(model_name):
    model = torch.load(f'PytorchLSTM\{model_name}')
    model.eval()
    return model

def generate_prediction(model, X_test):
    prediction = model(X_test)
    return prediction



def plot_loss(epoch, train_loss, val_loss):
    plt.plot(epoch, train_loss, label='Training loss')
    plt.plot(epoch, val_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_prediction(prediction, y_actual):
    plt.plot(prediction, label='Prediction')
    plt.plot(y_actual, label='Actual')
    plt.xlabel('Time')
    plt.ylabel('Time to Discharge')
    plt.legend()
    plt.show()

read_model('hyrbidmodel')


