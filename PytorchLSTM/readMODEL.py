import matplotlib.pyplot as plt
from model import ParametricCNNLSTM
import torch
from bitstring import BitArray

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

# read_model('hyrbidmodel')


ga_individual_solution = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]

gene_length = 3
    # decode GA solution to get hyperparamteres
lstm_layers_bit = BitArray(ga_individual_solution[0:gene_length]) # don't understand the bitarray stuff yet or the length given per hyperparameter
lstm_neurons_bit = BitArray(ga_individual_solution[gene_length:2*gene_length])
lstm_sequential_length_bit = BitArray(ga_individual_solution[2*gene_length:3*gene_length])
learning_rate_bit = BitArray(ga_individual_solution[3*gene_length:4*gene_length])
cnn_layers_bit = BitArray(ga_individual_solution[4*gene_length:5*gene_length])
cnn_kernel_size_bit = BitArray(ga_individual_solution[5*gene_length:6*gene_length])
cnn_stride_bit = BitArray(ga_individual_solution[6*gene_length:7*gene_length])
cnn_padding_bit = BitArray(ga_individual_solution[7*gene_length:8*gene_length])
cnn_output_size_bit = BitArray(ga_individual_solution[8*gene_length:9*gene_length])
hidden_neurons_dense_bit = BitArray(ga_individual_solution[9*gene_length:10*gene_length])
batch_size_bit = BitArray(ga_individual_solution[10*gene_length:11*gene_length])

lstm_layers = lstm_layers_bit.uint
lstm_sequential_length = lstm_sequential_length_bit.uint
lstm_neurons = lstm_neurons_bit.uint
learning_rate = learning_rate_bit.uint
cnn_layers = cnn_layers_bit.uint
cnn_kernel_size = cnn_kernel_size_bit.uint
cnn_stride = cnn_stride_bit.uint
cnn_padding = cnn_padding_bit.uint
cnn_output_size = cnn_output_size_bit.uint
hidden_neurons_dense = hidden_neurons_dense_bit.uint

batch_size = batch_size_bit.uint

# resize hyperparameters
lstm_layers += 1
lstm_sequential_length += 1
lstm_neurons += 1
learning_rate += 1
cnn_layers += 1
cnn_kernel_size += 1
cnn_stride += 1
cnn_padding += 1
cnn_output_size += 1
hidden_neurons_dense += 1
batch_size += 1
learning_rate = learning_rate/100
batch_size = batch_size * 100
lstm_neurons *= 10 


print(f"lstm Layers =  {lstm_layers}")
print(f"lstm Sequential Length =  {lstm_sequential_length}")
print(f"lstm Neurons =  {lstm_neurons}")
print(f"learning rate =  {learning_rate}")
print(f"cnn layers =  {cnn_layers}")
print(f"cnn kernel size =  {cnn_kernel_size}")
print(f"cnn stride =  {cnn_stride}")
print(f"cnn padding =  {cnn_padding}")
print(f"cnn neurons =  {cnn_output_size}")
print(f"hidden neurons =  {hidden_neurons_dense}")
print(f"batch size =  {batch_size}")
