import matplotlib.pyplot as plt
from model import ParametricLSTMCNN
import torch
from desperate_kfold import load_data_normalise_cv
from sklearn.model_selection import train_test_split

def read_model(model_name, hyperparameters):
    lr, seq, batch_size, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense = hyperparameters
    if model_name == 'hybrid':
        input_lstm = 8
    elif model_name == 'LSTM-CNN':
        input_lstm = 7
    else:
        print('Invalid model name')
        print(f'model name: {model_name}')
        return
    model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq, input_lstm)
    # model.load_state_dict(torch.load(f'C:/Users/alexi/OneDrive - Delft University of Technology/BSc2 AE/Q3/Project/B03/PytorchLSTM/'+model_name+'model.pt'))
    model.load_state_dict(torch.load(f'C:/Users/gowri/Documents/B03/PytorchLSTM/Overnight optimization/{model_name}model_B0007.pt'))
    # PytorchLSTM\LSTM-CNNmodel.pt
    model.eval()
    print(f'Loaded model: {model_name}')
    return model

# def load_data_kfold(k, battery, which_model):
#     all_batteries = ['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032']
#     K_fold_batteries = all_batteries[:k]
#     normalized_test_data = load_data_normalise_cv(K_fold_batteries[battery], which_model)
#     rest_of_batteries = K_fold_batteries.drop(K_fold_batteries[battery])
#     normalized_train_data = load_data_normalise_cv(rest_of_batteries, which_model)
#     y = normalized_train_data["TTD"]
#     y_test = normalized_test_data["TTD"]
#     X = normalized_test_data.drop(columns=["TTD"])
#     X_test = normalized_test_data.drop(columns=["TTD"])
#     if which_model == 'hybrid':
#         X = X.drop(columns=["Voltage_measured"])
#         X_test = X_test.drop(columns=["Voltage_measured"])
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
#     return X_train, y_train, X_val, y_val, X_test, y_test

# def k_fold_plots(k_fold, model, which_model):
#     all_batteries = ['B0005', 'B0006', 'B0007', 'B0018', 'B0029', 'B0031', 'B0032']
#     k_fold_batteries = all_batteries[:k_fold]
#     predictions_test = []
    
#     for battery in k_fold_batteries:
#         X_train, y_train, X_val, y_val, X_test, y_test = load_data_kfold(k_fold, battery, which_model)
#         predictions_test.append(model(X_test))
    


# def scatter_plot(x, y, x_label, y_label):
#     plt.scatter(x, y)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.show()
#     return x, y, x_label, y_label

# def save_plot(plot_func, name):
#     x, y, x_label, y_label = plot_func
#     plt.savefig(f'{name}.png')
    

# def plot_loss(epoch, train_loss, val_loss):
#     plt.plot(epoch, train_loss, label='Training loss')
#     plt.plot(epoch, val_loss, label='Validation loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# def plot_prediction(prediction, y_actual):
#     plt.plot(prediction, label='Prediction')
#     plt.plot(y_actual, label='Actual')
#     plt.xlabel('Time')
#     plt.ylabel('Time to Discharge')
#     plt.legend()
#     plt.show()

# # read_model('hyrbidmodel')

# testing_hyperparameters = [0.00284, 7, 681, 4, [1, 3, 5, 8], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 20, 2, [1, 2, 4, 6, 1]] # best data driven
# testing_hyperparameters =  [0.00167, 8, 2000, 5, [1, 9, 18, 27, 36], [1, 5, 2.0, 7.0, 9.0], [1, 1, 1, 1, 1], [1, 1, 2, 3, 4], 14, 3, [24, 18, 12, 6, 1]] # best hybrid

# model = read_model('hybrid', testing_hyperparameters)