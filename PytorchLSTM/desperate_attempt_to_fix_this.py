import numpy as np
import torch
from desperate_kfold import *

# generate hyperparameters
#lr, seq, batch_size, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense 


# TO OPTIMISE: lr, seq, batch size, n_layers_conv, hidden_size lstm, n_layers_lstm
losses = []
lrs = np.arange(0.0001, 0.01, 0.001)
print(lrs)
a = 1
for lr in lrs:
    print(lr)
    print('Nr iter:', a, 'out of', len(lrs))
    hyperparams = [lr, 15, 500, 3, [1, 3, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], 40, 2, [1, 8, 8, 1]]
    loss = run_model_cv(hyperparams, 'hybrid', 4, save_for_plots = False)
    losses.append([loss, lr])
    print(losses)
    a += 1
    
print(losses)

# seqs = np.linspace(5, 30, )
# batch_sizes = np.linspace(100, 1000, 3)
# n_layers_convs = np.arange(1, 6)
# hidden_size_lstms = np.linspace(1, 50, 5)
# n_layers_lstms = np.arange(1, 6)
