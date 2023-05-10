import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from main import *

import torch
from model import *

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

data_fields = {
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Current_charge', 'Voltage_charge', 'Time', 'Capacity'}

def train_test_validation_split(X, y, test_size, cv_size):
    "Splits data into train, test and cross validation sets"
    X_train, X_test_cv, y_train, y_test_cv = train_test_split(
        X, y, test_size=test_size+cv_size, shuffle=False, random_state=0)

    test_size = test_size/(test_size+cv_size)

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test_cv, y_test_cv, test_size=test_size, shuffle=False, random_state=0)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]

def load_data(battery, test_size, cv_size):
    data = [pd.read_csv("data/" + i + "_TTD.csv") for i in battery]
    data = pd.concat(data)
    y = data["TTD"]
    X = data.drop(["TTD"], axis=1)

    # normalize the data
    X = (X-X.mean(axis=0))/X.std(axis=0)
    y = (y-y.mean(axis=0))/y.std(axis=0)
    
    # split data
    X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(X, y, test_size, cv_size)



    return X_train, y_train, X_test, y_test, X_cv, y_cv

# Prepare data (reshape) (CHECK WHICH SHAPE NEEDED)
def prepare_dataset(seq_length, X_train, y_train, X_test, y_test, X_cv, y_cv):
    "Process data into the correct shape for the model for given hyper parameters"
    #print(type(X_cv_raw))
    # reshape data (Gowri's code from load gpu data with batches func in data_processing.py)
    x_tr = []
    y_tr = []
    for i in range(seq_length, len(X_train)):
        x_tr.append(X_train.values[i-seq_length:i])
        y_tr.append(y_train.values[i])
        
    # Convert to numpy arrays
    x_tr = torch.tensor(np.array(x_tr))
    y_tr = torch.tensor(y_tr).unsqueeze(1).unsqueeze(2)
    #print(y_tr.shape)

    x_v = []
    y_v = []
    for i in range(seq_length, len(X_cv)):
        x_v.append(X_cv.values[i-seq_length:i])
        y_v.append(y_cv.values[i])

    # Convert to numpy arrays
    x_v = torch.tensor(x_v)
    y_v = torch.tensor(y_v).unsqueeze(1).unsqueeze(2)

    x_t = []
    y_t = []
    for i in range(seq_length, len(X_test)):
        x_t.append(X_test.values[i-seq_length:i])
        y_t.append(y_test.values[i])

    # Convert to numpy arrays
    x_t = torch.tensor(x_t)
    y_t = torch.tensor(y_t).unsqueeze(1).unsqueeze(2)

    print(f'shape of x_t is {x_t.shape},  and shape of yt is {y_t.shape}')

    # gpu the data
    print(f"GPU available: {torch.cuda.is_available()}")

    if torch.cuda.is_available() == True:
        print('Running on GPU')
        X_train = x_tr.to('cuda').double()
        y_train = y_tr.to('cuda').double()
        X_test = x_t.to('cuda').double()
        y_test = y_t.to('cuda').double()
        X_cv = x_v.to('cuda').double()
        y_cv = y_v.to('cuda').double()
        # print("X_train and y_train are on GPU: ", X_train.is_cuda, y_train.is_cuda)
        # print("X_test and y_test are on GPU: ", X_test.is_cuda, y_test.is_cuda)
        # print("X_cv and y_cv are on GPU: ", X_cv.is_cuda, y_cv.is_cuda)
        # print(f"size of X_train: {X_train.size()} and y_train: {y_train.size()}")
                

    # if torch.cuda.is_available() == True:
    #     print("Running on GPU")
    #     X_train = torch.tensor(x_tr.values).to('cuda')
    #     y_train = torch.tensor(y_tr.values).to('cuda')
    #     X_test = torch.tensor(x_t.values).to('cuda')
    #     y_test = torch.tensor(y_t.values).to('cuda')
    #     X_cv = torch.tensor(x_v.values).to('cuda')
    #     y_cv = torch.tensor(y_v.values).to('cuda')
    else:
        print("THIS GA WILL TAKE A LONG TIME TO RUN ESPECIALLY WITHOUT THE GPU!!!")
    return X_train, y_train, X_test, y_test, X_cv, y_cv

# train evaluate (GA individuals)
def train_evaluate(ga_individual_solution):
    gene_length = 4
    import random
    # decode GA solution to get hyperparamteres
    lstm_layers_bit = BitArray(ga_individual_solution[0:gene_length]) # don't understand the bitarray stuff yet or the length given per hyperparameter
    lstm_neurons_bit = BitArray(ga_individual_solution[gene_length:2*gene_length])
    lstm_sequential_length_bit = BitArray(ga_individual_solution[2*gene_length:3*gene_length])
    learning_rate_bit = BitArray(ga_individual_solution[3*gene_length:4*gene_length])
    cnn_layers_bit = BitArray(ga_individual_solution[4*gene_length:5*gene_length])
    # cnn_kernel_size_bit = BitArray(ga_individual_solution[5*gene_length:6*gene_length])
    # cnn_stride_bit = BitArray(ga_individual_solution[6*gene_length:7*gene_length])
    # cnn_padding_bit = BitArray(ga_individual_solution[7*gene_length:8*gene_length])
    # cnn_output_size_bit = BitArray(ga_individual_solution[8*gene_length:9*gene_length])
    # hidden_neurons_dense_bit = BitArray(ga_individual_solution[9*gene_length:10*gene_length])
    batch_size_bit = BitArray(ga_individual_solution[10*gene_length:11*gene_length])

    lstm_layers = lstm_layers_bit.uint
    lstm_sequential_length = lstm_sequential_length_bit.uint
    lstm_neurons = lstm_neurons_bit.uint
    learning_rate = learning_rate_bit.uint
    cnn_layers = cnn_layers_bit.uint

    bit_kernel = []
    bit_stride = []
    bit_padding = []
    bit_output = []
    bit_hidden_dense = []

    for i in range(10):
        if i == cnn_layers-1:
            bit_kernel.append(BitArray(ga_individual_solution[(i+5)*gene_length:(6+i)*gene_length]).uint)
            bit_stride.append(BitArray(ga_individual_solution[(i+6)*gene_length:(i+7)*gene_length]).uint)
            bit_padding .append(BitArray(ga_individual_solution[(i+7)*gene_length:(i+8)*gene_length]).uint)
            bit_output.append(1)
            bit_hidden_dense.append(1)
        else:
            bit_kernel.append(BitArray(ga_individual_solution[(5+i)*gene_length:(6+i)*gene_length]).uint)
            bit_stride.append(BitArray(ga_individual_solution[(i+6)*gene_length:(i+7)*gene_length]).uint)
            bit_padding .append(BitArray(ga_individual_solution[(i+7)*gene_length:(i+8)*gene_length]).uint)
            bit_output.append(BitArray(ga_individual_solution[(i+8)*gene_length:(i+9)*gene_length]).uint)
            bit_hidden_dense.append(BitArray(ga_individual_solution[(i+9)*gene_length:(i+10)*gene_length]).uint)
           
    # cnn_kernel_size = cnn_kernel_size_bit.uint
    # cnn_stride = cnn_stride_bit.uint
    # cnn_padding = cnn_padding_bit.uint
    # cnn_output_size = cnn_output_size_bit.uint
    # hidden_neurons_dense = hidden_neurons_dense_bit.uint
    batch_size = batch_size_bit.uint

    # resize hyperparameters
    learning_rate = learning_rate/100
    batch_size = batch_size * 10

    # ensure lists are the correct length
    # cnn_output_size = [cnn_output_size] * (cnn_layers+1)
    # cnn_kernel_size = [cnn_kernel_size] * (cnn_layers+1)
    # cnn_stride = [cnn_stride] * (cnn_layers+1)
    # cnn_padding = [cnn_padding] * (cnn_layers+1)
    # hidden_neurons_dense = [hidden_neurons_dense] * (cnn_layers+1)

    cnn_output_size = bit_output
    cnn_kernel_size = bit_kernel
    cnn_stride = bit_stride
    cnn_padding = bit_padding
    hidden_neurons_dense = bit_hidden_dense

    batch_size += 1
    cnn_layers += 1
    lstm_layers += 1


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


    # Return 100 fitness if any hyperparameter == 0
    if batch_size == 0 or lstm_layers == 0 or lstm_sequential_length == 0 or lstm_neurons == 0 or learning_rate == 0 or batch_size == 0 or cnn_layers == 0 or cnn_kernel_size == 0 or cnn_stride == 0 or cnn_padding == 0 or hidden_neurons_dense == 0:
        print("One of the hyperparameters is 0 - try again haha")
        return 100

    else:    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # change data so that seq len and batch size is changed (use prepare_dataset func
        X_train, y_train, X_test, y_test, X_cv, y_cv = prepare_dataset(lstm_sequential_length, X_train_raw, y_train_raw, X_test_raw, y_test_raw, X_cv_raw, y_cv_raw)
        #print(f'shape of x_train {X_train.shape}')
        dataset = SeqDataset(x_data=X_train, y_data = y_train, seq_len = lstm_sequential_length, batch=batch_size)
        datasetv = SeqDataset(x_data=X_cv, y_data=y_cv, seq_len=lstm_sequential_length, batch=batch_size)
        
        
        # intitialize the model based on the new hyperparameters
        model = ParametricCNNLSTM(cnn_layers, cnn_output_size, cnn_kernel_size, cnn_stride, cnn_padding, lstm_neurons, lstm_layers, hidden_neurons_dense, lstm_sequential_length).double()

        model.to(device)
        # train model 
        model.train()
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        num_epochs = 20
        
        train_hist, val_hist = trainbatch(model, dataset, datasetv, num_epochs, lf = criterion, optimizer = optimizer, verbose = True)
        model.eval()
        predictions = model(X_test).to('cpu').detach().numpy()

        plot = True
        if plot != False:
            epoch = np.linspace(1, n_epoch+1, n_epoch)
            plt.plot(epoch, predictions.squeeze(2), label='predictions')
            plt.plot(y_test.squeeze(2).to('cpu').detach().numpy(), label='actual')
            plt.legend()
            plt.show()

        # evaluate model
        loss_model = ((predictions - y_test.squeeze(2).to('cpu').detach().numpy()) ** 2).mean()


        print(f"loss of model = {loss_model}")

        return loss_model

if __name__ == '__main__':  

    # init variables and implementation of Ga using DEAP 
    battery = ["B0005"]
    population_size = 4
    num_generations = 4
    entire_bit_array_length = 19 * 4 # 10 hyperparameters * 6 bits each  # make sure you change this in train_evaluate func too
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, X_cv_raw, y_cv_raw = load_data(battery, test_size=0.2, cv_size=0.2)
    input_size = len(data_fields) - 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # basically creates classes for the fitness and individual
    creator.create('FitnessMax', base.Fitness, weights=[-1.0])
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # create toolbox & initialize population (bernoulli random variables)
    toolbox = base.Toolbox()
    toolbox.register("binary", bernoulli.rvs, 0.5) 
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n=entire_bit_array_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ordered cross over for mating
    toolbox.register("mate", tools.cxOrdered)
    # mutations
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.6)
    # selection algorithm
    toolbox.register("select", tools.selRoulette)
    # evaluation fitness of individuals
    toolbox.register("evaluate", train_evaluate) # this train evaluate might not be allowed to have gene)length as input

    population = toolbox.population(n=population_size)
    r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1, ngen=num_generations, verbose=True)


    # print best solution found
    best_individuals = tools.selBest(population, k=1)
    print(f"\nBest individual is {best_individuals[0]}, \nwith fitness {best_individuals[0].fitness}")