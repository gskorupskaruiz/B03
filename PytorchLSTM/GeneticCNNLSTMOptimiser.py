import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
# not sure if i need the following imports as they are used in the notebook to generate the model
# from keras.layers import LSTM, Input, Dense
# from keras.models import Model

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

    # gpu the data
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() == True:
        print("Running on GPU")
        X_train = torch.tensor(X_train.values).to('cuda')
        y_train = torch.tensor(y_train.values).to('cuda')
        X_test = torch.tensor(X_test.values).to('cuda')
        y_test = torch.tensor(y_test.values).to('cuda')
        X_cv = torch.tensor(X_cv.values).to('cuda')
        y_cv = torch.tensor(y_cv.values).to('cuda')
    else:
        print("THIS GA WILL TAKE A LONG TIME TO RUN ESPECIALLY WITHOUT THE GPU!!!")

    return X_train, y_train, X_test, y_test, X_cv, y_cv

# Prepare data (reshape) (CHECK WHICH SHAPE NEEDED)
def prepare_dataset(seq_length, X_train, y_train, X_test, y_test, X_cv, y_cv):
    "Process data into the correct shape for the model for given hyper parameters"
    
    # reshape data (Gowri's code from load gpu data func in data_processing.py)
    lex = len(X_train)
    lex = lex/seq_length
    X_train = X_train.reshape(int(lex), seq_length, len(data_fields)) # changed the reshaping of this 
    y_train = y_train.view(int(lex), seq_length, 1) # gowri, any reason why this is .view and the rest ,reshape?
    lexxx = len(X_test)
    lexxx = lexxx/seq_length
    X_test = X_test.reshape(int(lexxx), seq_length, len(data_fields))
    y_test = y_test.reshape(int(lexxx), seq_length, 1)
    lexx = len(X_cv)
    lexx = lexx/seq_length
    X_cv = X_cv.reshape(int(lexx), seq_length, len(data_fields))
    y_cv = y_cv.view(int(lexx), seq_length, 1)
    

    return X_train, y_train, X_test, y_test, X_cv, y_cv

# train evaluate (GA individuals)
def train_evaluate(ga_individual_solution, gene_length):

    # decode GA solution to get hyperparamteres
    lstm_layers_bit = BitArray(ga_individual_solution[0:gene_length]) # don't understand the bitarray stuff yet or the length given per hyperparameter
    lstm_neurons_bit = BitArray(ga_individual_solution[gene_length:2*gene_length])
    lstm_sequential_length_bit = BitArray(ga_individual_solution[gene_length:2*gene_length])
    learning_rate_bit = BitArray(ga_individual_solution[2*gene_length:3*gene_length])
    cnn_layers_bit = BitArray(ga_individual_solution[4*gene_length:5*gene_length])
    cnn_kernel_size_bit = BitArray(ga_individual_solution[5*gene_length:6*gene_length])
    cnn_stride_bit = BitArray(ga_individual_solution[6*gene_length:7*gene_length])
    cnn_padding_bit = BitArray(ga_individual_solution[7*gene_length:8*gene_length])
    cnn_output_size_bit = BitArray(ga_individual_solution[8*gene_length:9*gene_length])

    lstm_layers = lstm_layers_bit.uint
    lstm_sequential_length = lstm_sequential_length_bit.uint
    lstm_neurons = lstm_neurons_bit.uint
    learning_rate = learning_rate_bit.uint
    cnn_layers = cnn_layers_bit.uint
    cnn_kernel_size = cnn_kernel_size_bit.uint
    cnn_stride = cnn_stride_bit.uint
    cnn_padding = cnn_padding_bit.uint
    cnn_output_size = cnn_output_size_bit.uint

    print(f"lstm Layers =  {lstm_layers}")
    print(f"lstm Sequential Length =  {lstm_sequential_length}")
    print(f"lstm Neurons =  {lstm_neurons}")
    print(f"learning rate =  {learning_rate}")
    print(f"cnn layers =  {cnn_layers}")
    print(f"cnn kernel size =  {cnn_kernel_size}")
    print(f"cnn stride =  {cnn_stride}")
    print(f"cnn padding =  {cnn_padding}")
    print(f"cnn neurons =  {cnn_output_size}")


    # Return 100 fitness if any hyperparameter == 0
    if lstm_layers == 0 or lstm_sequential_length == 0 or lstm_neurons == 0 or learning_rate == 0 or batch_size == 0 or cnn_layers == 0 or cnn_kernel_size == 0 or cnn_stride == 0 or cnn_padding == 0 or cnn_neurons == 0:
        return 100
    
    # change data so that seq len and batch size is changed (use prepare_dataset func
    X_train, y_train, X_test, y_test, X_cv, y_cv = prepare_dataset(lstm_sequential_length)
    # intitialize the model based on the new hyperparameters
    model = ParametricCNNLSTM(input)
    # train model

    y_true, y_cv_pred = 0 # for now

    # evaluate model
    mse_model = mse(y_true, y_cv_pred)
    print(f"Validation MSE of model = {mse_model}")

    return mse_model

if __name__ == '__main__':  

    # init variables and implementation of Ga using DEAP 
    battery = ["B0005"]
    population_size = 4
    num_generations = 4
    gene_length = 6 # not sure if this is correct (depends on how many hyperparameters we want to optimize or on the range of values for each hyperparameter)
    X_train, y_train, X_test, y_test, X_cv, y_cv = load_data(battery, test_size=0.2, cv_size=0.2)
    input_size = len(data_fields)

    # basically creates classes for the fitness and individual
    creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # create toolbox & initialize population (bernoulli random variables)
    toolbox = base.Toolbox()
    toolbox.register*("binary", bernoulli.rvs, 0.5) 
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ordered cross over for mating
    toolbox.register("mate", tools.cxOrdered)
    # mutations
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.6)
    # selection algorithm
    toolbox.register("select", tools.selRoulette)
    # evaluation fitness of individuals
    toolbox.register("evaluate", train_evaluate(gene_length = gene_length)) # this train evaluate might not be allowed to have gene)length as input

    population = toolbox.population(n=population_size)
    r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1, ngen=num_generations, verbose=False)


    # print best solution found
    best_individuals = tools.selBest(population, k=1)
    print(f"\nBest individual is {best_individuals[0]}, \nwith fitness {best_individuals[0].fitness}")