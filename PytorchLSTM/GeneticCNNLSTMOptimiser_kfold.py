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
from desperate_kfold import *


# train evaluate (GA individuals)
def train_evaluate(ga_individual_solution):
    gene_length = 3
    # decode GA solution to get hyperparamteres
    ga_individual_solution = [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1] 
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


    # get rid of possibility of Kernel size being bigger than input size
    if cnn_kernel_size > cnn_output_size + 2* cnn_padding:
        cnn_kernel_size = cnn_output_size + 2* cnn_padding 
        print(f'cnn kernel size changed to {cnn_kernel_size} as it was bigger than the input size')


    # ensure lists are the correct length
    cnn_output_size = [cnn_output_size] * cnn_layers
    cnn_kernel_size = [cnn_kernel_size] * cnn_layers
    cnn_stride = [cnn_stride] * cnn_layers
    cnn_padding = [cnn_padding] * cnn_layers
    hidden_neurons_dense = [hidden_neurons_dense] * (cnn_layers)
    hidden_neurons_dense.append(1)
    hidden_neurons_dense[0] = lstm_sequential_length

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
    
    hyperparams_for_kfold = [120, 60, learning_rate*1000, lstm_sequential_length, batch_size, cnn_layers, cnn_output_size[0], cnn_kernel_size[0], cnn_stride[0], cnn_padding[0], lstm_neurons, lstm_layers, hidden_neurons_dense[1]]

    print('Current hyperparameters:', hyperparams_for_kfold)
    
    try:
        
        loss_model = run_model_cv(hyperparams_for_kfold, 'hybrid', 7)

    #    print(f"loss of model at  = {loss_model}")

    except TypeError or RuntimeError:
        print('Something went wrong, probably invalid set of hyper paremeters')
        loss_model = 100
    return [loss_model]

if __name__ == '__main__':  

    # init variables and implementation of Ga using DEAP 
    
    population_size = 5
    num_generations = 10
    entire_bit_array_length = 19 * 3 # 10 hyperparameters * 6 bits each  # make sure you change this in train_evaluate func too

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
    # mutati
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.6)
    # selection algorithm
    toolbox.register("select", tools.selTournament, tournsize=int(population_size/2))
    # evaluation fitness of individuals
    toolbox.register("evaluate", train_evaluate) # this train evaluate might not be allowed to have gene)length as input

    population = toolbox.population(n=population_size)
    r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1, ngen=num_generations, verbose=True)


    # print best solution found
    best_individuals = tools.selBest(population, k=5)
    print('Best ever individual = ', best_individuals[0], '\nFitness = ', best_individuals[0].fitness.values[0])
    print(f'list of individuals = {best_individuals}')