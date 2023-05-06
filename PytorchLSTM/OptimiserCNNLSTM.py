import pyswarms
import numpy
from sketchymain import run_model

import config
import matplotlib.pyplot as plt

#I USE PSO IN ORDER TO OPTIMIZE TWO PARAMETERS:
    #NUMBER OF FILTERS USED ON CONVOLUTION LAYERS
    #NUMBER OF EPOCHS USED IN TRAINING PROCESS

def optimizeCNNLSTM(particleDimensions):

    '''
    This is loss function applied by all particles in iterations
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param particleDimensions: numpy array with dimensions of a n particle --> 2 dimensions (filters, epochs)
    :param stride: by default=1, integer represents stride length of convolution
    :return: loss --> result of application of loss equation
    '''

    try:

        #RETRIEVE DIMENSIONS OF PARTICLE
    

        #CALL CNN FUNCTION cnn --> RETURN accuracy
        
        #APPLY LOST FUNCTION --> THE MAIN OBJECTIVE IS TO MINIMIZE LOSS --> MAXIMIZE ACCURACY AND AT SAME TIME MINIMIZE THE NUMBER OF EPOCHS
                                #AND FILTERS, TO REDUCE TIME AND COMPUTACIONAL POWER
        loss = run_model(n_hidden=particleDimensions[0], n_layer=particleDimensions[1], n_epoch=particleDimensions[2], lr=particleDimensions[3], test_size=particleDimensions[4], cv_size=particleDimensions[5], seq=particleDimensions[6])
       
        return loss

    except:
        raise

def particleIteration(particles):

    '''
    This is function that calls loss function, and returns all losses return by all particles on one iteration
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :param particles: numpy array --> (particles, dimensions)
    :return: lossArray --> all losses returned by all particles
    '''

    try:

        numberParticles = particles.shape[0]
        allLosses = [optimizeCNNLSTM(particleDimensions=particles[i])for i in range(numberParticles)]

        return allLosses #NEED TO RETURN THIS PYSWARMS NEED THIS

    except:
        raise

def callCNNOptimization(numberParticles,bounds, **kwargs):

    '''
    This is the function that defines all PSO context and calls loss function for every particles (fill all iterations)
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :param numberParticles: integer --> number of particles of swarm
    :param iterations: integer --> number of iterations
    :param bounds: numpy array (minBound, maxBound) --> minBound: numpyArray - shape(dimensions), maxBound: numpyArray - shape(dimensions)
    :return cost: integer --> minimum loss
    :return pos: numpy array with n dimensions --> [filterValue, epochValue], with best cost (minimum cost)
    :return optimizer: SWARM Optimization Optimizer USED IN DEFINITION AND OPTIMIZATION OF PSO
    '''

    try:

        #GET PSO PARAMETERS
        psoType = kwargs.get(config.TYPE)
        options = kwargs.get(config.OPTIONS)

        #DIMENSIONS OF PROBLEM
        dimensions = 7

        #OPTIMIZER FUNCTION
        if psoType == config.GLOBAL_BEST:
            optimizer = pyswarms.single.GlobalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                     options=options, bounds=bounds)
        elif psoType == config.LOCAL_BEST:
            optimizer = pyswarms.single.LocalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                     options=options, bounds=bounds)
        else:
            raise AttributeError
        #GET BEST COST AND PARTICLE POSITION
        cost, pos = optimizer.optimize(objective_func=particleIteration(numberParticles ))

        return cost, pos, optimizer

    except:
        raise
print(callCNNOptimization(50, 3))