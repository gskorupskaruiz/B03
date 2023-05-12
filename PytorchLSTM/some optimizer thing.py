import pyswarms
import numpy
from sketchymain import run_model

import config
import matplotlib.pyplot as plt

import copy
import numpy as np
import datetime

class Particle(object):
    """Particle class for PSO

    This class encapsulates the behavior of each particle in PSO and provides
    an efficient way to do bookkeeping about the state of the swarm in any given
    iteration.

    Args:
        lower_bound (np.array): Vector of lower boundaries for particle dimensions.
        upper_bound (np.array): Vector of upper boundaries for particle dimensions.
        dimensions (int): Number of dimensions of the search space.
        objective function (function): Black-box function to evaluate.

    """
    
    def __init__(self,
                 lower_bound,
                 upper_bound,
                 dimensions,
                 objective_function):
        
        self.reset(dimensions, lower_bound, upper_bound, objective_function)
        
        
    def reset(self,
              dimensions,
              lower_bound,
              upper_bound,
              objective_function):
        """Particle reset

        Allows for reset of a particle without reallocation.

		Args:
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			dimensions (int): Number of dimensions of the search space.

        """
    
        position = []
        for i in range(dimensions):
            if lower_bound[i] < upper_bound[i]:
                position.extend(np.random.randint(lower_bound[i], upper_bound[i] + 1, 1, dtype=int))
            elif lower_bound[i] == upper_bound[i]:
                position.extend(np.array([lower_bound[i]], dtype=int))
            else:
                assert False

        self.position = [position]

        self.velocity = [np.multiply(np.random.rand(dimensions),
                                     (upper_bound - lower_bound)).astype(int)]

        self.best_position = self.position[:]

        self.function_value = [objective_function(self.best_position[-1])]
        self.best_function_value = self.function_value[:]

    def update_velocity(self, omega, phip, phig, best_swarm_position):
        """Particle velocity update

		Args:
			omega (float): Velocity equation constant.
			phip (float): Velocity equation constant.
			phig (float): Velocity equation constant.
			best_swarm_position (np.array): Best particle position.

        """
        random_coefficient_p = np.random.uniform(size=np.asarray(self.position[-1]).shape)
        random_coefficient_g = np.random.uniform(size=np.asarray(self.position[-1]).shape)

        self.velocity.append(omega
                             * np.asarray(self.velocity[-1])
                             + phip
                             * random_coefficient_p
                             * (np.asarray(self.best_position[-1])
                                - np.asarray(self.position[-1]))
                             + phig
                             * random_coefficient_g
                             * (np.asarray(best_swarm_position)
                                - np.asarray(self.position[-1])))

        self.velocity[-1] = self.velocity[-1].astype(int)

    def update_position(self, lower_bound, upper_bound, objective_function):
        """Particle position update

		Args:
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			objective function (function): Black-box function to evaluate.

        """
        new_position = self.position[-1] + self.velocity[-1]

        if np.array_equal(self.position[-1], new_position):
            self.function_value.append(self.function_value[-1])
        else:
            mark1 = new_position < lower_bound
            mark2 = new_position > upper_bound

            new_position[mark1] = lower_bound[mark1]
            new_position[mark2] = upper_bound[mark2]

            self.function_value.append(objective_function(self.position[-1]))

        self.position.append(new_position.tolist())

        if self.function_value[-1] < self.best_function_value[-1]:
            self.best_position.append(self.position[-1][:])
            self.best_function_value.append(self.function_value[-1])
            

class Pso(object):
    """PSO wrapper

    This class contains the particles and provides an abstraction to hold all the context
    of the PSO algorithm

    Args:
        swarmsize (int): Number of particles in the swarm
        maxiter (int): Maximum number of generations the swarm will run

    """
    def __init__(self, swarmsize=1, maxiter=4):
        self.max_generations = maxiter
        self.swarmsize = swarmsize
        self.total_runs = self.swarmsize * (self.max_generations + 1)
        print('***PSO initialised***')
        self.omega = 0.9
        self.phip = 0.3
        self.phig = 0.3

        self.minstep = 1e-4
        self.minfunc = 1e-4

        self.best_position = [None]
        self.best_function_value = [1]

        self.particles = []

        self.retired_particles = []

        
    def run(self, function, lower_bound, upper_bound, kwargs=None):
        """Perform a particle swarm optimization (PSO)

		Args:
			objective_function (function): The function to be minimized.
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.

		Returns:
			best_position (np.array): Best known position
			accuracy (float): Objective value at best_position
			:param kwargs:

        """
    
        
        if kwargs is None:
            kwargs = {}

        objective_function = lambda x: function(x, **kwargs)
        assert hasattr(function, '__call__'), 'Invalid function handle'

        assert len(lower_bound) == len(upper_bound), 'Invalid bounds length'

        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)

        assert np.all(upper_bound > lower_bound), 'Invalid boundary values'


        dimensions = len(lower_bound)

        self.particles = self.initialize_particles(lower_bound,
                                                   upper_bound,
                                                   dimensions,
                                                   objective_function)

        # Start evolution
        generation = 1
        while generation <= self.max_generations:
            
            print('GENERATION: ', generation+1)
            
            if generation == self.max_generations: print('Finalising... (last run)')
            n = 0
            for particle in self.particles:
                n += 1
                print(' (Particle ', n, ')')
                particle.update_velocity(self.omega, self.phip, self.phig, self.best_position[-1])
                particle.update_position(lower_bound, upper_bound, objective_function)

                if particle.best_function_value[-1] == 0:
                    self.retired_particles.append(copy.deepcopy(particle))
                    particle.reset(dimensions, lower_bound, upper_bound, objective_function)
                elif particle.best_function_value[-1] < self.best_function_value[-1]:
                    stepsize = np.sqrt(np.sum((np.asarray(self.best_position[-1])
                                               - np.asarray(particle.position[-1])) ** 2))

                    if np.abs(np.asarray(self.best_function_value[-1])
                              - np.asarray(particle.best_function_value[-1])) \
                            <= self.minfunc:
                        return particle.best_position[-1], particle.best_function_value[-1]
                    elif stepsize <= self.minstep:
                        return particle.best_position[-1], particle.best_function_value[-1]
                    else:
                        self.best_function_value.append(particle.best_function_value[-1])
                        self.best_position.append(particle.best_position[-1][:])
                print('----> Progress: ', round(n * generation*100/self.total_runs), '%')


            generation += 1
        
        return self.best_position[-1], self.best_function_value[-1]
    
    def initialize_particles(self,
                             lower_bound,
                             upper_bound,
                             dimensions,
                             objective_function):
        """Initializes the particles for the swarm

		Args:
			objective_function (function): The function to be minimized.
			lower_bound (np.array): Vector of lower boundaries for particle dimensions.
			upper_bound (np.array): Vector of upper boundaries for particle dimensions.
			dimensions (int): Number of dimensions of the search space.

		Returns:
			particles (list): Collection or particles in the swarm

        """
        particles = []
        for _ in range(self.swarmsize):
            particles.append(Particle(lower_bound,
                                      upper_bound,
                                      dimensions,
                                      objective_function))
            
            if particles[-1].best_function_value[-1] < self.best_function_value[-1]:
                self.best_function_value.append(particles[-1].best_function_value[-1])
                self.best_position.append(particles[-1].best_position[-1])


        self.best_position = [self.best_position[-1]]
        self.best_function_value = [self.best_function_value[-1]]

        return particles
    
    
class NormalOptimizer:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
  
    def initialize_particle(self):
        
        return (np.random.randint(self.lower_bound, self.upper_bound))
    
    def compute(self, fun):
        
        particle = self.initialize_particle()
        loss = fun(particle)
        
        return loss, particle
        
    def initial_run(self, fun, n_init):
        initial_results = []
        
        for i in range(n_init):
            initial_results.append(self.compute(fun))
        return initial_results
    
    def best_option(self, fun, n_init):
        
        initiation = np.array(self.initial_run(fun, n_init))
        losses = list(initiation.T[0])
        params = list(initiation.T[1])
        min_loss = min(losses)
        
        
        starting_hyperparams = params[losses.index(min_loss)]
        
        return starting_hyperparams, min_loss
    
    def run(self, fun, n_iter, n_init):
        
        best_params, best_loss = self.best_option(fun, n_init)
        
        iter_step_up = [3, 1, 8, 5, 1, 1, 1, 1, 1, 1, 1, 1, 5]
        iter_step_down = [3, 0, 8, 5, 0, 0, 1, 1, 1, 1, 1, 0, 5]
        
        self.lower_bound = best_params - iter_step_down
        self.upper_bound = best_params + iter_step_up
        
        first_iter = self.best_option(fun, n_iter)
        
        min_iter = np.array(first_iter).T[1]
        min_iter_params = np.array(first_iter).T[0]
        
        if min_iter < best_loss:
            return min_iter_params, min_iter
        else:
            print('Rerunning...')
            print('Current best: ', best_params, best_loss)
            self.rerun(fun, n_iter, n_init)
    
    def rerun(self, fun, n_iter, n_init):
        self.run(fun, n_iter, n_init)
        
# pso = Pso(swarmsize=1,maxiter=4)
# n_hidden, n_layer, lr, seq, batch_size, num_layers_conv, output_channels_val, kernel_sizes_val, stride_sizes_val, padding_sizes_val, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense_val
# 
lower_limit = [20, 2, 1, 10, 1, 1, 1, 1, 1, 1, 1, 2, 10]
upper_limit = [120, 5, 900, 60, 10, 4, 8, 8, 8, 8, 8, 5, 120]

opt = NormalOptimizer(lower_limit, upper_limit)
optimized = opt.run(run_model, n_iter=5, n_init=100)

print(optimized)
# bp,value = pso.run(run_model,lower_limit, upper_limit)
# # n_hidden, n_layer, n_epoch, lr, test_size, cv_size, seq
# print('DONEEEEEEEEEEEEEEE')
# print(bp, value)
# print(datetime.datetime.now())