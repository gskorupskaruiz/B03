from desperate_kfold import run_model_cv
import numpy as np

import matplotlib.pyplot as plt

import numpy as np
import datetime

"""
THIS IS A RANDOM OPTIMIZER FOR THE LSTM MODEL

To run:
1. Specify hyperparameter limits
2. Specify number of iterations at each step
3. Specify hyperparameter limits after iteration

For k-fold cross validation:
1. Specify k in sketchymain.py
2. Specify _TTD or _TTD - with SOC in sketchymain.py
3.  - If you're running the hybrid model change the number of LSTM inputs in model.py (line 243) to 8
    - If you're just running LSTM-CNN, set it to 6

"""
    
    
class NormalOptimizer:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
  
    def initialize_particle(self):
        
        return (np.random.randint(self.lower_bound, self.upper_bound))
    
    def compute(self, fun):
        
        particle = self.initialize_particle()
        loss = fun(particle, 'hybrid', 4    , False)
        
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
        
        iter_step_up =   [3, 1, 8, 5, 500, 1, 1, 1, 1, 1, 1, 1, 5]
        iter_step_down = [8, 0, 0, 4, 500, 0, 0, 0, 0, 0, 0, 0, 5]
        
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
        

lower_limit = [60, 1, 1, 5, 100, 1, 1, 1, 1, 1, 1, 1, 15]
upper_limit = [120, 3, 100, 60, 1000, 8, 8, 8, 8, 8, 20, 3, 60]

opt = NormalOptimizer(lower_limit, upper_limit)
optimized = opt.run(run_model_cv, n_iter=3, n_init=3)

print(optimized)

# print('Finished at time:', datetime.datetime.now())