import torch
from torch import nn
import numpy as np
import pandas as pd
import csv
import random

#PREPROCESSING

with open('PytorchLSTM/4e_hyper.txt', 'r') as f:
    data = f.readlines()
all_data = len(data)
test_size = int(all_data*0.2)



for i in range(len(data)):
    data[i] = data[i].replace('[','')               # Please ignore my total lack of string processing skills
    data[i] = data[i].replace(']','')
    data[i] = data[i].replace('\n','')
    data[i] = data[i].replace('     ', '\t')
    data[i] = data[i].replace('    ', '\t')
    data[i] = data[i].replace('   ', '\t')
    data[i] = data[i].replace('  ', '\t')
    data[i] = data[i].replace(' ', '\t')
    data[i] = data[i].split('\t')
    data[i] = data[i][1:]
    
    data[i] = [float(j) for j in data[i]]
data = random.sample(data, len(data))

data = np.array(data, dtype = np.float32)
train = data[:all_data-test_size]
test = data[all_data-test_size:]

    
X_train = torch.tensor(train[:,:-1])
X_test = torch.tensor(test[:,:-1])
y_train = torch.tensor(train[:,-1]).unsqueeze(1)
y_test = torch.tensor(test[:,-1]).unsqueeze(1)


X_train = (X_train - torch.mean(X_train))/torch.std(X_train)
X_test = (X_test - torch.mean(X_test))/torch.std(X_test)
y_train = (y_train - torch.mean(y_train))/torch.std(y_train)
y_test = (y_test - torch.mean(y_test))/torch.std(y_test)




#MODEL

def train(lr, n_hidden1, n_hidden2, n_epochs):

    model = nn.Sequential(nn.Linear(13, n_hidden1),nn.Sigmoid(),nn.Linear(n_hidden1, n_hidden2), nn.Sigmoid(),nn.Linear(n_hidden2, 1))

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.MSELoss()

    #TRAINING

    for epoch in range(n_epochs):

        optimizer.zero_grad()
        y_pred = model(X_train)
        #y_train = y_train.to(torch.float32)
        loss = loss_fn(y_pred, y_train)
        
        #print('Loss at epoch', epoch, ':', loss)
        loss.backward()
        loss = loss.item()
        
        optimizer.step()
    
    model.eval()
    y_val = model(X_test)
    loss_val = loss_fn(y_val, y_test)
    loss_val = loss_val.item()
    print(y_val)
    return loss_val, model


#HYPERPARAMETER OPTIMISATION

lrs = np.linspace(0.001, 0.201, 15)
n_hidden1 = np.arange(20, 100, 11)
n_hidden2 = np.arange(20, 100, 11)
n_epochs = np.arange(15, 51, 10)

losses = []

for lr in lrs:
    for n in n_hidden1:
        for m in n_epochs:
            for i in n_hidden2:
                loss, model = train(lr, n, i, m)
                print(loss, lr, n, i, m)
                iter = [loss, lr, n, i, m]
                losses.append(iter)
        
losses = np.array(losses)

min_loss = min(losses[:,0])
min_loss_index = np.where(losses[:,0] == min_loss)
best_hyper = losses[min_loss_index][0]

lr_opt = float(best_hyper[1])
n_hidden1_opt = int(best_hyper[2])
n_hidden2_opt = int(best_hyper[3])
n_epochs_opt = int(best_hyper[4])

model_trained = train(lr_opt, n_hidden1_opt, n_hidden2_opt, n_epochs_opt)[1]
print('Best loss and hyperparameters', best_hyper)
y_val = model_trained(X_test)
print(y_val, y_test)


#print('Test loss:', float(nn.MSELoss(y_val, y_test)))

#VALIDATION
#lr_opt = float(best_hyper[1])
#print(lr_opt)
#n_hidden_opt = int(best_hyper[2])
#print(n_hidden_opt)
#model_trained = train(lr_opt, n_hidden_opt, n_epochs)[1]

#y_val = model_trained(X_test)
#print(y_val)
#print('Test loss:', float(nn.MSELoss(y_val, y_test)))

