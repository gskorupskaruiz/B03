import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

data_fields = {
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Current_charge', 'Voltage_charge', 'Time', 'Capacity'}

def find_TTD(battery): #read csv as pandas dataframe
	data = pd.read_csv("data/" + battery + ".csv") #ask Twan for name of csv with TTD
	data = data.drop(data[data.type != "discharge"].index) # only keep discharge data
	data = data.drop(["start_time", "type", "Sense_current", 'Battery_current', 'Current_ratio', 
		  'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct', 'ambient_temp'], axis=1) # removes columns that are used for impedance, type and start time
	data = data.reset_index(drop = True) # resetting index so that previous row is always i-1 (so will not match up with rows of excel from now on)
	data["TTD"] = 0
	print(data)

	time_diff = data["Time"].diff() # make column of change in time
	discharge_time_index = time_diff[time_diff < 0].index -1
	discharge_time_index = list(discharge_time_index) + [len(data["Time"]) - 1] #find index + the last one

	for i1, i2 in zip([-1]+discharge_time_index, discharge_time_index):
		data["TTD"][i1+1:i2+1] = data["Time"][i2] - data["Time"][i1+1:i2+1] # calculate TTD for each row
	data.to_csv("data/" + battery + "_TTD.csv", index=False) # save to csv
	#return data


def load_data(battery):
	data = pd.read_csv("data/" + battery + "_TTD.csv")
	print("Battery: ", battery)
	print(data)
	return data


def train_test_validation_split(X, y, test_size, cv_size):
    """
    TODO:
    Part 0, Step 3: 
        - Use the sklearn {train_test_split} function to split the dataset (and the labels) into
            train, test and cross-validation sets
    """
    X_train, X_test_cv, y_train, y_test_cv = train_test_split(
        X, y, test_size=test_size+cv_size, shuffle=False, random_state=0)

    test_size = test_size/(test_size+cv_size)

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test_cv, y_test_cv, test_size=test_size, shuffle=False, random_state=0)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]


def load_gpu_data_with_batches(data, test_size, cv_size, seq_length):
	y = data["TTD"][:50000]
	X = data[:50000].drop(["TTD"], axis=1)
	X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(X, y, test_size, cv_size)

	print(X_train.shape, X_test.shape, X_cv.shape)

	# Create sliding windows of length seq_len for xtrain and ytrain
	x_tr = []
	y_tr = []
	for i in range(seq_length, len(X_train)):
		x_tr.append(X_train.values[i-seq_length:i])
		y_tr.append(y_train.values[i])
		
	# Convert to numpy arrays
	x_tr = torch.tensor(np.array(x_tr))
	y_tr = torch.tensor(y_tr).unsqueeze(1).unsqueeze(2)
	print(y_tr.shape)

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


	# go to gpu, "google gpu pytorch python"
	print("GPU is availible: ", torch.cuda.is_available())
	if torch.cuda.is_available() == True:
		print('Running on GPU')

		X_train = x_tr.to('cuda').double()
		y_train = y_tr.to('cuda').double()
		X_test = x_t.to('cuda').double()
		y_test = y_t.to('cuda').double()
		X_cv = x_v.to('cuda').double()
		y_cv = y_v.to('cuda').double()
		print("X_train and y_train are on GPU: ", X_train.is_cuda, y_train.is_cuda)
		print("X_test and y_test are on GPU: ", X_test.is_cuda, y_test.is_cuda)
		print("X_cv and y_cv are on GPU: ", X_cv.is_cuda, y_cv.is_cuda)
		print(f"size of X_train: {X_train.size()} and y_train: {y_train.size()}")
	
	return x_tr, y_tr, x_t, y_t, x_v, y_v


def load_gpu_data(data, test_size, cv_size, seq_length):
	y = data["TTD"][:50000]
	X = data[:50000].drop(["TTD"], axis=1)
	X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(X, y, test_size, cv_size)

	print(X_train.shape, X_test.shape, X_cv.shape)

	lex = len(X_train)
	lex = lex/seq_length
	X_train = torch.tensor(X_train.values).reshape(int(lex), seq_length, len(data_fields)) # changed the reshaping of this 
	y_train = torch.tensor(y_train.values).view(int(lex), seq_length, 1)

	lexxx = len(X_test)
	lexxx = lexxx/seq_length
	X_test = torch.tensor(X_test.values).reshape(int(lexxx), seq_length, len(data_fields))
	y_test = torch.tensor(y_test.values).reshape(int(lexxx), seq_length, 1)
	
	lexx = len(X_cv)
	lexx = lexx/seq_length

	X_cv = torch.tensor(X_cv.values).reshape(int(lexx), seq_length, len(data_fields))
	y_cv = torch.tensor(y_cv.values).view(int(lexx), seq_length, 1)

	# go to gpu, "google gpu pytorch python"
	print("GPU is availible: ", torch.cuda.is_available())
	if torch.cuda.is_available() == True:
		print('Running on GPU')

		X_train = X_train.to('cuda').double()
		y_train = y_train.to('cuda').double()
		X_test = X_test.to('cuda').double()
		y_test = y_test.to('cuda').double()
		X_cv = X_cv.to('cuda').double()
		y_cv = y_cv.to('cuda').double()
		print("X_train and y_train are on GPU: ", X_train.is_cuda, y_train.is_cuda)
		print("X_test and y_test are on GPU: ", X_test.is_cuda, y_test.is_cuda)
		print("X_cv and y_cv are on GPU: ", X_cv.is_cuda, y_cv.is_cuda)
		print(f"size of X_train: {X_train.size()} and y_train: {y_train.size()}")
	
	return X_train, y_train, X_test, y_test, X_cv, y_cv


# if __name__ == '__main__': 
# 	find_TTD('B0005')
# 	find_TTD('B0006')
# 	find_TTD('B0007')
# 	find_TTD('B0018')