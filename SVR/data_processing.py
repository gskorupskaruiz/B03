import torch
import pandas as pd
from sklearn.model_selection import train_test_split

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
	return data


# def train_test_validation_split(X, y, test_size, cv_size):
#     """
#     TODO:
#     Part 0, Step 3: 
#         - Use the sklearn {train_test_split} function to split the dataset (and the labels) into
#             train, test and cross-validation sets
#     """
#     X_train, X_test_cv, y_train, y_test_cv = train_test_split(
#         X, y, test_size=test_size+cv_size, shuffle=True, random_state=0)

#     test_size = test_size/(test_size+cv_size)

#     X_cv, X_test, y_cv, y_test = train_test_split(
#         X_test_cv, y_test_cv, test_size=test_size, shuffle=True, random_state=0)

#     # return split data
#     return [X_train, y_train, X_test, y_test, X_cv, y_cv]


def split_data(data, test_size):
    #data = (data-data.mean())/data.std()
    y = data["TTD"]
    X = data.drop(["TTD"], axis=1)
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = True, random_state = 0)
    return X_train, X_test, y_train, y_test, X, y


if __name__ == '__main__': 
	find_TTD('B0005')
	find_TTD('B0006')
	find_TTD('B0007')
	find_TTD('B0018')