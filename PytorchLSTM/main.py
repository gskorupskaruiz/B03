import torch
import datetime
import pandas as pd

data_fields = {
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Current_charge', 'Voltage_charge', 'Time', 'Capacity'}

def load_data(battery): #read csv as pandas dataframe
	data = pd.read_csv("data/" + battery + ".csv") #ask Twan for name of csv with TTD
	data = data.drop(data[data.type != "discharge"].index) # only keep discharge data
	data = data.drop(["start_time", "type", "Sense_current", 'Battery_current', 'Current_ratio', 
		  'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct'], axis=1) # removes columns that are used for impedance, type and start time
	data = data.reset_index(drop = True) # reseting index so that previous row is always i-1 (so will not match up with rows of excel from now on)
	data["TTD"] = 0
	print(data)

	time_diff = data["Time"].diff() # make column of change in time
	discharge_time_index = time_diff[time_diff < 0].index -1
	discharge_time_index = list(discharge_time_index) + [len(data["Time"]) - 1] #find index + the last one

	for i1, i2 in zip([-1]+discharge_time_index, discharge_time_index):
		data["TTD"][i1+1:i2+1] = data["Time"][i2] - data["Time"][i1+1:i2+1]
	data.to_csv("data/" + battery + "_TTD.csv", index=False)
	#return data
def main():
	load_data("B0005")


if __name__ == '__main__': 
	main()
