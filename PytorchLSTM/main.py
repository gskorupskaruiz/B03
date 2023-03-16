import torch
import datetime
import pandas as pd

data_fields = {
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Current_charge', 'Voltage_charge', 'Time', 'Capacity'}

def load_data(battery): #read csv save a torch tensors
	data = pd.read_csv("data/" + battery + ".csv") #ask Twan for name of csv with TTD
	print(data)
	print(data.dtypes)
	data = data.drop(data[data.type != "discharge"].index) # only keep discharge data
	data = data.drop(["start_time", "type", "Sense_current", 'Battery_current', 'Current_ratio', 
		  'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct'], axis=1) # removes columns that are used for impedance, type and start time
	print(data.dtypes)	
	print(data)


def main():
	load_data("B0005")


if __name__ == '__main__': 
	main()
