import torch
import datetime
import pandas as pd
from scipy.io import loadcsv


def load_data(battery): #read csv save a torch tensors
	data = loadcsv("data/" + battery + ".csv") #ask Twan for name of csv with TTD

	print(data)
	
	



def main():
	load_data("B0005")


if __name__ == '__main__': 
	main()
