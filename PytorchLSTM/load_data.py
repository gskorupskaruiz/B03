import pandas as pd


data_fields = {
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Current_charge', 'Voltage_charge', 'Time', 'Capacity'}

def load_data(battery): #read csv save a torch tensors
	data = pd.read_csv("data/" + battery + ".csv") #ask Twan for name of csv with TTD
	print(data.type())
    



def _datevec2datetime(vec):
    '''Convert MATLAB datevecs to Python DateTime Objects

    MATLAB datevec example: 
    `[2008.   ,    5.   ,   22.   ,   21.   ,   48.   ,   39.015]`

    Parameters:
    - `vec`: list-like object in MATLAB datevec format
    '''
    return datetime(
        year=int(vec[0]),
        month=int(vec[1]),
        day=int(vec[2]),
        hour=int(vec[3]),
        minute=int(vec[4]),
        second=int(vec[5]),
        microsecond=int((vec[5]-int(vec[5]))*1000)
    )

