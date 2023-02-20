import scipy.io
import numpy as np
import datetime

data = scipy.io.loadmat("B0005.mat")
data = data.items()
data = list(data)


data = np.array(data, dtype = object)
data = data[3][1][0][0][0][0]
#initial data filtering above

allcycles = []

for i in range(len(data)+1):
    onecycle = []
    
    data = data[i]
    print(data)
    cycle = data[0][0]
    cycle_list = np.full(len(data[3][0][0][0][0]), cycle)

    temp = data[1][0][0]
    temp_list = np.full(len(data[3][0][0][0][0]), temp)

    date = data[2][0]
    date_time = datetime.datetime(int(date[0]),
        int(date[1]),
        int(date[2]),
        int(date[3]),
        int(date[4])) + datetime.timedelta(seconds=int(date[5]))
    date_list = np.full(len(data[3][0][0][0][0]), date_time)

    data = data[3][0][0]

    if cycle == 'charge':
        voltage = np.array(data[0][0])
        current = np.array(data[1][0])
        temperature = np.array(data[2][0])
        current_charge = np.array(data[3][0])
        voltage_charge = np.array(data[4][0])
        time = np.array(data[5][0])
        onecycle.append(voltage)
        onecycle.append(current)
        onecycle.append(temperature)
        onecycle.append(current_charge)
        onecycle.append(voltage_charge)
        onecycle.append(time)
        
    elif cycle == 'discharge':
        voltage = np.array(data[0][0])
        current = np.array(data[1][0])
        temperature = np.array(data[2][0])
        current_load = np.array(data[3][0])
        voltage_load = np.array(data[4][0])
        time = np.array(data[5][0])
        capacity = np.array(data[6][0])
        onecycle.append(voltage)
        onecycle.append(current)
        onecycle.append(temperature)
        onecycle.append(current_load)
        onecycle.append(voltage_load)
        onecycle.append(time)
        onecycle.append(capacity)
        
    elif cycle == 'impedance':
        sense_current = np.array(data[0][0])
        battery_current = np.array(data[1][0])
        current_ratio = np.array(data[2][0])
        battery_impedance = np.array(data[3][0])
        rectified_impedance = np.array(data[4][0])
        re = np.array(data[5][0])
        rct = np.array(data[6][0])
        onecycle.append(sense_current)
        onecycle.append(battery_current)
        onecycle.append(current_ratio)
        onecycle.append(battery_impedance)
        onecycle.append(rectified_impedance)
        re_list = np.full(len(sense_current), re)
        rct_list = np.full(len(battery_current), rct)
        onecycle.append(re_list)
        onecycle.append(rct_list)
    
    allcycles.append(onecycle)
