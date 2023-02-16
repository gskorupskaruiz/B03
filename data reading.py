import scipy.io
import numpy as np

data = scipy.io.loadmat("B0005.mat")
data = data.items()
data = list(data)


data = np.array(data, dtype = object)
data = data[3][1][0][0][0][0]
print(data)


all_tests = []

test = []

data = data[0]

cycle = data[0][0]
temp = data[1][0][0]
date = data[2][0]

data = data[3][0][0]

voltage = data[0][0]
current = data[1][0]
temperature = data[2][0]
current_charge = data[3][0]
voltage_charge = data[4][0]
time = data[5][0]

test.append(cycle)
test.append(temp)
test.append(date)
test.append(voltage)
test.append(current)
test.append(temperature)
test.append(current_charge)
test.append(voltage_charge)
test.append(time)
all_tests.append(test)

#print(mode)
#print(temp)
#print(time)