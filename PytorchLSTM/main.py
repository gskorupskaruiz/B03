import torch

def load_data(battery):
    mat = loadmat('C:/Users/alexi/OneDrive/Documents/GitHub/GitHub/Data-Driven-Modelling-of-Lithium-ion-batteries/' + battery + '.mat')
    print('Total data in dataset: ', len(mat[battery][0, 0]['cycle'][0]))
    counter = 0
    dataset = []
    capacity_data = []

    for i in range(len(mat[battery][0, 0]['cycle'][0])):
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':
        ambient_temperature = row['ambient_temperature'][0][0]
        date_time = datetime.datetime(int(row['time'][0][0]),
                                    int(row['time'][0][1]),
                                    int(row['time'][0][2]),
                                    int(row['time'][0][3]),
                                    int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
        data = row['data']
        capacity = data[0][0]['Capacity'][0][0]
        for j in range(len(data[0][0]['Voltage_measured'][0])):
            voltage_measured = data[0][0]['Voltage_measured'][0][j]
            current_measured = data[0][0]['Current_measured'][0][j]
            temperature_measured = data[0][0]['Temperature_measured'][0][j]
            current_load = data[0][0]['Current_load'][0][j]
            voltage_load = data[0][0]['Voltage_load'][0][j]
            time = data[0][0]['Time'][0][j]
            dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                            voltage_measured, current_measured,
                            temperature_measured, current_load,
                            voltage_load, time])
        capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
        counter = counter + 1
    print(dataset[0])
    return [pd.DataFrame(data=dataset,
                        columns=['cycle', 'ambient_temperature', 'datetime',
                                    'capacity', 'voltage_measured',
                                    'current_measured', 'temperature_measured',
                                    'current_load', 'voltage_load', 'time']),
            pd.DataFrame(data=capacity_data,
                        columns=['cycle', 'ambient_temperature', 'datetime',
                                    'capacity'])]
      return 





def main():
	







    return










if __name__ == '__main__': 
	main()