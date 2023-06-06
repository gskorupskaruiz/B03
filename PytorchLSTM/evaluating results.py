import readMODEL as r
import desperate_kfold as d 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import csv 

hyperparam_hybrid = [0.00167, 8, 2000, 5, [1, 9, 18, 27, 36], [1, 5, 2.0, 7.0, 9.0], [1, 1, 1, 1, 1], [1, 1, 2, 3, 4], 14, 3, [24, 18, 12, 6, 1]] # best hybrid
hyperparam_lstmcnn = [0.01264, 12, 425, 1, [1], [1], [1], [1], 22, 1, [1, 1]] 
model_hybrid = r.read_model('hybrid', hyperparam_hybrid)
model_data = r.read_model('LSTM-CNN', hyperparam_lstmcnn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_hybrid.to(device)
model_hybrid.double()
model_data.to(device)
model_data.double()

### one plot - best plot 
testing_battery = ['B0007']
test_battery_h, time_mean_h, time_std_h = d.load_data_normalise_cv(testing_battery, 'hybrid')
test_data_h, test_truth_h,  _, _, _ = d.load_gpu_data_with_batches_cv(test_battery_h, seq_length=hyperparam_hybrid[1], which_model='hybrid')

test_battery, time_mean_, time_std_ = d.load_data_normalise_cv(testing_battery, 'LSTM-CNN')
test_data_, test_truth_,  _, _, _ = d.load_gpu_data_with_batches_cv(test_battery, seq_length=hyperparam_lstmcnn[1], which_model='LSTM-CNN')

predictions_hybrid = model_hybrid(test_data_h).to('cpu').detach().numpy()
predict_hybrid = predictions_hybrid.squeeze(2) * time_std_h + time_mean_h
test_truth_h = test_truth_h.squeeze(2).to('cpu').detach().numpy() * time_std_h + time_mean_h
predictions_lstmcnn = model_data(test_data_).to('cpu').detach().numpy()
predict_lstmcnn = predictions_lstmcnn.squeeze(2) * time_std_ + time_mean_


plt.figure()
plt.plot(predict_hybrid, label='Hybrid model predictions', linewidth=1, color='red')
plt.plot(test_truth_h, label='Ground truth', linewidth=1, color='black')
plt.plot(predict_lstmcnn, label='Data-driven model predictions', linewidth=1, color='blue')
plt.xlabel('Instance (-)')
plt.ylabel('Time to disharge (seconds)')
plt.legend()
plt.style.use('seaborn-darkgrid')
plt.show()

with open('data_alambda_plots.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Predict Hybrid', 'Predict Data Driven', 'Test Truth'])
    csv_writer.writerows([predict_hybrid, predict_lstmcnn, test_truth_h])

### error plot for k fold 
b_0005 = ['B0005']
b_0006 = ['B0006']
b_0007 = ['B0007']
b_0018 = ['B0018']

b_0005, time_mean_5, time_std_5 = d.load_data_normalise_cv(b_0005, type_of_model)
test_data_5, test_truth_5,  _, _, _ = d.load_gpu_data_with_batches_cv(test_battery, seq_length=final_hyperparameters[1], which_model=type_of_model)

b_0006, time_mean_6, time_std_6 = d.load_data_normalise_cv(b_0006, type_of_model)
test_data_6, test_truth_6,  _, _, _ = d.load_gpu_data_with_batches_cv(test_battery, seq_length=final_hyperparameters[1], which_model=type_of_model)

b_0007, time_mean_7, time_std_7 = d.load_data_normalise_cv(b_0007, type_of_model)
test_data_7, test_truth_7,  _, _, _ = d.load_gpu_data_with_batches_cv(test_battery, seq_length=final_hyperparameters[1], which_model=type_of_model)

b_0018, time_mean_18, time_std_18 = d.load_data_normalise_cv(b_0018, type_of_model)
test_data_18, test_truth_18,  _, _, _ = d.load_gpu_data_with_batches_cv(test_battery, seq_length=final_hyperparameters[1], which_model=type_of_model)

pred_5 = model(test_data_5).squeeze(2).to('cpu').detach().numpy() * time_std_5 + time_mean_5
truth_5 = test_truth_5.squeeze(2).to('cpu').detach().numpy() * time_std_5 + time_mean_5
error_5 = truth_5 - pred_5
print(f'done with 5')
pred_6 = model(test_data_6).squeeze(2).to('cpu').detach().numpy() * time_std_6 + time_mean_6
truth_6 = test_truth_6.squeeze(2).to('cpu').detach().numpy() * time_std_6 + time_mean_6
error_6 = truth_6 - pred_6
print('done with 6')
pred_7 = model(test_data_7).squeeze(2).to('cpu').detach().numpy() * time_std_7 + time_mean_7
truth_7 = test_truth_7.squeeze(2).to('cpu').detach().numpy() * time_std_7 + time_mean_7
error_7 = truth_7 - pred_7
print(f'done with 7')
pred_18 = model(test_data_18).squeeze(2).to('cpu').detach().numpy() * time_std_18 + time_mean_18
truth_18 = test_truth_18.squeeze(2).to('cpu').detach().numpy() * time_std_18 + time_mean_18
error_18 = truth_18 - pred_18
print(f'done with 18')
plt.figure()
# plt.plot(error_5, label='Test battery 005', linewidth=1, color='red')
# plt.plot(error_6, label='Test battery 006', linewidth=1, color='blue')
plt.plot(error_7, label='Test battery 007', linewidth=1, color='green')
# plt.plot(error_18, label='Test battery 018', linewidth=1, color='black')
plt.xlabel('Instance (-)')
plt.ylabel('Error between model predictions and ground truth')
plt.legend()
plt.style.use('seaborn-darkgrid')


plt.figure()
plt.plot(predictions_plot, label='Model predictions', linewidth=1, color='red')
plt.plot(y_kfold, label='Ground truth', linewidth=1, color='blue')

plt.xlabel('Instance (-)')
plt.ylabel('Time to disharge (seconds)')
plt.legend()
plt.style.use('seaborn-darkgrid')
plt.show()