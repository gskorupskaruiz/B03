import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
k_1 = 0
k_2 = 340

alpha = 0.1
df2 = pd.read_csv(r"Mark\data_alambda_plotssss.csv").to_numpy()[k_1:k_2,:]

pred_hybrid = df2[:,0]
pred_dd = df2[:,1]
test2 = df2[:,2]


# CONES:
indices = [ np.where(test2 == max(test2))[0][0], 100 + np.where(test2[100:] == 0)[0][0] ]
pred_hybrid = pred_hybrid[indices[0]:indices[1]]
pred_dd = pred_dd[indices[0]:indices[1]]
test2 = test2[indices[0]:indices[1]]
PHdatadriven_index = []
PHHyrbid_index = []
# for i in range(indices[1] - indices[0] - 1):
#     PHdatadriven_index.append(i) if (np.where(pred_dd[i] < (test2[i] - alpha* test2[i]) and (pred_dd[i+1] > test2[i+1] - alpha* test2[i+1]))) else  # only checks when enters from bottom
#     PHHyrbid_index.append(i) if (np.where(pred_hybrid[i] < (test2[i] - alpha* test2[i]) and (pred_hybrid[i+1] > test2[i+1] - alpha* test2[i+1]))) else
# print(f'ph data driven: {PHdatadriven_index}')
# print(f'ph hybrid: {PHHyrbid_index}')

fig = plt.figure()
ax = fig.add_subplot(111)    # The big subplot
#

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

xaxis = np.linspace(0,1, indices[1]-indices[0])
ax1.plot(xaxis, pred_hybrid,color = 'black', linewidth=2,label = r'$y_{pred}$ (hybrid)')
ax2.plot(xaxis, pred_dd,color = 'black', linewidth=2, label = r'$y_{pred}$ (data-driven)')

ax1.plot(xaxis, test2,'-.', color = 'blue', label = r'$y_{true}$')
ax2.plot(xaxis, test2,'-.', color = 'blue', label = r'$y_{true}$')

# ax1.vlines(xaxis[PHHyrbid_index], 0, 300, color = 'red', label = f'PH')
# ax2.vlines(xaxis[PHdatadriven_index], 0, 1, color = 'red', label = f'PH')

ax1.fill_between(xaxis, test2+alpha*test2,test2-alpha*test2, facecolor='orchid', alpha=0.4,label = r'$\alpha = 10$ %')
ax2.fill_between(xaxis, test2+alpha*test2,test2-alpha*test2, facecolor='orchid', alpha=0.4,label = r'$\alpha = 10$ %')
# fig.tight_layout()


ax1.legend()
ax2.legend()
ax.set_xlabel(r'$\lambda$ [$t/t_{TTD}$]')
ax.set_ylabel('Time to discharge [s]')
plt.show()