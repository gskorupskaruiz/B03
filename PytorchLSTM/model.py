 # inspired by https://github.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Alexis stuff
# class LSTM1(nn.Module):
#     """LSTM architecture"""

#     def __init__(self, input_size, hidden_size, num_layers, seq_length=1):
#         super(LSTM1, self).__init__()
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # hidden state
#         self.num_layers = num_layers  # number of layers
#         self.seq_length = seq_length  # sequence length

#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
#                             dropout=0.1)
#         self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
#         self.fc_2 = nn.Linear(16, 8)  # fully connected 2
#         self.fc = nn.Linear(8, 1)  # fully connected last layer

#         self.dropout = nn.Dropout(0.1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         """
#         :param x: input features
#         :return: prediction results
#         """
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # h_0 = Variable(torch.zeros((self.num_layers, x.size(0), self.hidden_size))).to(device).double() # hidden state
#         # c_0 = Variable(torch.zeros((self.num_layers, x.size(0), self.hidden_size))).to(device).double() # internal state
#         h_0 = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float64).to(device)
#         c_0 = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float64).to(device)
#         out, (hn, cn) = self.lstm(x, (h_0 , c_0))  # lstm with input, hidden, and internal state


#         # hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :], dtype=torch.float64)
#         # hn_o = hn_o.view(-1, self.hidden_size)
#         # hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :], dtype=torch.float64)
#         # hn_1 = hn_1.view(-1, self.hidden_size)

#         hn_o = hn[-1,:].detach().clone().to(torch.float64)
#         hn_o = hn_o.view(-1, self.hidden_size)
#         hn_1 = hn[1,:].detach().clone().to(torch.float64)
#         hn_1 = hn_1.view(-1, self.hidden_size)


#         out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
#         out = self.relu(self.fc_2(out))
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out
    


### my stuff
class LSTM1(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, hidden_size, num_layers, seq_length, k, p):
        super(LSTM1, self).__init__()
        # self.input_size = input_size  # input size
        # self.hidden_size = hidden_size  # hidden state
        # self.num_layers = num_layers  # number of layers
        # self.seq_length = seq_length  # sequence length

        # self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=20, padding ='same') # bias = False
        # self.act1 = nn.ReLU(inplace= True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # self.lstm = nn.LSTM(input_size=20, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
        #                     dropout=0.1)
        # self.fc_1 = nn.Linear(hidden_size, 2000)  # fully connected 1
        # self.fc_2 = nn.Linear(2000, 10)  # fully connected 2
        # self.fc = nn.Linear(10, self.seq_length)  # fully connected last layer

        # self.dropout = nn.Dropout(0.1)
        # self.relu = nn.ReLU()

        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        self.conv1 = nn.Conv1d(self.seq_length, hidden_size, kernel_size=k, padding=p, stride=2, bias=False)
        self.act1 = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
        self.fc_1 = nn.Linear(hidden_size, 20)  # fully connected 1
        self.fc_2 = nn.Linear(20, 10)  # fully connected 2
        self.fc = nn.Linear(10, self.seq_length)  # fully connected last layer
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
    

    def forward(self, x):
        """
        :param x: input features
        :return: prediction results
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        out = self.conv1(x)
        out = self.relu(out)
   
        h_0 = torch.randn(self.num_layers, out.size(0), self.hidden_size).to(device).double()
        c_0 = torch.randn(self.num_layers, out.size(0), self.hidden_size).to(device).double()

        output, (hn, cn) = self.lstm(out, (h_0, c_0))  # lstm with input, hidden, and internal state

        numpy_array = hn.to('cpu').detach().numpy()

        hn_o = torch.Tensor(hn.to('cpu').detach().numpy()[-1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size).double().to(device)
        hn_1 = torch.Tensor(hn.to('cpu').detach().numpy()[1, :, :])
        hn_1 = hn_1.view(-1, self.hidden_size).double().to(device)

        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out


import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()
        
        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=64*int(input_shape[1]/4)*int(input_shape[2]/4), hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # Define output layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Apply CNN layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        
        # Reshape data for LSTM input
        x = x.reshape(x.size(0), 1, -1)
        
        # Apply LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Apply output layer
        out = self.fc(lstm_out[:, -1, :])
        
        return out