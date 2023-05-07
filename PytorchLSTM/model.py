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

    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.2)
        self.fc_1 = nn.Linear(hidden_size, 30)  # fully connected 1
        self.fc_2 = nn.Linear(30, 10)  # fully connected 2
        self.fc = nn.Linear(10, 1)  # fully connected last layer
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    

    def forward(self, x):
        """
        :param x: input features
        :return: prediction results
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device).double()
        c_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device).double()

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        # numpy_array = hn.to('cpu').detach().numpy()
        # essentially says return sequence is false...
        hn_o = torch.Tensor(hn.to('cpu').detach().numpy()[-1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size).double().to(device)
        hn_1 = torch.Tensor(hn.to('cpu').detach().numpy()[1, :, :])
        hn_1 = hn_1.view(-1, self.hidden_size).double().to(device)

        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))    
        #print(f"output of lstm {(self.relu(output)).shape}")
        #print(f"after the first dense {out.shape}")
        out = self.relu(self.fc_2(out))
        #print(f"after the second dense {out.shape}")
        out = self.dropout(out)
        out = self.fc(out)
        #print(out.shape)
        return out


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length, batch):
        super(CNNLSTM, self).__init__()
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length
        self.batch = batch   

        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.2)
        self.fc_1 = nn.Linear(hidden_size, 20)  # fully connected 1

        self.conv1 = nn.Conv1d(20, 32, kernel_size=2, stride=1) # note: output_shape_conv1 = (input_channel_conv1 - kernel + 2*padding)/stride + 1   [so here (20-2)/1+1 = 19]
        self.batch1 =nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 1, kernel_size=1, stride = 1, padding=2) # note: output_shape_conv3 = output_shape_conv2 - kernel + 2*padding)/stride + 1  [so here (19-1+4)/1+1 = 23]
        self.batch2 =nn.BatchNorm1d(1)

        self.fc_2 = nn.Linear(23, 10)  # fully connected 2 - here the number of hidden_input_neurons = output_shape_conv3
        self.fc = nn.Linear(10, 1)  # fully connected last layer
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device).double()
        c_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device).double()

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        out = self.relu(self.fc_1(self.relu(output)))
        #out = self.relu(self.fc_2(out))

        # print(f"after lstm {output.shape}")
        # print(f"after first dense lstm {out.shape}")

        out = self.conv1(out)
        # print(f"after first conv {out.shape}")

        out = self.batch1(out)
        # print(f"after batchnorm1 {out.shape}")
        out = self.conv2(out)
        # print(f"after conv2 {out.shape}")
        out = self.batch2(out)
        # print(f"after batchnorm2 {out.shape}")
        out = self.relu(self.fc_2(out))
        # print(f"after dense {out.shape}")
        out = self.dropout(out)
        
        outg = self.fc(out)
        # print(f"last {outg.shape}")
        return outg
    

class CNNLSTMog(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(CNNLSTMog, self).__init__()

        self.conv1 = nn.Conv1d(output_size, 64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(64,32,kernel_size=1, stride = 1, padding=1)
        self.batch1 =nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride = 1, padding=1)
        self.batch2 =nn.BatchNorm1d(32)
        
        self.LSTM = nn.LSTM(input_size=10, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(32*hidden_size, output_size) #ouput size
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU
        

    def forward(self, x):

        x = self.conv1(x)
        #x = self.relu(x)
        x = self.conv2(x)
        #x = self.relu(x)
        x = self.batch1(x)
        x = self.conv3(x)
        #x = self.relu(x)
        x = self.batch2(x)
        
        x, h = self.LSTM(x) 
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))

        x = self.dropout(x)
        output = self.fc1(x)
        return output
    


class ParametricCNNLSTM():
    def __init__(self, input_size output_size, hidden_size, num_layers, cnn_layers, cnn_kernel_size, cnn_stride, cnn_padding, cnn_output_size):
        super(CNNLSTM, self).__init__()
        for i in cnn_layers:
            setattr(self, 'conv'+i, nn.Conv1d(in_channels = input_size , out_channels = cnn_output_size, kernel_size= cnn_kernel_size , stride = cnn_stride , padding= cnn_padding))


        
        self.batch1 =nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride = 1, padding=1)
        self.batch2 =nn.BatchNorm1d(32)
        
        self.LSTM = nn.LSTM(input_size=10, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(32*hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU
        

    def forward(self, x):

        x = self.conv1(x)
        #x = self.relu(x)
        x = self.conv2(x)
        #x = self.relu(x)
        x = self.batch1(x)
        x = self.conv3(x)
        #x = self.relu(x)
        x = self.batch2(x)
        
        x, h = self.LSTM(x) 
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))

        x = self.dropout(x)
        output = self.fc1(x)
        return output
    
