 # inspired by https://github.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM1(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, hidden_size, num_layers, seq_length=1):
        super(LSTM1, self).__init__()
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
        self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: input features
        :return: prediction results
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # h_0 = Variable(torch.zeros((self.num_layers, x.size(0), self.hidden_size))) # hidden state
        # c_0 = Variable(torch.zeros((self.num_layers, x.size(0), self.hidden_size))) # internal state
        h_0 = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float64).to(device)
        c_0 = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float64).to(device)
        print((h_0).dtype)
        print((c_0).dtype)
        print((x).dtype)
        print(h_0.shape)
        print(c_0.shape)
        print(x.shape)
        out, (hn, cn) = self.lstm(x, (h_0 , c_0))  # lstm with input, hidden, and internal state
        print((hn).dtype)
        print((cn).dtype)
        print((out).dtype)

        hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :], dtype=torch.float64)
        hn_o = hn_o.view(-1, self.hidden_size)
        hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :], dtype=torch.float64)
        hn_1 = hn_1.view(-1, self.hidden_size)

        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out