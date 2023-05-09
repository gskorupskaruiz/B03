import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # LSTM layer
        self.fc = nn.Linear(hidden_size, output_size) #output layer

    def forward(self, x):
        # Set initial hidden and cell states to 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  
        return out
