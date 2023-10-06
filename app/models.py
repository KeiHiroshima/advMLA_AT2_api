import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.n_layers = num_layers
        self.hidden_dim = hidden_size
        self.n_output = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        # cell state
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_dim)
        y = self.fc(hn)

        y = y.reshape(self.n_output, -1)

        # print(y)

        return y
