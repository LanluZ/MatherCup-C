import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.layer0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.layer1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer0(x, None)
        x = self.layer1(x)
        return x
