import torch
import torch.nn as nn

class QuantileLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super(QuantileLSTM, self).__init__()

        dynamic_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dynamic_dropout
        )

        self.linear = nn.Linear(hidden_size, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_day = lstm_out[:, -1, :]

        prediction = self.linear(final_day)
        return prediction