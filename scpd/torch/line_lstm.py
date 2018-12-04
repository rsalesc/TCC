import torch

from torch import nn

from . import utils


class LineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1,
                 bias=True, dropout=0.0, bidirectional=True):

        self.num_layers = layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=layers,
                            bias=bias,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.hidden = None
        self.batch_size = None

    def setup(self, batch_size):
        self.batch_size = batch_size
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, packed_sequence):
        out = self.lstm(packed_sequence, self.hidden)
        return utils.last_step_of_packed_sequence(out)
