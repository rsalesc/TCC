import torch
import itertools
import numpy as np

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from . import utils, losses


class LineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1,
                 bias=True, dropout=0.0, bidirectional=True):
        super(LineLSTM, self).__init__()
        self.num_layers = layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=layers,
                            bias=bias,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.directions = int(bidirectional) + 1
        self.hidden = None
        self.batch_size = None

    def output_size(self):
        return self.directions * self.hidden_size

    def setup(self, batch_size):
        directions = self.directions
        self.batch_size = batch_size
        self.hidden = (
            torch.zeros(self.num_layers * directions,
                        batch_size, self.hidden_size),
            torch.zeros(self.num_layers * directions,
                        batch_size, self.hidden_size))

        # Move hidden states to GPU
        if torch.cuda.is_available():
            self.hidden = tuple([x.cuda() for x in self.hidden])

    def forward(self, packed_sequence):
        out = self.lstm(packed_sequence, self.hidden)[0]
        return utils.last_step_of_packed_sequence(out)


class TripletLSTM(nn.Module):
    def __init__(self,
                 alphabet=None,
                 alphabet_embedding=24,
                 char_hidden_size=64,
                 line_hidden_size=64,
                 embedding_size=128,
                 optimizer_fn=None,
                 margin=0.2):
        super().__init__()
        self.embed = nn.Embedding(alphabet, alphabet_embedding)
        self.char_level_lstm = LineLSTM(alphabet_embedding,
                                        char_hidden_size)
        self.line_level_lstm = LineLSTM(self.char_level_lstm.output_size(),
                                        line_hidden_size)
        self.line_level_seq = nn.Sequential(
            nn.Linear(self.line_level_lstm.output_size(), embedding_size)
        )

        cuda = torch.cuda.is_available()
        # Move to GPU before passing parameters to optimizer
        if cuda:
            self.cuda()

        self.optimizer = optimizer_fn(self.parameters())
        self.triplet_selector = losses.SemihardNegativeTripletSelector(
            margin, not cuda)
        self.loss = losses.OnlineTripletLoss(margin, self.triplet_selector)

    def forward(self, X):
        cuda = torch.cuda.is_available()

        batch_lengths = []
        lines = []
        for sample in X:
            n_lines = len(sample)
            batch_lengths.append(n_lines)

            # convert to list of LongTensors
            sample = [torch.LongTensor(x) for x in sample]
            sample, lengths, perm = utils.sort_and_pad_tensor(
                sample, type=torch.LongTensor)
            if cuda:
                sample = sample.cuda()
            sample = self.embed(sample)
            packed_sample = pack_padded_sequence(sample, lengths.cpu().numpy(),
                                                 batch_first=True)

            if cuda:
                packed_sample = packed_sample.cuda()

            self.char_level_lstm.setup(n_lines)
            out = self.char_level_lstm(packed_sample)
            out = out[utils.inverse_p(perm)]
            lines.append(out)

        batch_size = len(lines)
        batch, lengths, perm = utils.sort_and_pad_tensor(
            lines,
            type=torch.FloatTensor,
            features=self.char_level_lstm.output_size())
        packed_batch = pack_padded_sequence(batch, lengths.cpu().numpy(),
                                            batch_first=True)

        if cuda:
            packed_batch = packed_batch.cuda()

        self.line_level_lstm.setup(batch_size)
        out = self.line_level_lstm(packed_batch)
        out = out[utils.inverse_p(perm)]

        if cuda:
            out = out.cuda()
        out = self.line_level_seq(out)
        out = F.normalize(out, p=2, dim=-1)
        return out

    def train_on_batch(self, X, y):
        self.train(mode=True)
        self.zero_grad()

        cuda = torch.cuda.is_available()

        y = torch.from_numpy(np.array(y))

        out = self.forward(X)
        if cuda:
            out = out.cuda()
            y = y.cuda()

        loss, triplets = self.loss(out, y)
        loss.backward()
        self.optimizer.step()

        return out.detach().numpy(), loss.detach().numpy()

    def predict_on_batch(self, X):
        with torch.no_grad():
            self.train(mode=False)
            return self.forward(X).numpy()
