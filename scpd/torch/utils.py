import torch
import numpy as np


def last_step_indices(self):
    """A helper function for :func:`last_step_tensor`.
    The returned indices is used to select the last step of a :class:`PackedSequence` object.
    Arguments:
    Returns:
        List of int containing the indices of the last step of each sequence.
    """
    indices = []
    sum_batch_size = 0
    for i, batch_size in enumerate(self.batch_sizes):
        sum_batch_size += batch_size
        if i == len(self.batch_sizes) - 1:
            for j in range(batch_size):
                indices.append(sum_batch_size - 1 - j)
        elif batch_size > self.batch_sizes[i + 1]:
            for j in range(batch_size - self.batch_sizes[i + 1]):
                indices.append(sum_batch_size - 1 - j)
    return indices[::-1]


def last_step_of_packed_sequence(self):
    """Extract the last step of each sequence of a :class:`PackedSequence` object.
    It is useful for extracting rnn's output
    The returned Variable's data will be of size Bx*, where B is the batch size.
    Arguments:
    Returns:
        Variable containing the last step of each sequence in the batch.
    """
    indices = torch.LongTensor(last_step_indices(self))
    if self.data.data.is_cuda:
        indices = indices.cuda(self.data.data.get_device())
    last_step = self.data.index_select(0, indices)
    return last_step


def sort_and_pad_tensor(batch, type, lengths=None, features=tuple()):
    if isinstance(features, int):
        features = (features,)
    if not isinstance(features, tuple):
        features = tuple(features)
    if lengths is None:
        lengths = [len(line) for line in batch]
    lengths = torch.LongTensor(lengths)

    seq_tensor = torch.zeros((len(batch), lengths.max()) + features)
    seq_tensor = seq_tensor.type(type)
    for idx, (seq, seqlen) in enumerate(zip(batch, lengths)):
        seq_tensor[idx, :seqlen] = seq.type(type)

    lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    return seq_tensor, lengths, perm_idx


def sort_lengths(lengths):
    lengths = torch.LongTensor(lengths)
    lengths, perm_idx = lengths.sort(0, descending=True)
    return lengths, perm_idx


def inverse_p(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse
