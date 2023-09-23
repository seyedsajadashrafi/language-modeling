import torch
from torch.utils.data import Dataset

import torchtext

import config


def data_process(tokenizer, vocab, raw_text_iter, batch_size, bptt):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item))+vocab(['<eos>']), dtype=torch.long) for item in raw_text_iter]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 1, data)))
    # trim
    seq_len = len(data) // batch_size
    inputs = data[:seq_len * batch_size]
    targets = data[1:seq_len * batch_size + 1]
    # reshape the data & target
    inputs = inputs.view(batch_size, seq_len).t().contiguous()  # why t and conti...??
    targets = targets.view(batch_size, seq_len).t().contiguous()  # why t and conti...??
    # batchify the data & target
    inputs = inputs.unfold(dimension=0, size=bptt, step=bptt)
    targets = targets.unfold(dimension=0, size=bptt, step=bptt)
    return inputs, targets


class LanguageModelDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]  # .flatten()
