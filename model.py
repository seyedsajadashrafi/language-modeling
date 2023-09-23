import math

import torch
from torch import nn

import config


# class LanguageModel(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, dropouth=0.5, dropouti=0.5,
#                  dropout_embd=0.1, pretrained=None, tied=False):  # , tie_weights):
#
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.embedding_dim = embedding_dim
#
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         if pretrained:
#             vocab = torch.load('vocab.pt')
#             weights = pretrained.get_vecs_by_tokens(list(vocab.get_stoi().keys()))
#             self.embedding = self.embedding.from_pretrained(weights)
#
#         # self.lstms = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropouth, batch_first=True)
#         self.lstms = []
#         self.lstms.append(nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True))
#         self.lstms.append(nn.LSTM(hidden_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True))
#         self.lstms.append(nn.LSTM(hidden_dim, embedding_dim, num_layers=1, dropout=0, batch_first=True))
#
#         if config.weight_drop:
#             self.lstms = [WeightDrop(lstm, ['weight_hh_l0'], dropout=config.weight_drop) for lstm in self.lstms]
#         self.lstms = nn.ModuleList(self.lstms)
#
#         self.lockdrop = LockedDropout()
#         self.dropoute = dropout_embd
#         self.dropouti = dropouti
#         self.dropouth = dropouth
#         self.dropout = dropout
#
#         self.fc = nn.Linear(embedding_dim, vocab_size)
#
#         if tied:
#             # assert embedding_dim == hidden_dim, 'cannot tie, check dims'
#             # self.embedding.weight = self.fc.weight
#             self.embedding.weight = self.fc.weight
#             # self.fc.weight = nn.Parameter(self.embedding.weight.clone())
#
#         self.init_weights()
#
#     def init_weights(self):
#         init_range_emb = 0.1
#         self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
#         self.fc.weight.data.uniform_(-init_range_emb, init_range_emb)
#         self.fc.bias.data.zero_()
#         # init_range_other = 1 / math.sqrt(self.hidden_dim)
#         # for i in range(self.num_layers):
#         #     self.lstms.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
#         #                                                     self.hidden_dim).uniform_(-init_range_other,
#         #                                                                               init_range_other)
#         #     self.lstms.all_weights[i][1] = torch.FloatTensor(self.hidden_dim,
#         #                                                     self.hidden_dim).uniform_(-init_range_other,
#         #                                                                               init_range_other)
#
#     def forward(self, src, hidden):
#         # embedding = self.embedding(src)
#         embedding = embedded_dropout(self.embedding, src, dropout=self.dropoute if self.training else 0)
#         embedding = self.lockdrop(embedding, self.dropouti)
#
#         # output, hidden = self.lstms(embedding, hidden)
#         new_hiddens = []
#         for l, lstm in enumerate(self.lstms):
#             embedding, new_hidden = lstm(embedding, hidden[l])
#             new_hiddens.append(new_hidden)
#             if l != self.num_layers - 1:
#                 embedding = self.lockdrop(embedding, self.dropouth)
#         hidden = new_hiddens
#
#         output = self.lockdrop(embedding, self.dropout)
#         prediction = self.fc(output)
#         # prediction = prediction.view(-1, prediction.shape[-1])
#         return prediction  # , hidden
#
#     def init_hidden_states(self, batch_size, device):
#         """
#         Initialize the hidden states for the LSTM layers.
#
#         Args:
#             batch_size (int): The size of the batch.
#             device (str): The device (e.g., 'cuda' or 'cpu') to place the hidden states on.
#
#         Returns:
#             tuple: A tuple containing the initial hidden states for the LSTM layers.
#         """
#         # Initialize hidden and cell states with zeros.
#         hidden = []
#         for lstm in self.lstms:
#             hidden.append((torch.zeros(1, batch_size, lstm.module.hidden_size).to(device),
#                            torch.zeros(1, batch_size, lstm.module.hidden_size).to(device)))
#         return hidden
#
#     def repackage_hidden_states(self, hidden):
#         """
#         Detach and clone hidden states to prevent gradient propagation through time.
#
#         Args:
#             hidden (tuple): A tuple of hidden states.
#
#         Returns:
#             tuple: A tuple of detached and cloned hidden states.
#         """
#         if isinstance(hidden, torch.Tensor):
#             return hidden.detach()
#         else:
#             return tuple(self.repackage_hidden_states(h) for h in hidden)


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding)
        prediction = self.fc(output)
        return prediction  # , hidden

    def init_hidden_states(self, batch_size, device):
        """
        Initialize the hidden states for the LSTM layers.

        Args:
            batch_size (int): The size of the batch.
            device (str): The device (e.g., 'cuda' or 'cpu') to place the hidden states on.

        Returns:
            tuple: A tuple containing the initial hidden states for the LSTM layers.
        """
        # Initialize hidden and cell states with zeros.
        hidden = []
        for lstm in self.lstms:
            hidden.append((torch.zeros(1, batch_size, lstm.module.hidden_size).to(device),
                           torch.zeros(1, batch_size, lstm.module.hidden_size).to(device)))
        return hidden

    def repackage_hidden_states(self, hidden):
        """
        Detach and clone hidden states to prevent gradient propagation through time.

        Args:
            hidden (tuple): A tuple of hidden states.

        Returns:
            tuple: A tuple of detached and cloned hidden states.
        """
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(self.repackage_hidden_states(h) for h in hidden)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    embedding = torch.nn.functional.embedding(words, masked_embed_weight,
                                              padding_idx, embed.max_norm, embed.norm_type,
                                              embed.scale_grad_by_freq, embed.sparse)
    return embedding


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout):
        # # Generate a fixed dropout mask with the same shape as the input
        # dropout_mask = torch.zeros_like(x).bernoulli(1 - dropout_rate)
        # # Apply the dropout mask to the input
        # x = x * dropout_mask / (1 - dropout_rate)
        # return x
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            # w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            mask = torch.nn.functional.dropout(torch.ones_like(raw_w), p=self.dropout, training=True) * (1 - self.dropout)
            setattr(self.module, name_w, raw_w * mask)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
