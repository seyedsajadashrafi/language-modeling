# Import
import numpy as np
import matplotlib.pyplot as plt

import torchtext
from torchtext.data.utils import get_tokenizer
import torchtext.data as data
from torchtext.vocab import build_vocab_from_iterator, GloVe, vocab
from torchtext.datasets import WikiText2, WikiText103

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import tqdm
import torchmetrics as tm

import config
from utils import *
from config import *
from dataset import *
from model import LanguageModel
from train_eval import train_one_epoch, evaluate

import wandb

import math


def main():
    # Set the random seed manually for reproducibility.
    np.random.seed(100)
    torch.manual_seed(100)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(100)

    # Create a vocabulary
    train_iter, valid_iter, test_iter = WikiText2('./data/')
    tokenizer = get_tokenizer('basic_english')

    if config.pretrained:
        glove = GloVe(name='6B', dim=config.glove_dim)
        vocabulary = vocab(glove.stoi, min_freq=0)
        vocabulary.set_default_index(vocabulary['unk'])
        embedding_pretrained = glove
    else:
        vocabulary = build_vocab_from_iterator(map(tokenizer, train_iter), min_freq=0, specials=['<unk>'])
        vocabulary.append_token('<eos>')
        vocabulary.set_default_index(vocabulary['<unk>'])
        torch.save(vocabulary, 'vocab.pt')
        # glove = GloVe(name='6B', dim=config.glove_dim)
        # embedding_pretrained = glove
        embedding_pretrained = None

    vocab_size = len(vocabulary)

    X_train, y_train = data_process(tokenizer, vocabulary, train_iter, config.batch_size, config.seq_len, int(config.seq_len/2))
    X_valid, y_valid = data_process(tokenizer, vocabulary, valid_iter, config.batch_size, config.seq_len, config.seq_len)
    # X_test, y_test = data_process(tokenizer, vocabulary, test_iter, config.batch_size * 2, config.seq_len)

    train_set = LanguageModelDataset(X_train, y_train)
    valid_set = LanguageModelDataset(X_valid, y_valid)
    # test_set = LanguageModelDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False)

    model = LanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, 0.4, 0.25, 0.4, 0.1,
                          pretrained=embedding_pretrained, tied=tie_weights).to(device)

    if load_pretrain_model:
        model = torch.load(model_name)
    model.requires_grad_(True)
    num_params = num_trainable_params(model)
    print(f'The model has {num_params:,} trainable parameters!')

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)#, momentum=0.9, nesterov=True)
    # optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.)
    if scheduler:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    loss_fn = nn.CrossEntropyLoss()
    metric = tm.text.Perplexity().to(device)

    best_loss_valid = torch.inf
    epoch_counter = 0
    for epoch in range(num_epochs):

        # Train
        # model, loss_train, metric_train = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch+1)

        # Validation
        loss_valid, metric_valid = evaluate(model, valid_loader, loss_fn, metric)

        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)

        metric_train_hist.append(metric_train)
        metric_valid_hist.append(metric_valid)

        if loss_valid < best_loss_valid:
            torch.save(model, f'model.pt')
            best_loss_valid = loss_valid
            print('\nModel Saved!')

        print(f'Valid: Loss = {loss_valid:.4}, Metric = {metric_valid:.4}\n')

        epoch_counter += 1

        if scheduler:
            lr_scheduler.step()

        if wandb_enable:
            wandb.log({"metric_train": metric_train, "loss_train": loss_train,
                       "metric_valid": metric_valid, "loss_valid": loss_valid})
        else:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(np.arange(epoch+1), loss_train_hist)
            ax[0].plot(np.arange(epoch+1), loss_valid_hist)
            ax[0].set_title('Loss')
            ax[0].legend(['Train', 'Valid'])
            ax[0].set_xlim(0, num_epochs)
            ax[1].plot(np.arange(epoch+1), metric_train_hist)
            ax[1].plot(np.arange(epoch+1), metric_valid_hist)
            ax[1].set_title('PPL')
            ax[1].legend(['Train', 'Valid'])
            ax[1].set_xlim(0, num_epochs)
            plt.savefig('learning-curve.png')


if __name__ == '__main__':
    main()


