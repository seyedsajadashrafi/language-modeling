from sys import platform

import torch

import wandb


pretrained = False

batch_size = 20

seq_len = 70

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

glove_dim = 300
vocab_dim = 400
embedding_dim = glove_dim if pretrained else vocab_dim

hidden_dim = 1150  # vocab_dim if pretrained else 256

num_layers = 3

dropout_rate = 0.2

tie_weights = True

lr = 30
wd = 1.2e-6

loss_train_hist, loss_valid_hist = [], []
metric_train_hist, metric_valid_hist = [], []

num_epochs = 300

scheduler = False

load_pretrain_model = False
model_name = 'model-ppl_120.7.pt'

clipping = True
clip = 0.25

weight_drop = 0.5

if platform == "linux" or platform == "linux2":
    wandb_enable = True
else:
    wandb_enable = False
if wandb_enable:
    wandb.login(key='6438a64a1b76437c5a6711319da484eaf4b53ec9')

    run = wandb.init(
        # Set the project where this run will be logged
        project="language-modeling-lstms",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
        })
