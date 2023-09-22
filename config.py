import torch

import wandb


pretrained = False

batch_size = 128

seq_len = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

glove_dim = 300
vocab_dim = 300
embedding_dim = glove_dim if pretrained else vocab_dim

hidden_dim = 1150  # vocab_dim if pretrained else 256

num_layers = 3

dropout_rate = 0.2

tie_weights = False

lr = 15
wd = 1.2e-6

loss_train_hist, loss_valid_hist = [], []
metric_train_hist, metric_valid_hist = [], []

num_epochs = 300

scheduler = False

load_pretrain_model = True
model_name = 'model-ppl_120.7.pt'

clipping = True
clip = 0.25

weight_drop = 0.5

wandb_enable = True
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
