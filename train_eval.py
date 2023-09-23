import torch
import tqdm

from utils import AverageMeter
import config


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
    model.train()
    loss_train = AverageMeter()
    metric.reset()

    with tqdm.tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch:
                tepoch.set_description(f'Epoch {epoch}')

            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            outputs = model(inputs)
            # print(outputs.shape, targets.shape)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]),
                           targets.flatten())

            loss.backward()
            if config.clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), n=len(targets))
            metric.update(outputs, targets)

            tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())

    return model, loss_train.avg, metric.compute().item()


def evaluate(model, test_loader, loss_fn, metric):
    model.eval()
    loss_eval = AverageMeter()
    metric.reset()
    # hidden = model.init_hidden_states(test_loader.batch_size, config.device)

    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            outputs = model(inputs)

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]),
                           targets.flatten())
            loss_eval.update(loss.item(), n=len(targets))

            metric(outputs, targets)

    return loss_eval.avg, metric.compute().item()
