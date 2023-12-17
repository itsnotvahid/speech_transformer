from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from torcheval.metrics import Mean
import torch
from config import args


def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=None, device=args.device):
    model.train()
    metric = Mean().to(device)
    with tqdm(train_loader, unit='batch') as t_epoch:
        for wave, labels in t_epoch:
            if epoch is not None:
                t_epoch.set_description(f'epoch:{epoch}')
            yp = model(wave.to(device), labels.to(device)[:, :-1])
            loss = loss_fn(yp.transpose(2, 1), labels.to(device)[:, 1:])
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()
            metric.update(loss)

            t_epoch.set_postfix(loss=metric.compute().item())
    return model, metric.compute().item()


def evaluate(model, test_loader, loss_fn, device=args.device):
    model.eval()
    metric = Mean().to(device)
    with torch.no_grad():
        for wave, labels in test_loader:
            yp = model(wave.to(device), labels.to(device)[:, :-1])
            loss = loss_fn(yp.transpose(2, 1), labels.to(device)[:, 1:])
            metric.update(loss)
    print(metric.compute().item())
    return metric.compute().item()