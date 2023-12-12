from torcheval.metrics import Mean
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from config import args
from data import vocab


def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=None, device=args.device):
    model.train()
    metric = Mean().to(device)
    with tqdm(train_loader, unit='batch') as tepochs:
        for wave, labels in tepochs:
            if epoch is not None:
                tepochs.set_description(f'epoch:{epoch}')
            yp = model(wave.to(device), labels.to(device)[:, :-1])
            loss = loss_fn(yp.transpose(2, 1), labels.to(device)[:, 1:])
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()
            metric.update(loss)

            tepochs.set_postfix(loss=metric.compute().item())
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


def generate(model, wave, many):
    model.eval()
    generated = [1]
    itos = vocab.get_itos()

    for i in range(many):
        with torch.no_grad():
            preded = model(wave[0].to(args.device).unsqueeze(0),
                           torch.LongTensor(generated).to(args.device).unsqueeze(0))
        if i != 0:
            argm = preded.squeeze().argmax(-1)[-1]
        else:
            argm = preded.squeeze().argmax(-1)
        if argm.item() == 2:
            break
        generated.append(argm.squeeze())
    generated = ''.join([itos[d] for d in generated if itos[d] != 'B'])
    return generated
