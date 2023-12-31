import torch
from torch import nn
from torch.optim import SGD
from config import args
from data import vocab, train_loader, valid_loader
from model import ASRNeuralNetwork
from loops import train_one_epoch, evaluate
from utils import inference

if __name__ == '__main__':
    choice = input('inf or train')
    if choice == 'train':
        best_loss_valid = 1e+4
        model = ASRNeuralNetwork(len(vocab)).to(args.device)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        for epoch in range(args.epoch):
            model, loss_train = train_one_epoch(model,
                                                loss_fn=loss_function,
                                                train_loader=train_loader,
                                                optimizer=optimizer,
                                                epoch=epoch)
            loss_valid = evaluate(model, valid_loader, loss_fn=loss_function)
            if loss_valid < best_loss_valid:
                best_loss_valid = loss_valid
                torch.save(model, 'model.pt')
                print('model saved')
    else:
        model = torch.load('model.pt')

        for i in range(10):
            wave, label = next(iter(valid_loader))
            inference(model, wave, label=label)
            print('>>'*50)
