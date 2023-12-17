import torch
from data import vocab

index_to_char = vocab.get_itos()


def generate(model, wave, many):
    model.eval()
    generated = [1]

    for i in range(many):
        with torch.no_grad():
            prediction = model(wave[0].to(args.device).unsqueeze(0),
                               torch.LongTensor(generated).to(args.device).unsqueeze(0))
        if i != 0:
            argmax = prediction.squeeze().argmax(-1)[-1]
        else:
            argmax = prediction.squeeze().argmax(-1)
        if argmax.item() == 2:
            break
        generated.append(argmax.squeeze())
    generated = ''.join([index_to_char[d] for d in generated if index_to_char[d] != 'B'])
    return generated


def inference(model, wave, label):
    model.eval()

    generated = generate(model, many=300, wave=wave)
    print('predicted: ', generated)
    labelx = ''.join([index_to_char[d] for d in label[0]]).replace('B', '').replace('E', '')
    print('label: ', labelx)
