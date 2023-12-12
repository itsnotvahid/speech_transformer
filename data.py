import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import re
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from config import args
from torch.nn.utils.rnn import pad_sequence
import __main__

data = pd.read_csv('new_data.csv')
get_lower = lambda x: re.sub(r"[^a-z\s']", '', x.lower())
vocab = build_vocab_from_iterator(
    data['info'].apply(get_lower), special_first=True, specials=args.specials)
vocab.set_default_index(vocab[''])


def collate(batch):
    mel_waves = pad_sequence([b[0].squeeze() for b in batch], batch_first=True).unsqueeze(1)
    labels = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=vocab[''])
    return mel_waves, labels


class LJSpeechSet(Dataset):
    def __init__(self, _data):
        pathx = 'LJSpeech-1.1/wavs/'
        self.data = dict()
        self.wave_forms, self.labels, self.sample_rates = list(), list(), list()

        for i, row in _data.reset_index().iterrows():
            wave_form, s_r = torchaudio.load(pathx + str(row['f_name']))
            label = torch.LongTensor([vocab[c] for c in "B" +
                                      re.sub(r"[^a-z\s']", '', str(row['info'].lower())) + "E"])

            self.data[i] = {'wave': wave_form, 'label': label}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        wave, label = self.data[ind]['wave'], self.data[ind]['label']
        return wave, label


setattr(__main__, "LJSpeechSet", LJSpeechSet)

train_set = torch.load('train_set')
valid_set = torch.load('valid_set')

train_loader = DataLoader(train_set, args.bs, shuffle=True, collate_fn=collate)
valid_loader = DataLoader(valid_set, args.bs, shuffle=True, collate_fn=collate)