import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-lr', default=0.5)
parser.add_argument('-weight_decay', default=1e-6)
parser.add_argument('-epoch', default=100)
parser.add_argument('-specials', default=["", "B", "E"])
parser.add_argument('-bs', default=32)
parser.add_argument('-d_model', type=int, default=128*10) # 128 channel of CNN and 1/8 from mel numbers
parser.add_argument('-cnn_channels', type=int, default=3)
parser.add_argument('-n_head', type=int,  default=4)
parser.add_argument('-encoder_layer', type=int, default=4)
parser.add_argument('-decoder_layer', type=int, default=1)
parser.add_argument('-n_mels', default=80, type=int)
parser.add_argument('-device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
