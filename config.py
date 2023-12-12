import argparse
import torch

__name__ = '__main__'

parser = argparse.ArgumentParser()
parser.add_argument('-d_model', type=int, default=300)
parser.add_argument('-cnn_channels', type=int, default=2)
parser.add_argument('-n_head', type=int,  default=4)
parser.add_argument('-encoder_layer', type=int, default=8)
parser.add_argument('-decoder_layer', type=int, default=8)
parser.add_argument('-device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('-n_ft', type=int, default=300)
parser.add_argument('-h_len', type=int, default=400)
parser.add_argument('-sr', type=int, default=22050)
parser.add_argument('-freq_mask', default=10)

args = parser.parse_args()
