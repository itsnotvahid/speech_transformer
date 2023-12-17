from torch import nn
import torch

from config import args
from sub_models import PositionalEncoding, Embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   batch_first=True,
                                                   dropout=0.3,
                                                   activation='gelu',
                                                   nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.positional_encoding = PositionalEncoding(emb_size=d_model, dropout=0.1)

    def forward(self, src):
        output = self.encoder(src)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_classes):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   batch_first=True,
                                                   dropout=0.3,
                                                   activation='gelu',
                                                   nhead=num_heads)
        self.positional_encoding = PositionalEncoding(emb_size=d_model, dropout=0.1)
        self.embedding = Embeddings(d_model, num_classes)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tgt, encoder_outputs):
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(args.device)
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, encoder_outputs, tgt_mask)
        y = self.head(output)
        return y
