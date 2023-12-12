from torch import nn

from config import args
from conv import CNNFeatureExtractor
from transformer import TransformerEncoder, TransformerDecoder
from sub_models import Transform


class ASRNeural(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        d_model = args.d_model
        self.spectrogram = Transform()
        self.cnn_ = CNNFeatureExtractor(args.cnn_channels)
        self.encoder = TransformerEncoder(d_model,
                                          num_layers=args.encoder_layer,
                                          num_heads=args.n_head)

        self.decoder = TransformerDecoder(num_heads=args.n_head,
                                          d_model=d_model,
                                          num_layers=args.decoder_layer,
                                          num_classes=num_classes)

    def forward(self, wave, target):
        spectrogram = self.spectrogram(wave)
        encoder_out = self.cnn_(spectrogram)
        encoder_out = self.encoder(encoder_out)
        y = self.decoder(target, encoder_out)
        return y
