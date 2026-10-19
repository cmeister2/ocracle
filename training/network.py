"""Checkpoint-compatible CRNN, independently expressed from its layer specification.

Architecture reference: robertknight/ocrs-models, RecognitionModel at
068934f1725959b734edef025b13c46ddf784326. Inputs are normalized grayscale
[N,1,64,W]; output is CTC log probabilities [time,N,class].
"""
import torch
from torch import nn

ALPHABET = ' 0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


class RecognitionModel(nn.Module):
    """Convolutional features and two bidirectional GRU layers for CTC OCR."""

    def __init__(self, alphabet=ALPHABET):
        super().__init__()
        if not alphabet or len(set(alphabet))!=len(alphabet):
            raise ValueError("Alphabet must contain distinct Unicode characters")
        self.alphabet=alphabet
        layers = []
        # (channels in, channels out, batch norm, pool after, kernel)
        specs = [(1, 32, False, (2, 2), 3),
                 (32, 64, True, (2, 2), 3),
                 (64, 128, False, None, 3),
                 (128, 128, True, (2, 1), 3),
                 (128, 128, False, None, 3),
                 (128, 128, True, (2, 1), 3),
                 (128, 128, True, None, (2, 2))]
        for index, (cin, cout, norm, pool, kernel) in enumerate(specs):
            layers.append(nn.Conv2d(cin, cout, kernel, padding=1, bias=not norm))
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            if index != len(specs) - 1:
                layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(pool))
        layers.append(nn.AvgPool2d((4, 1)))
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(128, 256, num_layers=2, bidirectional=True)
        self.output = nn.Sequential(nn.Linear(512, len(alphabet) + 1), nn.LogSoftmax(2))

    def forward(self, images):
        features = self.conv(images).permute(3, 0, 1, 2)
        features = features.flatten(2)
        sequence, _ = self.gru(features)
        return self.output(sequence)


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = RecognitionModel(state.get('alphabet',ALPHABET))
    model.load_state_dict(state.get('model_state', state))
    return model


def decode(sequence, alphabet=ALPHABET):
    """Greedy CTC decode: collapse runs, then remove blank index zero."""
    output, previous = [], -1
    for index in sequence:
        if index != previous and index != 0:
            output.append(alphabet[index - 1])
        previous = index
    return ''.join(output)


def adapt_model(checkpoint, alphabet, seed=191993):
    """Create a Unicode profile head while retaining learned visual features.

    Matching character rows and the blank row are copied by character identity,
    rather than by position. Unseen glyph rows need genuine language training;
    this initialization alone does not provide recognition support.
    """
    source=load_model(checkpoint)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        target=RecognitionModel(alphabet)
    features={name:value for name,value in source.state_dict().items()
              if not name.startswith('output.0.')}
    missing,unexpected=target.load_state_dict(features,strict=False)
    if set(missing)!={'output.0.weight','output.0.bias'} or unexpected:
        raise RuntimeError('Backbone architecture differs from checkpoint')
    old_rows={character:index+1 for index,character in enumerate(source.alphabet)}
    with torch.no_grad():
        target.output[0].weight[0].copy_(source.output[0].weight[0])
        target.output[0].bias[0].copy_(source.output[0].bias[0])
        for index,character in enumerate(alphabet):
            if character in old_rows:
                old=old_rows[character]
                target.output[0].weight[index+1].copy_(source.output[0].weight[old])
                target.output[0].bias[index+1].copy_(source.output[0].bias[old])
    return target
