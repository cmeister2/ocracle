"""Append-only CTC vocabulary expansion for the exact native inference branch."""
import numpy as np
import paddle
from paddle import nn


def expand_vocabulary(model, prototypes):
    """Append Unicode scalars, preserving blank and every existing class index.

    Values name source glyphs for initialization only. Copying a prototype does
    not establish recognition of the new glyph. New biases are suppressed to
    limit the initial probability mass taken from the original vocabulary.
    """
    old = model.characters
    if any(len(char) != 1 or char in old or char in '\r\n'
           or 0xd800 <= ord(char) <= 0xdfff for char in prototypes):
        raise ValueError('Expansion requires unique new single Unicode scalars')
    if not prototypes or any(len(source) != 1 or source not in old for source in prototypes.values()):
        raise ValueError('Expansion needs existing source-glyph prototypes')
    weight = model.ctc_head.fc.weight.numpy().copy()
    bias = model.ctc_head.fc.bias.numpy().copy()
    source_indices = [old.index(source)+1 for source in prototypes.values()]
    new_head = nn.Linear(weight.shape[0], weight.shape[1]+len(prototypes))
    with paddle.no_grad():
        new_head.weight.set_value(np.concatenate([weight, weight[:, source_indices]], axis=1))
        new_head.bias.set_value(np.concatenate([bias, bias[source_indices]-6]))
    model.ctc_head.fc = new_head
    model.ctc_head.out_channels = weight.shape[1]+len(prototypes)
    model.characters += ''.join(prototypes)
    model.eval()
    if not np.array_equal(weight, new_head.weight.numpy()[:, :len(old)+1]):
        raise RuntimeError('Expansion changed original classifier columns')
    if not np.array_equal(bias, new_head.bias.numpy()[:len(old)+1]):
        raise RuntimeError('Expansion changed original classifier biases')
    return [{'character': char, 'class_index': len(old)+1+i,
             'prototype': source, 'prototype_class_index': old.index(source)+1,
             'initial_bias_offset': -6}
            for i, (char, source) in enumerate(prototypes.items())]
