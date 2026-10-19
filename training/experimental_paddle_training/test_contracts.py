"""Focused contracts for append-only classes and the model input convention."""
import unittest
import numpy as np
import paddle
from paddle import nn
from PIL import Image
from adapt import expand_vocabulary
from inputs import tensor


class TinyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.characters = '!ß“ ABC'
        self.ctc_head = nn.Layer()
        self.ctc_head.fc = nn.Linear(3, len(self.characters)+1)
        self.ctc_head.out_channels = len(self.characters)+1


class ContractTests(unittest.TestCase):
    def test_append_preserves_blank_and_all_old_indices(self):
        model = TinyModel()
        old = model.characters
        weight = model.ctc_head.fc.weight.numpy().copy()
        bias = model.ctc_head.fc.bias.numpy().copy()
        mapping = expand_vocabulary(model, {'¡': '!', 'ẞ': 'ß', '„': '“'})
        self.assertEqual(model.characters, old+'¡ẞ„')
        np.testing.assert_array_equal(model.ctc_head.fc.weight.numpy()[:, :len(old)+1], weight)
        np.testing.assert_array_equal(model.ctc_head.fc.bias.numpy()[:len(old)+1], bias)
        for offset, record in enumerate(mapping):
            self.assertEqual(record['class_index'], len(old)+1+offset)
            np.testing.assert_array_equal(model.ctc_head.fc.weight.numpy()[:, record['class_index']],
                                          weight[:, old.index(record['prototype'])+1])
        self.assertFalse(model.training)

    def test_invalid_expansion_does_not_mutate(self):
        for prototypes in [{}, {'!': '!'}, {'XY': '!'}, {'¡': 'XY'}, {'¡': 'Q'}, {'\ud800': '!'}]:
            model = TinyModel()
            before = model.ctc_head.fc.weight.numpy().copy()
            with self.subTest(prototypes=repr(prototypes)), self.assertRaises(ValueError):
                expand_vocabulary(model, prototypes)
            self.assertEqual(model.characters, '!ß“ ABC')
            np.testing.assert_array_equal(model.ctc_head.fc.weight.numpy(), before)

    def test_input_bgr_affine_and_normalized_zero_padding(self):
        image = Image.new('RGB', (3, 5), (255, 128, 0))
        result = tensor(image)
        self.assertEqual(result.shape, (1, 3, 48, 320))
        self.assertEqual(result.dtype, np.float32)
        # ceil(3*48/5)=29; BGR reverses red and blue.
        np.testing.assert_allclose(result[0, :, 0, 0], [-1, 128*2/255-1, 1], atol=1e-7)
        self.assertTrue(np.all(result[:, :, :, 29:] == 0))


if __name__ == '__main__':
    unittest.main()
