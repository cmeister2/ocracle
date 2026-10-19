"""Small CPU checks for CTC semantics, shape contracts, splits and corruptions."""
import unittest
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from network import ALPHABET,RecognitionModel,decode
from prepare import EXCLUDED_NAMES,normal,split
from train import batch,degraded


class TrainingContracts(unittest.TestCase):
    def test_ctc_repeated_letters_require_blank(self):
        index=ALPHABET.index('l')+1
        self.assertEqual(decode([index,index,0,index]),'ll')
        self.assertEqual(decode([0,index,index,0]),'l')
        self.assertEqual(decode([0,0]),'')

    def test_all_printings_of_a_card_share_split(self):
        first={'name':'Lightning Bolt','id':'print-one'}
        second={'name':'LIGHTNING BOLT','id':'print-two'}
        self.assertEqual(split(first),split(second))

    def test_frozen_identity_splits_and_gallery_exclusions(self):
        corpus=Path(__file__).resolve().parent/'corpus'
        cards=json.loads((corpus/'cards.json').read_text())
        fixtures=Path(__file__).resolve().parents[1]/'docs/gallery/assets/fixtures.json'
        gallery_names={sample['source']['name'].casefold() for sample in json.loads(fixtures.read_text())['samples']}
        self.assertTrue(gallery_names.issubset(EXCLUDED_NAMES))
        self.assertFalse(gallery_names.intersection(card['name'].casefold() for card in cards))
        assignments=defaultdict(set)
        for card in cards:
            self.assertNotIn(card['name'].casefold(),EXCLUDED_NAMES)
            assignments[card['oracle_id']].add(split(card))
        self.assertTrue(all(len(values)==1 for values in assignments.values()))
        for line in json.loads((corpus/'real-lines.json').read_text()):
            self.assertEqual(line['split'],split({'name':line['card_name']}))
            self.assertNotIn(line['card_name'].casefold(),EXCLUDED_NAMES)

    def test_normalization_is_explicit_and_does_not_leave_smart_punctuation(self):
        self.assertEqual(normal('“Élan”—can’t'),'"Elan"-can\'t')

    def test_degradations_are_reproducible_and_preserve_shape(self):
        pixels=np.full((36,160),240,dtype=np.uint8)
        pixels[6:30,25:28]=10
        pixels[10:14,10:130]=20
        image=Image.fromarray(pixels)
        for kind in ['gaussian','severe','motion','downsample','jpeg','combined']:
            with self.subTest(kind=kind):
                first=np.array(degraded(image,kind,1234))
                second=np.array(degraded(image,kind,1234))
                self.assertEqual(first.shape,pixels.shape)
                self.assertTrue(np.array_equal(first,second))
                self.assertFalse(np.array_equal(first,pixels))

    def test_collation_preserves_ctc_lengths_and_white_padding(self):
        images,lengths=batch([torch.zeros(1,64,80),torch.zeros(1,64,100)])
        self.assertEqual(tuple(images.shape),(2,1,64,100))
        self.assertEqual(lengths.tolist(),[21,26])
        self.assertTrue(torch.all(images[0,:,:,80:]==.5))

    def test_network_accepts_variable_width_and_batch(self):
        torch.set_num_threads(1)
        model=RecognitionModel().eval()
        with torch.inference_mode():
            for width,batch_size in [(64,1),(100,2)]:
                scores=model(torch.zeros(batch_size,1,64,width))
                self.assertEqual(tuple(scores.shape),(width//4+1,batch_size,len(ALPHABET)+1))
                self.assertTrue(torch.allclose(scores.exp().sum(-1),torch.ones_like(scores[:,:,0]),atol=1e-5))


if __name__=='__main__':
    unittest.main()
