"""Unicode head expansion preserves old visual knowledge without claiming training."""
import tempfile
from pathlib import Path
import unittest
import torch
from network import ALPHABET,RecognitionModel,adapt_model,decode,load_model
from language_head import alphabet_for_cards,normalized_printed


class UnicodeProfiles(unittest.TestCase):
    def test_unicode_normalization_preserves_orthography(self):
        self.assertEqual(normalized_printed('Cafe\u0301 {G} — Ж 日'),'Café — Ж 日')
        cards=[{'id':'localized-print','lang':'fr','printed_name':'Éclair','printed_text':'Inflige 3 blessures à une cible.'}]
        alphabet,lines=alphabet_for_cards(cards,'fr')
        self.assertIn('É',alphabet)
        self.assertIn('à',alphabet)
        self.assertTrue(all(line['language']=='fr' for line in lines))
        self.assertEqual(decode([1,1,0,2],alphabet='日Ж'),'日Ж')

    def test_mixed_languages_and_oracle_fallback_are_rejected(self):
        with self.assertRaises(ValueError):
            alphabet_for_cards([{'id':'wrong','lang':'de','printed_name':'Blitzschlag'}],'fr')
        with self.assertRaises(ValueError):
            alphabet_for_cards([{'id':'unlocalized','lang':'fr','name':'Lightning Bolt','oracle_text':'English Oracle text'}],'fr')

    def test_expansion_preserves_backbone_and_reordered_character_rows(self):
        torch.set_num_threads(1)
        source=RecognitionModel()
        alphabet='日éЖ'+ALPHABET[::-1]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            torch.save({'model_state':source.state_dict()},path)
            expanded=adapt_model(path,alphabet)
            for name,value in source.state_dict().items():
                if not name.startswith('output.0.'):
                    self.assertTrue(torch.equal(value,expanded.state_dict()[name]))
            self.assertTrue(torch.equal(source.output[0].weight[0],expanded.output[0].weight[0]))
            for char in ['A','l','€',' ']:
                self.assertTrue(torch.equal(source.output[0].weight[ALPHABET.index(char)+1],expanded.output[0].weight[alphabet.index(char)+1]))
            torch.save({'model_state':expanded.state_dict(),'alphabet':alphabet},path)
            self.assertEqual(load_model(path).alphabet,alphabet)
            with torch.inference_mode():
                self.assertEqual(tuple(expanded(torch.zeros(1,1,64,100)).shape),(26,1,len(alphabet)+1))


if __name__=='__main__':
    unittest.main()
