"""Label restoration must preserve localized text and reject word fragments."""
import unittest
import json
from pathlib import Path
from latin_bootstrap_body import restored_label


class LocalizedLabelAlignment(unittest.TestCase):
    def test_restore_accents_without_english_oracle_substitution(self):
        self.assertEqual(restored_label('Il gagne la protection contre les creatures.','Il gagne la protection contre les créatures.'),'Il gagne la protection contre les créatures.')
        self.assertEqual(restored_label('Gnadenstoss verhindert diesen Schaden.','Gnadenstoß verhindert diesen Schaden.'),'Gnadenstoß verhindert diesen Schaden.')
        self.assertEqual(restored_label('O cao ganha vigilancia.','O cão ganha vigilância.'),'O cão ganha vigilância.')

    def test_independent_physical_eval_is_disjoint_from_training_sources(self):
        root=Path(__file__).resolve().parents[1]
        manifest=root/'training/latin-eval/manifest.jsonl'
        if not manifest.exists():
            self.skipTest('Independent physical evaluation fixture not installed')
        holdouts=[json.loads(line) for line in manifest.read_text().splitlines()]
        names={record['card_name'].casefold() for record in holdouts}
        oracles={record['oracle_id'] for record in holdouts}
        for path in [root/'training/corpus/cards.json',root/'training/latin-corpus/cards.json']:
            if not path.exists():continue
            cards=json.loads(path.read_text())
            self.assertFalse(names.intersection(card['name'].casefold() for card in cards))
            self.assertFalse(oracles.intersection(card['oracle_id'] for card in cards))

    def test_cropped_word_fragments_are_not_valid_ground_truth(self):
        self.assertIsNone(restored_label('ature gains flying','This creature gains flying.'))
        self.assertIsNone(restored_label('murdered creatures','Return destroyed creatures.'))


if __name__=='__main__':
    unittest.main()
