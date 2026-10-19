"""Contracts for physical-label review and V2's preservation training policy."""
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from network import RecognitionModel
from latin_v2_train import fixed_bn,selection_score
from latin_v2_acquire import FUTURE_RESERVED,excluded

ROOT=Path(__file__).resolve().parents[1]


class LatinV2Contracts(unittest.TestCase):
    def test_fixed_statistics_keep_trainable_parameters(self):
        model=RecognitionModel()
        fixed_bn(model)
        norms=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
        self.assertTrue(norms)
        self.assertTrue(all(not m.training for m in norms))
        self.assertTrue(model.gru.training)
        self.assertTrue(all(p.requires_grad for p in model.gru.parameters()))

    def test_selection_uses_only_named_validation_groups(self):
        groups={'new-physical':{'cer':.1,'exact_line_accuracy':.5},'english-preservation':{'cer':.02}}
        self.assertAlmostEqual(selection_score({'groups':groups}),.152)

    def test_reviewed_bodies_exclude_known_label_error(self):
        records=json.loads((ROOT/'training/latin-corpus/body-reviewed.json').read_text())
        accepted=[r for r in records if r['review_status']=='accepted']
        self.assertEqual(len(accepted),21)
        self.assertFalse(any(r['proposal_index']==97 for r in accepted))
        self.assertTrue(all(r['italic'] and r['kind']=='reminder' for r in accepted if r['proposal_index'] in [251,252]))

    def test_complete_standard_orthography_inventory(self):
        orth=json.loads((ROOT/'training/latin-v2-corpus/orthography.json').read_text())
        self.assertIn('Ä',orth['de']['letters']);self.assertIn('ẞ',orth['de']['letters'])
        self.assertIn('ï',orth['fr']['letters']);self.assertIn('Ê',orth['pt']['letters'])
        self.assertIn('È',orth['it']['letters']);self.assertIn('ù',orth['it']['letters'])
        self.assertTrue(set('‘’“”–—')<=set(orth['punctuation']['characters']))

    def test_frozen_training_has_positive_required_glyph_coverage(self):
        summary=json.loads((ROOT/'training/latin-v2-corpus/summary.json').read_text())
        for char in 'ÄïÊÈù‘’“”–—':self.assertGreater(summary['training_character_counts'][char],0)
        self.assertIn('Opt',summary['late_holdout_parent_identities'])
        self.assertGreater(summary['late_holdout_parent_rows_excluded'],0)

    def test_new_acquisition_excludes_independent_identities(self):
        summary=json.loads((ROOT/'training/latin-v2-corpus/summary.json').read_text())
        names=set(summary['evaluation_excluded_names']);oracles=set(summary['evaluation_excluded_oracle_ids'])
        cards=json.loads((ROOT/'training/latin-v2-corpus/cards.json').read_text())
        self.assertFalse(any(c['name'].casefold() in names or c['oracle_id'] in oracles for c in cards))

    def test_wrong_language_images_are_rejected(self):
        records=json.loads((ROOT/'training/latin-v2-corpus/titles-reviewed.json').read_text())
        self.assertEqual(len(records),len({r['card_id'] for r in records}))
        self.assertTrue(any(r['review_status']=='rejected' and 'English' in r['review_reason'] for r in records))
        for r in records:
            self.assertIn(r['review_status'],['accepted','rejected'])
            self.assertTrue(r['crop_sha256']);self.assertTrue(r['scan_sha256'])

    def test_future_poster_identities_remain_reserved(self):
        names,oracles=excluded()
        self.assertTrue({n.casefold() for n in FUTURE_RESERVED}<=names)
        self.assertTrue(set(FUTURE_RESERVED.values())<=oracles)


if __name__=='__main__':unittest.main()
