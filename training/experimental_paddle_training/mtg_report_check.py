#!/usr/bin/env python3
"""Check the frozen native MTG pilot offline, without importing ML packages.

Ignored model/pixel caches are optional. Present caches must match; use
--require-caches on the original experiment machine to require all of them.
This verifies recorded evidence, not a new training or inference replay.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = {
    'mtg-pilot.json': '16a7f8091ed380dbabdae1e3e686381291aa1ea80a698d02a7f92e5522d96eff',
    'mtg-dataset.jsonl': 'f64cf8216f3f2ee917c6fce2dfbe807a3e2be5fda00b26f41736d2ae2507f695',
    'mtg-data.json': 'b7961ec2d9a7dbfd2c674f74748a8538c47cfd2a2811b8f0f60888730b8a5dd9',
    'source-files.json': 'f787051b59e91c9ccdb2843f19e1733926efa8056a6738a15d386d2f93865787',
    'source-checkpoint.json': 'c55a85fd224a12e99f4697a09487df71c71d78c6afe05a78e71fd8ab9d6952f2',
    'export.py': 'd10471d3672782c9faf0b4d75e2ddbbb09ab46ca34f8f858638edc590a382ba5',
    'export.json': 'dd6d91348e7e7b07f2b45137356fb93f7ae27d81effb883574128be29833bda4',
    'smoke.py': '45423c8a74a0347bd9195938efe56dce94e87f2d29646335ebcba24a1d60963d',
    'smoke.json': '72d5711f6ea38a330f9fc4f50cb22c4b62eb012a61322202444c4b917d6da045',
    'parity.py': '63dfd6ddf2afbf115c1ad65a1412621223f75eb81edbe3714867089bee9009e4',
    'parity.json': 'f5d60676e4949760dd17b765e5f85986012e83ab25394b92198c689666e7453f',
}
SOURCE_SHA = '25c9bd54b0e5900916e8bb6ada938abeffb1eac1baedac0ca54a45b1c9310825'
SELECTED_SHA = '451b0e2ada65195045037b3d1db2ad60403e41599691e093415dfa3c2c1625a4'
ROLES = {'supervised': 81, 'synthetic': 348, 'teacher-anchor': 179, 'validation': 23}
CONDITIONS = ['clean', 'gaussian', 'downsample', 'jpeg', 'motion', 'combined']
STEPS = [0, 25, 50, 75, 100]
EVIDENCE_MANIFEST_SHA = 'e4d5b1f625b38093f87148218412b2dc7d1b3d705bb212b7e44e2d5c3e008480'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def close(left, right, message):
    require(math.isfinite(left) and math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9), message)


def relative(base, name):
    path = (base/name).resolve()
    require(path.is_relative_to(base.resolve()), 'Path escapes experiment/repository')
    return path


def distance(left, right):
    row = list(range(len(right)+1))
    for i, char in enumerate(left, 1):
        following = [i]
        for j, other in enumerate(right, 1):
            following.append(min(row[j]+1, following[-1]+1, row[j-1]+(char != other)))
        row = following
    return row[-1]


def split(name):
    bucket = int(hashlib.sha256(name.casefold().encode()).hexdigest()[:8], 16)%10
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def validate_dataset(records, summary):
    require(len(records) == summary['records'] == 631, 'Dataset record count differs')
    require(dict(Counter(r['role'] for r in records)) == summary['roles'] == ROLES, 'Dataset roles differ')
    by_id = {r['id']: r for r in records}
    require(len(by_id) == len(records), 'Duplicate dataset ID')
    identities = {}
    for row in records:
        relative(HERE, row['path'])
        sha(row['image_sha256'])
        require(row['role'] != 'teacher-anchor' or 'text' not in row, 'Teacher-only anchor contains gold text')
        if row['role'] != 'teacher-anchor':
            text = row['text']
            require(bool(text.strip()) and unicodedata.normalize('NFC', text) == text, 'Non-NFC/empty gold label')
        if row.get('card_name'):
            expected = split(row['card_name'])
            require(row['canonical_split'] == expected == row['split'], 'Canonical identity split differs')
            require(expected == ('validation' if row['role'] == 'validation' else 'train'), 'Wrong role partition')
            prior = identities.setdefault(row['oracle_id'], expected)
            require(prior == expected, 'Oracle identity spans partitions')
            if row['role'] != 'validation':
                require(row['card_name'].casefold() not in summary['reserved_names'] and
                        row['oracle_id'] not in summary['reserved_oracle_ids'], 'Reserved identity entered training')
        if row['role'] in ('supervised', 'validation'):
            require(row['review_status'] == 'accepted' and row['image_mode'] == 'RGB', 'Unreviewed/non-RGB gold crop')
        if row['role'] != 'synthetic':
            sha(row['source_sha256'])
        else:
            require(row['font_sha256'] in {f['sha256'] for f in summary['fonts']}, 'Unpinned synthetic font')
    physical = Counter(r['origin'] for r in records if r['role'] == 'supervised')
    require(physical == {'v2-title': 58, 'v2-body': 20, 'punctuation': 3}, 'Reviewed physical composition differs')
    validation = [r for r in records if r['role'] == 'validation']
    require(all(r['origin'] == 'v2-title' for r in validation) and
            len({r['oracle_id'] for r in validation}) == 21, 'Selection validation identities differ')
    require(summary['fresh_validation'] is None and summary['selection_ready'] is True,
            'Fresh development corpus entered checkpoint selection')
    return by_id


def evaluated(value, by_id, tensor_bindings):
    expected_pairs = {(r['id'], c) for r in by_id.values() if r['role'] == 'validation'
                      for c in CONDITIONS[:4]}
    seen, groups = set(), {}
    for example in value['examples']:
        pair = (example['id'], example['condition'])
        require(pair in expected_pairs and pair not in seen, 'Validation duplicate/missing/unexpected condition')
        seen.add(pair)
        row = by_id[example['id']]
        require(example['target'] == row['text'], 'Validation target differs from frozen reviewed text')
        signature = sha(example['tensor_sha256'])
        require(tensor_bindings.setdefault(pair, signature) == signature, 'Validation tensor changed between checkpoints')
        target, prediction = example['target'], example['prediction']
        require(prediction == prediction.strip(), 'Recorded prediction normalization differs')
        edits = distance(target, prediction)
        for key in ['all', row['origin'], 'language:'+row['language'], 'condition:'+example['condition'],
                    'italic:'+str(row['italic']).lower()]:
            group = groups.setdefault(key, {'lines': 0, 'characters': 0, 'edits': 0, 'exact': 0})
            group['lines'] += 1
            group['characters'] += len(target)
            group['edits'] += edits
            group['exact'] += target == prediction
    require(seen == expected_pairs, 'Incomplete 92-case validation')
    for group in groups.values():
        group['cer'] = group['edits']/group['characters']
    require(value['groups'] == groups, 'Recorded validation CER/count groups differ from predictions')
    require(value['normalization'] == 'Literal NFC labels; complete CTC output; strip predicted boundary whitespace only.',
            'Validation normalization differs')
    return groups['v2-title']['cer']


def preservation(value, by_id, bindings):
    ids = sorted(r['id'] for r in by_id.values() if r['role'] == 'teacher-anchor')[:8]
    require([r['id'] for r in value['examples']] == ids and value['training_anchor_lines'] == 8,
            'Clean training anchor cohort differs')
    for row in value['examples']:
        signature = (sha(row['tensor_sha256']), row['teacher_text'])
        require(bindings.setdefault(row['id'], signature) == signature, 'Clean teacher/tensor changed')
        require(row['same_text'] == (row['teacher_text'] == row['student_text']), 'Anchor disagreement flag differs')
        require(math.isfinite(row['teacher_kl']) and row['teacher_kl'] >= -1e-6, 'Invalid teacher KL')
    count = sum(r['same_text'] for r in value['examples'])
    require(value['same_teacher_text'] == count, 'Anchor aggregate differs')
    close(value['mean_teacher_kl'], sum(r['teacher_kl'] for r in value['examples'])/8, 'Mean KL differs')
    return 1-count/8


def augmentation(condition, seed):
    rng = random.Random(seed)
    result = {'condition': condition, 'seed': seed}
    if condition in ('gaussian', 'combined'):
        result['sigma_source_pixels'] = rng.uniform(1, 2.8)
    if condition in ('downsample', 'combined'):
        result['downsample_factor'] = rng.uniform(.35, .65)
    if condition == 'motion':
        result.update(motion_length=rng.choice([5, 9, 13]), motion_angle_radians=rng.uniform(-math.pi/3, math.pi/3))
    if condition in ('jpeg', 'combined'):
        result['jpeg_quality'] = rng.randint(15, 55)
    return result


def validate_report(report, records, summary):
    by_id = validate_dataset(records, summary)
    require(report['schema_version'] == 1 and report['mode'] == 'pilot' and report['steps'] == 100 and
            report['seed'] == summary['seed'] == 20261021, 'Wrong pilot mode/seed/update count')
    require(report['source_checkpoint_sha256'] == SOURCE_SHA and report['dataset_roles'] == ROLES,
            'Source model or dataset roles differ')
    require(report['dataset_manifest_sha256'] == summary['manifest_sha256'] == PINS['mtg-dataset.jsonl'] and
            report['dataset_summary_sha256'] == PINS['mtg-data.json'], 'Pilot data binding differs')
    require(report['selection_protocol'] == summary['selection_protocol'] and
            report['selection_protocol']['checkpoint_steps'] == STEPS and
            report['selection_protocol']['tie'] == 'keep earliest', 'Selection protocol differs')
    require(report['validation_inference_performed'] is True and report['test_inference_performed'] is False,
            'Pilot test/validation declaration differs')
    require(report['classes'] == 18713 and
            [(r['character'], r['class_index']) for r in report['appended_classes']] ==
            [('¡', 18710), ('„', 18711), ('ẞ', 18712)], 'Vocabulary indices changed')
    require(report['batch_norm_state_tensors_unchanged'] == 116 and
            all(math.isfinite(v) and v > 0 for v in report['parameter_l2_change'].values()),
            'Frozen BN or parameter update evidence differs')
    require((report['head_lr'], report['neck_lr'], report['upper_backbone_lr']) == (1e-4, 1e-5, 5e-6), 'Learning rates differ')
    require(report['objective'] == 'physical CTC +0.25 synthetic CTC +1.0 clean old-teacher KL in full expanded softmax',
            'Training objective differs')
    pools = {origin: [r for r in records if r['role'] == 'supervised' and r['origin'] == origin]
             for origin in ['v2-title', 'v2-body', 'punctuation']}
    synthetic = [r for r in records if r['role'] == 'synthetic']
    glyphs = [r for r in synthetic if r['kind'] == 'glyph-practice']
    anchors = [r for r in records if r['role'] == 'teacher-anchor']
    rng, anchor_tensors = random.Random(report['seed']), {}
    require(len(report['history']) == 100, 'Incomplete 100-update history')
    for step, row in enumerate(report['history']):
        origin = rng.choices(list(pools), weights=[.35, .4, .25])[0]
        physical = rng.choice(pools[origin])
        condition = rng.choices(CONDITIONS, weights=[.25, .2, .15, .1, .15, .15])[0]
        practice = rng.choice(glyphs if step%2 == 0 else synthetic)
        anchor = rng.choice(anchors)
        parameters = augmentation(condition, rng.randrange(2**32))
        synthetic_parameters = augmentation(rng.choice(CONDITIONS[:3]), rng.randrange(2**32))
        require((row['step'], row['physical_id'], row['physical_origin'], row['physical_text'],
                 row['synthetic_id'], row['anchor_id']) ==
                (step+1, physical['id'], origin, physical['text'], practice['id'], anchor['id']),
                'Seeded curriculum/identity replay differs at step '+str(step+1))
        require(row['augmentation'] == parameters and row['synthetic_augmentation'] == synthetic_parameters,
                'Seeded augmentation replay differs')
        for key in ['physical_tensor_sha256', 'synthetic_tensor_sha256', 'anchor_tensor_sha256']:
            sha(row[key])
        require(anchor_tensors.setdefault(anchor['id'], row['anchor_tensor_sha256']) == row['anchor_tensor_sha256'],
                'Clean teacher anchor tensor changed during training')
        for key in ['physical_ctc', 'synthetic_ctc', 'teacher_kl', 'loss', 'update_seconds']:
            require(math.isfinite(row[key]) and row[key] >= -1e-6, 'Invalid update loss/timing')
        require(set(row['gradient_l2_before_clipping']) == {'head', 'neck', 'upper_backbone'} and
                all(math.isfinite(v) and v > 0 for v in row['gradient_l2_before_clipping'].values()), 'Missing gradient')
        require(math.isclose(row['loss'], row['physical_ctc']+.25*row['synthetic_ctc']+row['teacher_kl'],
                             rel_tol=2e-6, abs_tol=2e-6), 'Recorded objective is inconsistent')
    require(report['update_seconds'] == [r['update_seconds'] for r in report['history']], 'Update timing list differs')
    close(report['estimated_100_update_compute_seconds_excluding_selection'], sum(report['update_seconds']), 'Compute estimate differs')
    require(report['training_seconds'] >= sum(report['update_seconds']), 'Total timing omits update computation')
    tensors, anchor_bindings = {}, {}
    evaluated(report['baseline_validation']['official_source'], by_id, tensors)
    evaluated(report['baseline_validation']['expanded_source'], by_id, tensors)
    preservation(report['before_clean_training_anchors'], by_id, anchor_bindings)
    require([r['step'] for r in report['selection_history']] == STEPS, 'Five scheduled selections missing')
    for row in report['selection_history']:
        cer = evaluated(row['validation'], by_id, tensors)
        disagreement = preservation(row['clean_anchor_preservation'], by_id, anchor_bindings)
        close(row['score'], .9*cer+.1*disagreement, 'Predeclared selection score differs')
    selected = min(report['selection_history'], key=lambda r: (r['score'], r['step']))
    require(selected['step'] == report['selected_step'] == 25, 'Wrong validation-selected checkpoint')
    require(report['selection_history'][0]['validation'] == report['baseline_validation']['expanded_source'], 'Step0 differs')
    require(report['after_clean_training_anchors'] == selected['clean_anchor_preservation'], 'Final weights not selected state')
    require([r['step'] for r in report['saved_checkpoints']] == STEPS, 'Checkpoint inventory differs')
    for row in report['saved_checkpoints']:
        sha(row['sha256']); relative(HERE, row['path'])
    require(report['checkpoint']['sha256'] == SELECTED_SHA == report['saved_checkpoints'][1]['sha256'] and
            report['checkpoint']['distributed'] is False, 'Selected native weight binding differs')
    for key in ['source_checkpoint_sha256', 'dataset_manifest_sha256', 'dataset_summary_sha256', 'selection_protocol', 'steps', 'mode']:
        require(report['process'][key] == report[key], 'Original process binding differs: '+key)


def check(require_caches=False):
    for name, expected in PINS.items():
        require(digest(HERE/name) == expected, 'Frozen committed bytes changed: '+name)
    report = json.loads((HERE/'mtg-pilot.json').read_text())
    summary = json.loads((HERE/'mtg-data.json').read_text())
    records = [json.loads(line) for line in (HERE/'mtg-dataset.jsonl').read_text().splitlines()]
    validate_report(report, records, summary)
    require(digest(HERE/'mtg_prepare.py') == summary['implementation_sha256'], 'Preparation implementation changed')
    for name, expected in report['implementation_sha256'].items():
        require(digest(relative(HERE, name)) == sha(expected), 'Training implementation changed: '+name)
    for name, expected in summary['input_sha256'].items():
        require(digest(relative(ROOT, name)) == sha(expected), 'Reviewed source annotation changed: '+name)
    # Current holdouts include nested metadata records in style/rare corpora;
    # check their canonical identities as well as the original frozen list.
    names, oracles = {'opt'}, {'713332c1-5bd8-400f-bfff-c1ca0697a043'}
    for name in ['docs/gallery/assets/fixtures.json', 'training/latin-eval/cards.json',
                 'training/style-eval/cards.json', 'training/rare-scripts/cards.json', 'training/blur-corpus/cards.json']:
        value = json.loads((ROOT/name).read_text())
        rows = value if isinstance(value, list) else value.get('samples', value.get('cards', []))
        for row in rows:
            source = row.get('metadata', row.get('source', row))
            identity = row.get('card_name', row.get('name', source.get('name')))
            oracle = row.get('oracle_id', source.get('oracle_id'))
            if identity:
                names.add(identity.casefold())
            if oracle:
                oracles.add(oracle)
    for row in records:
        require(row.get('card_name', '').casefold() not in names and row.get('oracle_id', '') not in oracles,
                'Current held-out canonical identity entered training/selection')
    source = json.loads((HERE/'source-checkpoint.json').read_text())
    require(source['sha256'] == SOURCE_SHA, 'Official native source checkpoint changed')
    cached = [(relative(HERE, r['path']), r['image_sha256']) for r in records]
    cached += [(relative(HERE, r['path']), r['sha256']) for r in report['saved_checkpoints']+[report['checkpoint']]]
    cached += [(relative(HERE, source['path']), source['sha256'])]
    missing, checked = [], 0
    for path, expected in cached:
        if path.exists():
            require(digest(path) == expected, 'Present ignored cache changed: '+str(path))
            checked += 1
        else:
            missing.append(str(path.relative_to(HERE)))
    require(not require_caches or not missing, 'Required ignored caches are absent')
    registry = json.loads((ROOT/'models/profiles.json').read_text())
    require('mtg-pilot' not in json.dumps(registry) and SELECTED_SHA not in json.dumps(registry), 'Pilot registered as production')
    return {'status': 'frozen_pilot_report_verified', 'updates': 100, 'scheduled_checkpoints': 5, 'selected_step': 25,
            'validation_observations_per_checkpoint': 92, 'pixel_and_weight_caches_checked': checked,
            'ignored_caches_absent': len(missing), 'cache_policy': 'Absent ignored caches allowed; present bytes verified.',
            'replay_scope': 'Seeded identity/augmentation schedule and recorded metrics; no inference or training replay.'}


def check_downstream():
    """Bind all final recorded artifacts without requiring ignored ML caches."""
    path = HERE/'mtg-artifacts.json'
    require(digest(path) == EVIDENCE_MANIFEST_SHA, 'Frozen evidence manifest changed')
    manifest = json.loads(path.read_text())
    for name, expected in manifest['file_sha256'].items():
        require(digest(relative(ROOT, name)) == sha(expected), 'Frozen experiment artifact changed: '+name)
    from mtg_export_check import check_export
    from mtg_assess import check_report
    exported = check_export()
    for cohort in ['development', 'gallery']:
        check_report(cohort)
        report = json.loads((HERE/('mtg-'+cohort+'.json')).read_text())
        correction = report['metadata_correction']
        require(correction['finalizer_sha256'] == digest(HERE/'mtg_finalize.py') and
                correction['original_inference_implementation_sha256'] == report['process']['implementation_sha256'] and
                correction['corrected_packaging_implementation_sha256'] == report['implementation_sha256'],
                'Metadata correction source binding differs')
        stdout = ''.join(r['process']['stdout'] for r in report['records']).encode()
        require(hashlib.sha256(stdout).hexdigest() == correction['raw_stdout_sequence_sha256_unchanged'],
                'Raw stdout sequence changed during metadata correction')
    return {'status': 'all_recorded_mtg_evidence_verified', 'export': exported,
            'assessment_observations': {'development': 448, 'gallery': 50}, 'production_registration': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-caches', action='store_true')
    parser.add_argument('--pilot-only', action='store_true', help='Skip the frozen export and later assessment evidence')
    args = parser.parse_args()
    print(json.dumps(check(args.require_caches), indent=2))
    if not args.pilot_only:
        print(json.dumps(check_downstream(), indent=2))
