#!/usr/bin/env python3
"""One post-selection Rust assessment; resume recorded cases without re-inference.

All corpus cases are declared before inference. Results cannot change the
already frozen checkpoint, recipe, alphabet or default model registry.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import unicodedata

from mtg_report_check import HERE, ROOT, digest, distance, require, check as check_pilot

CLI_SHA = '1ede5e7d7a8c2060114ab471adf040d965d0e697eb55f8d8eb094a66b6b28e13'
SMALL_SHA = '5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634'
FRESH_SHA = 'cea621376a6e6a7667196011837d6c96d61f8299f0ee8fc82b86c4a698aa47b6'
MODES = ['original', 'enhanced', 'selected']


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()


def normalized(value):
    return ' '.join(unicodedata.normalize('NFC', value).split())


def selected(inference):
    candidates = inference['candidates']
    require(len(candidates) == 2 and candidates[0]['method'].startswith('original + ') and
            candidates[1]['method'].startswith('2x Lanczos + contrast stretch + unsharp mask + '), 'Wrong production passes')
    index, result = max(enumerate(candidates), key=lambda item: (item[1]['confidence'], item[0]))
    for key in ['text', 'confidence', 'method', 'lines']:
        require(result[key] == inference[key], 'Selected candidate is not the posterior maximum')
    for candidate in candidates:
        require(math.isfinite(candidate['confidence']) and 0 <= candidate['confidence'] <= 1, 'Invalid posterior score')
    return MODES[index]


def sources(cohort):
    """Return every fixed case and its pre-existing small-model observation."""
    bindings, cases, baselines = {}, [], []
    if cohort == 'development':
        manifest = ROOT/'training/blur-corpus/manifest.jsonl'
        require(digest(manifest) == FRESH_SHA, 'Frozen fresh-development manifest changed')
        paths = [manifest, ROOT/'training/blur-corpus/annotations.json', ROOT/'training/blur-corpus/provenance.json',
                 ROOT/'training/experimental_paddle_candidates/results-blur-development.json']
        report = json.loads(paths[-1].read_text())
        require(report['manifest_sha256'] == FRESH_SHA and report['cli_sha256'] == CLI_SHA, 'Baseline cohort/CLI differs')
        baseline = {(r['base_id'], r['condition']): r for r in report['records'] if r['model'] == 'v6-small'}
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            previous = baseline[(row['base_id'], row['condition'])]
            require(previous['image_sha256'] == row['image_sha256'] and previous['reference'] == row['text'] and
                    previous['model_sha256'] == SMALL_SHA and previous['language'] == row['language'], 'Baseline case differs')
            cases.append({**row, 'id': row['base_id']+'-'+row['condition'],
                          'image': 'training/blur-corpus/'+row['path'], 'reference': row['text'], 'cohort': cohort})
            baselines.append({'id': cases[-1]['id'], 'inference': previous['inference']})
        require(len(cases) == len(baselines) == len(baseline) == 448, 'Must assess all448 fresh observations')
        baseline_clis = [report['cli_sha256']]
    else:
        assets = ROOT/'docs/gallery/assets'
        fixture_path = assets/'fixtures.json'
        latin_path = ROOT/'training/latin-eval/manifest.jsonl'
        paths = [fixture_path, latin_path, assets/'results-paddle.json', assets/'latin/results-paddle.json']
        fixture = json.loads(fixture_path.read_text())
        for sample in fixture['samples']:
            for variant in sample['variants']:
                image = assets/variant['image']
                cases.append({'id': sample['id']+'-'+variant['id'], 'cohort': 'english-gallery',
                              'image': str(image.relative_to(ROOT)), 'reference': sample['ground_truth'],
                              'language': sample['language'], 'image_sha256': digest(image),
                              'condition': 'clean' if variant['id'] == 'clean' else 'severe',
                              'italic': sample['category'] in ['italic', 'flavor', 'reminder'], 'style': sample['font'],
                              'card_id': sample['source']['print_id'], 'card_name': sample['source']['name'],
                              'source': sample['source'], 'transform': variant['transform']})
        require(len(cases) == 30, 'Must retain all30 original English observations')
        for line in latin_path.read_text().splitlines():
            row = json.loads(line)
            cases.append({**row, 'id': 'latin-'+row['card_id']+'-'+row['severity'], 'cohort': 'latin-gallery',
                          'image': 'training/latin-eval/'+row['path'], 'reference': row['text'],
                          'condition': row['severity'], 'style': row['font']})
        require(len(cases) == 50, 'Must retain all20 independent Latin observations')
        observations, baseline_clis = {}, []
        for path in paths[-2:]:
            report = json.loads(path.read_text())
            require(report['recognition_model_sha256'] == SMALL_SHA and report['passes'] == 'all', 'Gallery baseline differs')
            baseline_clis.append(report['executable_sha256'])
            for row in report['records']:
                inference = row['inference']
                key = row['image_sha256'], inference['model']['language']
                require(key not in observations, 'Duplicate baseline image/language')
                observations[key] = inference
        for row in cases:
            baselines.append({'id': row['id'], 'inference': observations[(row['image_sha256'], row['language'])]})
    for path in paths:
        bindings[str(path.relative_to(ROOT))] = digest(path)
    require(len({r['id'] for r in cases}) == len(cases), 'Duplicate case ID')
    training = [json.loads(line) for line in (HERE/'mtg-dataset.jsonl').read_text().splitlines()]
    names = {r.get('card_name', '').casefold() for r in training}
    oracles = {r.get('oracle_id') for r in training if r.get('oracle_id')}
    for row in cases:
        require(row.get('card_name', '').casefold() not in names and row.get('oracle_id', '') not in oracles,
                'Assessment identity overlaps adaptation training/selection')
    return cases, baselines, bindings, sorted(set(baseline_clis))


def metrics(records, baseline):
    groups = defaultdict(lambda: {'observations': 0, 'characters': 0, 'edits': 0, 'exact': 0})
    previous = {r['id']: r['inference'] for r in baseline}
    for row in records:
        target = normalized(row['reference'])
        cohorts = ['all', 'cohort/'+row['cohort'], 'condition/'+row['condition'], 'language/'+row['language'],
                   'italic/'+str(row.get('italic', False)).lower(), 'style/'+row['style']]
        for model, inference in [('v6-small', previous[row['id']]), ('mtg-step25', row['inference'])]:
            for mode, raw in zip(MODES, [inference['candidates'][0]['text'], inference['candidates'][1]['text'], inference['text']]):
                prediction = normalized(raw)
                edits = distance(target, prediction)
                for group in cohorts:
                    counts = groups[model+'/'+mode+'/'+group]
                    counts['observations'] += 1
                    counts['characters'] += len(target)
                    counts['edits'] += edits
                    counts['exact'] += target == prediction
    return {key: {**value, 'cer': value['edits']/value['characters']} for key, value in sorted(groups.items())}


def validate_record(record, case, export):
    for key, value in case.items():
        require(record[key] == value, 'Recorded case/provenance changed: '+key)
    require(record['model_sha256'] == export['onnx']['sha256'] and record['cli_sha256'] == CLI_SHA and
            record['configuration_sha256'] == export['configurations'][case['language']]['sha256'], 'Inference binding differs')
    raw = record['process']
    require(raw['returncode'] == 0 and json.loads(raw['stdout']) == record['inference'], 'Failed/mismatched raw process')
    for key in ['stdout', 'stderr']:
        require(hashlib.sha256(raw[key].encode()).hexdigest() == raw[key+'_sha256'], 'Raw process bytes changed')
    command = raw['command']
    # Commands retain their original absolute bytes. A clean CI checkout has a
    # different root, so bind all arguments to one recorded root plus immutable
    # repository-relative paths rather than this process's current directory.
    image_suffix = '/'+case['image']
    require(len(command) == 10 and command[2].endswith(image_suffix), 'Recorded image argument differs')
    recorded_root = Path(command[2][:-len(image_suffix)])
    model_relative = (HERE/export['onnx']['path']).relative_to(ROOT)
    config_relative = (HERE/export['configurations'][case['language']]['path']).relative_to(ROOT)
    require(command[1:2] == ['recognize'] and command[3] == '--recognition-model' and command[5] == '--model-config' and
            command[7:] == ['--line', '--passes', 'all'] and
            Path(command[4]) == recorded_root/model_relative and Path(command[6]) == recorded_root/config_relative,
            'Recorded CLI invocation differs')
    inference = record['inference']
    require(inference['model']['sha256'] == export['onnx']['sha256'] and inference['model']['language'] == case['language'],
            'Actual CLI model/language differs')
    require(record['selected_pass'] == selected(inference), 'Selected-pass label differs')


def check_report(cohort):
    cases, baseline, bindings, baseline_clis = sources(cohort)
    export = json.loads((HERE/'mtg-export.json').read_text())
    report = json.loads((HERE/('mtg-'+cohort+'.json')).read_text())
    require(report['input_sha256'] == bindings and report['export_report_sha256'] == digest(HERE/'mtg-export.json') and
            report['pilot_report_sha256'] == digest(HERE/'mtg-pilot.json') and report['selected_step'] == 25,
            'Assessment report inputs differ')
    require(report['cli_sha256'] == CLI_SHA and report['baseline_cli_sha256'] == baseline_clis and
            report['implementation_sha256'] == digest(Path(__file__)), 'Assessment implementation/CLI differs')
    require(report['passes'] == 'all' and report['jobs'] == 2 and report['rayon_num_threads'] == 2 and
            report['training_or_checkpoint_retuning_performed'] is False, 'Assessment policy differs')
    require(len(report['records']) == len(cases), 'Missing assessment cases')
    for row, case in zip(report['records'], cases, strict=True):
        validate_record(row, case, export)
        path = ROOT/case['image']
        if path.exists():
            require(digest(path) == case['image_sha256'], 'Assessment input pixels changed')
    require(report['summary'] == metrics(report['records'], baseline), 'Assessment metrics differ from raw predictions')
    print(json.dumps({'status': 'frozen_assessment_verified', 'cohort': cohort, 'observations': len(cases),
                      'summary': {key: value for key, value in report['summary'].items() if key.endswith('/selected/all')}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=['development', 'gallery'], required=True)
    parser.add_argument('--cli', type=Path, default=HERE/'cache/mtg-ocracle-snapshot')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check_report(args.cohort)
        return
    check_pilot()
    cases, baseline, bindings, baseline_clis = sources(args.cohort)
    export = json.loads((HERE/'mtg-export.json').read_text())
    require(digest(args.cli) == CLI_SHA and digest(HERE/export['onnx']['path']) == export['onnx']['sha256'], 'Model/CLI changed')
    for entry in export['configurations'].values():
        require(digest(HERE/entry['path']) == entry['sha256'], 'Expanded configuration changed')
    for case in cases:
        require(digest(ROOT/case['image']) == case['image_sha256'], 'Assessment pixels changed')
    directory = HERE/'runs'/('mtg-assessment-'+args.cohort)
    directory.mkdir(parents=True, exist_ok=True)
    contract = {'input_sha256': bindings, 'cli_sha256': CLI_SHA, 'export_report_sha256': digest(HERE/'mtg-export.json'),
                'implementation_sha256': digest(Path(__file__)), 'cases': [c['id'] for c in cases]}
    contract_path = directory/'contract.json'
    if contract_path.exists():
        require(json.loads(contract_path.read_text()) == contract, 'Refusing to reuse raw observations from a different experiment')
    else:
        contract_path.write_bytes(encoded(contract))
    process = {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(), 'cohort': args.cohort,
               'observations': len(cases), **contract}
    (directory/'process.json').write_bytes(encoded(process))
    print(json.dumps({'event': 'started', **process}), flush=True)

    def infer(case):
        path = directory/(case['id']+'.json')
        if path.exists():
            record = json.loads(path.read_text())
            validate_record(record, case, export)
            return record
        command = [str(args.cli.resolve()), 'recognize', str(ROOT/case['image']), '--recognition-model',
                   str(HERE/export['onnx']['path']), '--model-config',
                   str(HERE/export['configurations'][case['language']]['path']), '--line', '--passes', 'all']
        process_path = directory/(case['id']+'.process.json')
        if process_path.exists():
            raw = json.loads(process_path.read_text())
            require(raw['command'] == command, 'Existing raw process has different arguments')
        else:
            started = time.monotonic()
            result = subprocess.run(command, text=True, capture_output=True, env={**os.environ, 'RAYON_NUM_THREADS': '2'})
            raw = {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
                   'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
                   'stderr_sha256': hashlib.sha256(result.stderr.encode()).hexdigest(), 'wall_seconds': time.monotonic()-started}
            # Preserve failure output too. A failed case will never be silently retried.
            process_path.write_bytes(encoded(raw))
        require(raw['returncode'] == 0, 'Rust assessment failed; inspect recorded process before continuing')
        inference = json.loads(raw['stdout'])
        record = {**case, 'model_sha256': export['onnx']['sha256'], 'cli_sha256': CLI_SHA,
                  'configuration_sha256': export['configurations'][case['language']]['sha256'],
                  'selected_pass': selected(inference), 'process': raw, 'inference': inference}
        validate_record(record, case, export)
        path.write_bytes(encoded(record))
        return record

    started = time.monotonic()
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, record in enumerate(pool.map(infer, cases), 1):
            records.append(record)
            if index%32 == 0:
                print(json.dumps({'completed': index, 'total': len(cases), 'seconds': time.monotonic()-started}), flush=True)
    require(digest(args.cli) == CLI_SHA and digest(HERE/export['onnx']['path']) == export['onnx']['sha256'], 'Inference files changed')
    require(sources(args.cohort)[2] == bindings, 'Cohort changed during assessment')
    report = {'schema_version': 1, 'cohort': args.cohort, 'partition': 'development' if args.cohort == 'development' else 'adaptation-held-out assessment',
              'selected_step': 25, 'pilot_report_sha256': digest(HERE/'mtg-pilot.json'),
              'export_report_sha256': digest(HERE/'mtg-export.json'), 'input_sha256': bindings,
              'cli_sha256': CLI_SHA, 'baseline_cli_sha256': baseline_clis,
              'implementation_sha256': digest(Path(__file__)), 'model_sha256': export['onnx']['sha256'],
              'configurations': export['configurations'], 'jobs': 2, 'rayon_num_threads': 2, 'passes': 'all',
              'metric_policy': 'NFC and collapsed whitespace only; retain case, accents and literal punctuation; character-weighted CER.',
              'selection_policy': 'Production maximum uncalibrated CTC score across original/enhanced; later candidate wins ties.',
              'training_or_checkpoint_retuning_performed': False, 'process': process,
              'assessment_wall_seconds': time.monotonic()-started, 'records': records, 'summary': metrics(records, baseline),
              'limitations': ['One assessment of the already validation-selected checkpoint; no recipe/selection changes.',
                              'Pretrained upstream print identities are unknown; identity holdout is from our adaptation only.',
                              'Fresh corpus is development evidence, not an untouched final test.',
                              'Frozen50 baseline executable differs from the fresh-development snapshot; both hashes are disclosed.'
                              if args.cohort == 'gallery' else 'Exact same executable as the fixed-small baseline.']}
    (HERE/('mtg-'+args.cohort+'.json')).write_bytes(encoded(report))
    check_report(args.cohort)


if __name__ == '__main__':
    main()
