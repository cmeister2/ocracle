#!/usr/bin/env python3
"""Verify recorded selected-pilot export evidence using only the standard library."""
import hashlib
import json
import math
from pathlib import Path

from mtg_report_check import HERE, ROOT, PINS, SELECTED_SHA, SOURCE_SHA, digest, require

REPORT_SHA = '6c0feaa8fa47496cf5662bd9b159de73aa64c9b677e6c20c106f9b8ac7110b51'
ONNX_SHA = 'f551ae37ba046b507a53f2c6b3a1f82077056815e68be55ebb9f7f78157cb09e'
CLI_SHA = '1ede5e7d7a8c2060114ab471adf040d965d0e697eb55f8d8eb094a66b6b28e13'


def check_export():
    require(digest(HERE/'mtg-export.json') == REPORT_SHA, 'Frozen selected export evidence changed')
    report = json.loads((HERE/'mtg-export.json').read_text())
    require(report['status'] == 'selected_mtg_native_onnx_rten_cli_parity_passed' and report['selected_step'] == 25 and
            report['selected_checkpoint_sha256'] == SELECTED_SHA and report['source_checkpoint_sha256'] == SOURCE_SHA,
            'Wrong exported checkpoint/source')
    require(report['pilot_report_sha256'] == PINS['mtg-pilot.json'] and
            report['dataset_manifest_sha256'] == PINS['mtg-dataset.jsonl'] and
            report['dataset_summary_sha256'] == PINS['mtg-data.json'], 'Export dataset/selection binding differs')
    require(report['implementation_sha256'] == digest(HERE/'mtg_export.py'), 'Exporter implementation changed')
    for name, expected in report['frozen_feasibility_files_sha256'].items():
        require(expected == PINS[name] == digest(HERE/name), 'Original feasibility artifact changed')
    require(report['cli_sha256'] == CLI_SHA and report['rten_version'] == '0.26.0', 'Wrong RTen/CLI contract')
    require(report['runner_source_sha256'] == digest(ROOT/'training/experimental_paddle/runner/src/main.rs') and
            report['runner_lock_sha256'] == digest(ROOT/'training/experimental_paddle/runner/Cargo.lock'), 'RTen runner source changed')
    graph = report['onnx']
    require(graph['sha256'] == ONNX_SHA and graph['classes'] == 18713 and graph['opset'] == 17 and
            graph['dynamic_batch'] is True and graph['dynamic_width'] is True and graph['bytes'] == 21331267,
            'Dynamic-width/class tensor contract differs')
    path = HERE/graph['path']
    cache_checked, cache_absent = 0, 0

    def cache(path, expected):
        nonlocal cache_checked, cache_absent
        if path.exists():
            require(digest(path) == expected, 'Present export cache changed: '+str(path))
            cache_checked += 1
        else:
            cache_absent += 1

    cache(path, ONNX_SHA)
    original = json.loads((ROOT/'models/paddle-v6/en.json').read_text())['alphabet']
    characters = original+'¡„ẞ'
    require(len(characters)+1 == 18713 and hashlib.sha256(characters.encode()).hexdigest() == report['alphabet_sha256'],
            'Export dictionary order differs')
    require(set(report['configurations']) == {'en', 'de', 'es', 'fr', 'it', 'pt'}, 'Wrong exported language configs')
    for language, entry in report['configurations'].items():
        config = {'alphabet': characters, 'input_height': 48,
                  'model_name': 'PP-OCRv6 small MTG pilot step25; experimental', 'language': language,
                  'architecture': 'LCNetV4 + LightSVTR; validation-selected MTG pilot',
                  'output_layout': 'NTC', 'output_values': 'probabilities',
                  'input': {'color_mode': 'bgr', 'scale': 2/255, 'offset': -1, 'padding_width': 320,
                            'width_rounding': 'ceil', 'min_content_width': 1}}
        encoded = (json.dumps(config, ensure_ascii=False, indent=2)+'\n').encode()
        require(hashlib.sha256(encoded).hexdigest() == entry['sha256'], 'Expanded config alphabet/language/input contract differs')
        cache(HERE/entry['path'], entry['sha256'])
    expected_shapes = [([1, 3, 48, 320], [1, 40, 18713]), ([1, 3, 48, 513], [1, 64, 18713]),
                       ([2, 3, 48, 320], [2, 40, 18713]), ([1, 3, 48, 513], [1, 64, 18713]),
                       ([1, 3, 48, 320], [1, 40, 18713])]
    require([r['id'] for r in report['records']] == ['random-width320', 'random-width513', 'random-batch2-width320',
                                                   'clean-anchor', 'blur-expanded'], 'Parity probe cohort differs')
    for record, (input_shape, output_shape) in zip(report['records'], expected_shapes, strict=True):
        require(record['input_shape'] == input_shape and record['output_shape'] == output_shape, 'Parity probe tensor shape differs')
        for key in ['native_onnx_max_absolute_probability_error', 'appended_classes_max_absolute_probability_error']:
            require(math.isfinite(record[key]) and 0 <= record[key] <= 5e-5, 'Native/ORT probability tolerance failed')
        require(record['native_onnx_identical_argmax'] is True, 'Native/ORT argmax differs')
        if input_shape[0] == 1:
            require(math.isfinite(record['onnx_rten_max_absolute_probability_error']) and
                    0 <= record['onnx_rten_max_absolute_probability_error'] <= 5e-5 and
                    record['onnx_rten_identical_argmax'] is True and
                    record['native_text'] == record['onnx_text'] == record['rten_text'], 'RTen tensor/decoder parity failed')
        if 'actual_cli' in record:
            cli = record['actual_cli']
            require(record['cli_matches_native_onnx_rten'] is True and cli['text'] == record['rten_text'] and
                    cli['model']['sha256'] == ONNX_SHA and cli['model']['language'] == 'en', 'Actual CLI parity failed')
            cache(HERE/record['image_path'], record['image_sha256'])
    registry = json.dumps(json.loads((ROOT/'models/profiles.json').read_text()))
    require(ONNX_SHA not in registry and 'mtg-export' not in registry, 'Adapted pilot registered as production')
    return {'status': 'selected_pilot_export_verified', 'native_ort_probes': 5, 'ort_rten_probes': 4,
            'actual_cli_probes': 2, 'export_caches_checked': cache_checked, 'export_caches_absent': cache_absent,
            'cache_policy': 'Absent ignored artifacts allowed; present bytes verified; no inference performed.'}


if __name__ == '__main__':
    print(json.dumps(check_export(), indent=2))
