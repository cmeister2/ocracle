#!/usr/bin/env python3
"""Apply explicit metadata-only assessment corrections; never run inference.

The first assessment accidentally mapped category=italic to false. Correct that
group metadata and bind the checkout-portable checker, preserving every raw
command, stdout, stderr and prediction byte. This script is idempotent.
"""
import hashlib
import json
from pathlib import Path

from mtg_assess import HERE, sources, metrics, digest, encoded, require


def finalize(cohort):
    path = HERE/('mtg-'+cohort+'.json')
    report = json.loads(path.read_text())
    raw = ''.join(row['process']['stdout'] for row in report['records']).encode()
    original_sha = hashlib.sha256(raw).hexdigest()
    cases, baseline, _, _ = sources(cohort)
    corrections = report.get('metadata_correction', {}).get('changes', [])[:]
    for record, case in zip(report['records'], cases, strict=True):
        for field, value in case.items():
            if record[field] != value:
                require(field == 'italic', 'Refusing a non-style change to recorded assessment data')
                corrections.append({'id': case['id'], 'field': field, 'before': record[field], 'after': value})
                record[field] = value
    report['summary'] = metrics(report['records'], baseline)
    report['implementation_sha256'] = digest(HERE/'mtg_assess.py')
    require(original_sha == hashlib.sha256(''.join(row['process']['stdout'] for row in report['records']).encode()).hexdigest(),
            'Raw prediction sequence changed')
    report['metadata_correction'] = {
        'reason': 'Use frozen English category=italic for style grouping; make historical-path checking portable. No inference, label, prediction, preprocessing, model, checkpoint or selection changes.',
        'original_inference_implementation_sha256': report['process']['implementation_sha256'],
        'corrected_packaging_implementation_sha256': report['implementation_sha256'],
        'finalizer_sha256': digest(Path(__file__)),
        'raw_stdout_sequence_sha256_unchanged': original_sha, 'changes': corrections}
    path.write_bytes(encoded(report))
    print(json.dumps({'cohort': cohort, 'style_flags_corrected': len(corrections), 'raw_stdout_sequence_sha256': original_sha}))


if __name__ == '__main__':
    finalize('development')
    finalize('gallery')
