#!/usr/bin/env python3
"""Prepare reviewed RGB MTG crops and exact open-font training-only practice.

No model inference occurs here. Existing feasibility scripts/reports are never
written. Existing frozen test transcriptions never enter the prepared corpus.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import unicodedata
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20261021
MANIFEST = HERE/'mtg-dataset.jsonl'
SUMMARY = HERE/'mtg-data.json'
OUTPUT = HERE/'runs/mtg-data'
OPTS_ORACLE = '713332c1-5bd8-400f-bfff-c1ca0697a043'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_split(name):
    bucket = int(hashlib.sha256(name.casefold().encode()).hexdigest()[:8], 16)%10
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def reserved():
    names, oracles = {'opt'}, {OPTS_ORACLE}
    paths = ['docs/gallery/assets/fixtures.json', 'training/latin-eval/cards.json',
             'training/style-eval/cards.json', 'training/rare-scripts/cards.json',
             'training/blur-corpus/cards.json']
    for relative in paths:
        value = json.loads((ROOT/relative).read_text())
        rows = value if isinstance(value, list) else value.get('samples', value.get('cards', []))
        for row in rows:
            source = row.get('source', row)
            name = row.get('card_name', row.get('name', source.get('name')))
            oracle = row.get('oracle_id', source.get('oracle_id'))
            if name: names.add(name.casefold())
            if oracle: oracles.add(oracle)
    return names, oracles


def source_scan(record):
    card = record['card_id']
    sha = record.get('scan_sha256', record.get('source_sha256'))
    candidates = [ROOT/f'training/data/latin-v2/scans-v2/{provider}-{card}.img'
                  for provider in ['scryfall', 'gatherer']]
    candidates += [ROOT/f'training/data/latin-v2/replay-scans/{card}.img',
                   ROOT/f'training/data/latin/scans/{card}.jpg',
                   ROOT/f'training/data/scans/{card}.jpg', OUTPUT/f'sources/{sha}.img']
    for path in candidates:
        if path.exists() and digest(path) == sha:
            return path
    url = record.get('image_url', record.get('source_url'))
    # Existing repository model/data fetcher also enforces this pinned digest.
    import sys
    sys.path.insert(0, str(ROOT/'training'))
    from download_models import fetch as download
    download(url, candidates[-1], sha)
    return candidates[-1]


def png_hash(image):
    data = io.BytesIO()
    image.save(data, format='PNG')
    return hashlib.sha256(data.getvalue()).hexdigest()


def physical(record, origin, role, verify_gray=True):
    source = source_scan(record)
    box = record['bbox']
    size = tuple(record.get('source_size', (672, 936)))
    with Image.open(source) as image:
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            raise RuntimeError('Invalid reviewed crop bounds')
        if verify_gray:
            gray = image.convert('L').resize(size, Image.Resampling.BICUBIC).crop(box)
            expected = record.get('crop_sha256', record.get('image_sha256'))
            if expected and png_hash(gray) != expected:
                raise RuntimeError('Reviewed grayscale source/bounds replay failed: '+record['card_id'])
        crop = image.convert('RGB').resize(size, Image.Resampling.BICUBIC).crop(box)
    if origin == 'punctuation' and png_hash(crop) != record['image_sha256']:
        raise RuntimeError('Reviewed punctuation RGB bytes differ')
    identifier = hashlib.sha256(f"{origin}|{record['card_id']}|{box}|{role}".encode()).hexdigest()[:24]
    path = OUTPUT/'images'/f'{identifier}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)
    result = {key: record[key] for key in ['card_id', 'oracle_id', 'card_name', 'split', 'kind', 'italic']}
    result.update(id=identifier, role=role, origin=origin, language=record.get('language', 'en'),
                  path=str(path.relative_to(HERE)), image_sha256=digest(path), image_mode='RGB',
                  bbox=box, source_size=list(size), source_sha256=digest(source),
                  source_url=record.get('image_url', record.get('source_url')),
                  canonical_split=canonical_split(record['card_name']))
    if 'set' in record:
        result['set'] = record['set']
    if role != 'teacher-anchor':
        result['text'] = record['text']
        if unicodedata.normalize('NFC', result['text']) != result['text']:
            raise RuntimeError('Gold label is not literal NFC')
        result['review_status'] = 'accepted'
    else:
        result['label_policy'] = 'KL only; original normalized label intentionally omitted'
    return result


def synthetic(text, font, parent=None):
    face = ImageFont.truetype(font['path'], 28)
    box = face.getbbox(text)
    image = Image.new('RGB', (box[2]-box[0]+18, box[3]-box[1]+16), (239, 234, 220))
    ImageDraw.Draw(image).text((9-box[0], 8-box[1]), text, font=face, fill=(29, 33, 39))
    identifier = hashlib.sha256((text+'|'+font['sha256']).encode()).hexdigest()[:24]
    path = OUTPUT/'synthetic'/f'{identifier}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    result = {'id': identifier, 'role': 'synthetic', 'origin': 'exact-open-font', 'split': 'train',
              'text': text, 'language': parent['language'] if parent else 'mul-latin',
              'path': str(path.relative_to(HERE)), 'image_sha256': digest(path), 'font_sha256': font['sha256'],
              'font': font['path'], 'italic': 'Italic' in font['path'] or 'Oblique' in font['path'],
              'kind': parent['kind'] if parent else 'glyph-practice'}
    if parent:
        result.update({key: parent[key] for key in ['card_name', 'card_id', 'oracle_id', 'canonical_split']})
    return result


def check():
    summary = json.loads(SUMMARY.read_text())
    if digest(MANIFEST) != summary['manifest_sha256']:
        raise RuntimeError('Prepared MTG manifest changed')
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    if len(records) != summary['records']:
        raise RuntimeError('Prepared MTG count differs')
    for relative, expected in summary['input_sha256'].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError('Reviewed source annotation/provenance changed: '+relative)
    for font in summary['fonts']:
        if digest(Path(font['path'])) != font['sha256']:
            raise RuntimeError('Prepared open font changed')
    if digest(Path(__file__)) != summary['implementation_sha256']:
        raise RuntimeError('Preparation implementation changed; replay before training')
    names, oracles = reserved()
    for record in records:
        if digest(HERE/record['path']) != record['image_sha256']:
            raise RuntimeError('Prepared RGB/practice pixels changed')
        if record['role'] == 'teacher-anchor' and 'text' in record:
            raise RuntimeError('Teacher anchor contains an unsuitable normalized gold label')
        if record['role'] != 'validation' and (record.get('card_name', '').casefold() in names or record.get('oracle_id') in oracles):
            raise RuntimeError('A newly reserved evaluation identity entered training')
        if record.get('card_name') and record['origin'] != 'fresh-blur-development':
            if record['canonical_split'] != canonical_split(record['card_name']):
                raise RuntimeError('Canonical card split binding changed')
    if summary.get('selection_ready'):
        validation = [row for row in records if row['role'] == 'validation']
        if len(validation) != 23 or any(row['origin'] != 'v2-title' or row['split'] != 'validation' for row in validation):
            raise RuntimeError('Checkpoint selection must use only the 23 frozen V2 title lines')
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        records, _ = check()
        print('Verified', len(records), 'prepared RGB/practice inputs')
        return
    names, oracles = reserved()
    metadata = ['training/latin-v2-corpus/titles-reviewed.json', 'training/latin-v2-corpus/body-lines.json',
                'training/punctuation-corpus/annotations.json', 'training/corpus/real-lines.json',
                'training/latin-v2-corpus/fonts.json']
    records = []
    selections = [('v2-title', json.loads((ROOT/metadata[0]).read_text())),
                  ('v2-body', json.loads((ROOT/metadata[1]).read_text())),
                  ('punctuation', json.loads((ROOT/metadata[2]).read_text()))]
    for origin, rows in selections:
        for record in rows:
            if record['split'] not in ('train', 'validation') or record.get('review_status') != 'accepted':
                continue
            if record['card_name'].casefold() in names or record['oracle_id'] in oracles:
                if record['card_name'].casefold() != 'opt':
                    raise RuntimeError('Reserved evaluation identity in proposed supervision')
                continue
            if record['split'] != canonical_split(record['card_name']):
                raise RuntimeError('Source canonical split changed')
            role = 'supervised' if record['split'] == 'train' else 'validation'
            records.append(physical(record, origin, role, origin != 'punctuation'))
    counts = Counter((r['origin'], r['role']) for r in records)
    expected = {('v2-title', 'supervised'): 58, ('v2-title', 'validation'): 23,
                ('v2-body', 'supervised'): 20, ('punctuation', 'supervised'): 3}
    if dict(counts) != expected:
        raise RuntimeError('Expected exactly the reviewed 81 train and 23 validation physical lines: '+str(counts))
    for record in json.loads((ROOT/metadata[3]).read_text()):
        if record['split'] != 'train' or record['card_name'].casefold() in names or record['oracle_id'] in oracles:
            continue
        records.append(physical(record, 'english-real-scan', 'teacher-anchor', False))
    selected_fonts = ['DejaVuSerif.ttf', 'DejaVuSerif-Italic.ttf', 'DejaVuSans.ttf', 'DejaVuSans-Oblique.ttf']
    fonts = [font for font in json.loads((ROOT/metadata[4]).read_text()) if Path(font['path']).name in selected_fonts]
    if len(fonts) != 4:
        raise RuntimeError('Expected four pinned regular/italic open fonts')
    practice = ['¡Salta! „Ähre“ ẞ', 'À Á Â Ã Ä Å Æ Ç È É Ê Ë Ì Í Î Ï',
                'à á â ã ä å æ ç è é ê ë ì í î ï', 'Ñ Ò Ó Ô Õ Ö Ø Ù Ú Û Ü Ý Œ',
                'ñ ò ó ô õ ö ø ù ú û ü ý ÿ œ ß', '‘ ’ “ ” „ ¡ ¿ – — . , : ; ! ?']
    for font in fonts:
        if digest(Path(font['path'])) != font['sha256']:
            raise RuntimeError('Open font changed')
        coverage = TTFont(font['path']).getBestCmap()
        for parent in [r for r in records if r['role'] == 'supervised']:
            if all(ord(char) in coverage for char in parent['text']):
                records.append(synthetic(parent['text'], font, parent))
        for text in practice:
            if not all(ord(char) in coverage for char in text):
                raise RuntimeError('Glyph curriculum would render missing-font boxes')
            records.append(synthetic(text, font))
    roles = defaultdict(set)
    for row in records:
        if row.get('oracle_id'): roles[row['oracle_id']].add('validation' if row['role'] == 'validation' else 'train')
    if any(len(value) != 1 for value in roles.values()):
        raise RuntimeError('A canonical card identity crosses training/selection roles')
    records.sort(key=lambda row: (row['role'], row['id']))
    MANIFEST.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in records))
    summary = {'schema_version': 1, 'seed': SEED, 'records': len(records), 'manifest_sha256': digest(MANIFEST),
               'roles': dict(Counter(row['role'] for row in records)),
               'supervised_origins': dict(counts), 'fonts': fonts,
               'input_sha256': {relative: digest(ROOT/relative) for relative in metadata},
               'reserved_names': sorted(names), 'reserved_oracle_ids': sorted(oracles),
               'fresh_validation': None, 'selection_ready': True,
               'selection_protocol': {'identity_lines': 23, 'observations': 92,
                   'conditions': {'clean': {}, 'gaussian': {'sigma_source_pixels': 2.4},
                                  'downsample': {'factor': .3, 'restore': 'PIL bilinear'},
                                  'jpeg': {'quality': 35, 'subsampling': 2}},
                   'score': '0.90*CER across92 title variants +0.10*decoded disagreement rate on8 fixed clean English training anchors',
                   'checkpoint_steps': [0, 25, 50, 75, 100], 'tie': 'keep earliest'},
               'policy': '81 reviewed physical train crops; RGB source recrops; original canonical splits; English real scans KL only; fresh corpus reserved for validation only.',
               'implementation_sha256': digest(Path(__file__))}
    summary['supervised_origins'] = {origin+':'+role: count for (origin, role), count in counts.items()}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    check()
    print(json.dumps({key: summary[key] for key in ['records', 'roles', 'manifest_sha256', 'selection_ready']}, indent=2))


if __name__ == '__main__':
    main()
