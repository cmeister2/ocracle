#!/usr/bin/env python3
"""Reproduce or verify reviewed rare-script crops without substituting Oracle text."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import unicodedata
import urllib.request

from PIL import Image, ImageDraw, ImageFilter, __version__ as pillow_version

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CACHE = REPO / 'training/data/rare-scripts'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def png(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def derivatives(image, box):
    crop = image.crop(box)
    severe = crop.filter(ImageFilter.GaussianBlur(2.4))
    severe = severe.resize(tuple(max(1, round(value * .3)) for value in crop.size), Image.Resampling.BILINEAR)
    return {'clean': crop, 'severe': severe}


def source(card, download):
    path = CACHE / (card['metadata']['id'] + '.jpg')
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'{path}: use --download-sources')
        request = urllib.request.Request(card['image_url'], headers={'User-Agent': 'ocracle-research/0.1'})
        content = urllib.request.urlopen(request, timeout=30).read()
        if digest(content) != card['scan_sha256']:
            raise ValueError('Downloaded source digest changed')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if digest(path.read_bytes()) != card['scan_sha256']:
        raise ValueError(f'Source digest mismatch: {path}')
    image = Image.open(path).convert('RGB')
    if list(image.size) != card['image_size']:
        raise ValueError('Source dimensions changed')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify frozen fixtures without rewriting')
    parser.add_argument('--download-sources', action='store_true', help='Check original scans and reproduce derivatives too')
    args = parser.parse_args()
    cards = json.loads((ROOT / 'cards.json').read_text())
    annotations = json.loads((ROOT / 'annotations.json').read_text())
    by_id = {card['metadata']['id']: card for card in cards}
    names = {card['metadata']['name'].casefold() for card in cards}
    oracles = {card['metadata']['oracle_id'] for card in cards}
    for corpus in ['corpus', 'latin-corpus', 'latin-v2-corpus']:
        path = REPO / 'training' / corpus / 'cards.json'
        if not path.exists():
            continue
        existing = json.loads(path.read_text())
        if names.intersection(card['name'].casefold() for card in existing) or oracles.intersection(card['oracle_id'] for card in existing):
            raise ValueError(f'External evaluation identity leaked into {corpus}')

    expected = {}
    originals = {}
    for annotation in annotations:
        if 'pending' in annotation['review_status']:
            raise ValueError('Annotation still awaiting review')
        if unicodedata.normalize('NFC', annotation['text']) != annotation['text']:
            raise ValueError('Expected text must be NFC without stripping marks')
        card = by_id[annotation['card_id']]
        if annotation['oracle_id'] != card['metadata']['oracle_id'] or annotation['language'] != card['language']:
            raise ValueError('Annotation identity mismatch')
        left, top, right, bottom = annotation['box']
        if not (0 <= left < right <= card['image_size'][0] and 0 <= top < bottom <= card['image_size'][1]):
            raise ValueError('Invalid original-image crop')
        if not args.check or args.download_sources:
            if annotation['card_id'] not in originals:
                originals[annotation['card_id']] = source(card, args.download_sources)
            images = derivatives(originals[annotation['card_id']], annotation['box'])
        else:
            images = None
        for severity in ['clean', 'severe']:
            key = (annotation['id'], severity)
            expected[key] = {'id': annotation['id'], 'path': f'images/{annotation["id"]}-{severity}.png', 'text': annotation['text'], 'language': annotation['language'], 'card_id': annotation['card_id'], 'oracle_id': annotation['oracle_id'], 'card_name': annotation['card_name'], 'role': annotation['role'], 'italic': annotation['italic'], 'severity': severity, 'split': 'external-evaluation', 'source': 'reviewed-real-scan'}
            if images is not None:
                content = png(images[severity])
                expected[key]['image_sha256'] = digest(content)
                expected[key]['dimensions'] = list(images[severity].size)
                if not args.check:
                    path = ROOT / expected[key]['path']
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(content)

    provenance = {'schema_version': 1, 'verified_on': '2026-10-19', 'cards_sha256': digest((ROOT / 'cards.json').read_bytes()), 'annotations_sha256': digest((ROOT / 'annotations.json').read_bytes()), 'pillow_version': pillow_version, 'observations': len(expected), 'source_images': len(cards), 'damage': 'Gaussian sigma2.4 native pixels, then 30% bilinear downsample; no transcript simplification', 'split_policy': 'Explicit singleton/rare-language holdouts override nominal card-name hash partitions; excluded from all project training/selection corpora', 'comparison': 'NFC and collapsed whitespace only; Hebrew FB20 wide ayin mapped explicitly in the annotation to logical U+05E2. No general compatibility folding or mark stripping.', 'rights': 'Wizards of the Coast card imagery/text, Scryfall scans/metadata; not covered by source-code MIT license', 'pretraining_note': 'Generic upstream OCR pretraining membership is unknown; project training exclusion is verified'}
    manifest = ROOT / 'manifest.jsonl'
    if args.check:
        recorded = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len(recorded) != len(expected) or len({(row['id'], row['severity']) for row in recorded}) != len(expected):
            raise ValueError('Missing or duplicate observation')
        for row in recorded:
            reference = expected[(row['id'], row['severity'])]
            if any(row.get(key) != value for key, value in reference.items()):
                raise ValueError(f'Fixture differs from reviewed annotation/source: {row["id"]}')
            path = ROOT / row['path']
            if digest(path.read_bytes()) != row['image_sha256'] or list(Image.open(path).size) != row['dimensions']:
                raise ValueError(f'Fixture digest/dimensions mismatch: {path}')
        frozen = json.loads((ROOT / 'provenance.json').read_text())
        for key in ['cards_sha256', 'annotations_sha256', 'observations']:
            if frozen[key] != provenance[key]:
                raise ValueError(f'Frozen provenance changed: {key}')
    else:
        manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in expected.values()))
        (ROOT / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
        sheet = Image.new('RGB', (750, 100 * len(annotations)), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, annotation in enumerate(annotations):
            draw.text((10, index * 100 + 4), annotation['id'], fill='black')
            sheet.paste(Image.open(ROOT / f'images/{annotation["id"]}-clean.png'), (20, index * 100 + 25))
        sheet.save(ROOT / 'review.png')
    print(f'Verified {len(annotations)} reviewed rare-script crops, {len(expected)} observations, source provenance and project training isolation')


if __name__ == '__main__':
    main()
