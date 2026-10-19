#!/usr/bin/env python3
"""Fetch a versioned MTG corpus and render reproducible regular/italic line data.

Scryfall text and scans retain their original rights. Font files are local open
substitutes, never claimed to be Magic's proprietary typefaces. User-supplied
font directories may be added to represent licensed historic fonts exactly.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'training/data'
EXCLUDED_NAMES = {'llanowar elves', 'air elemental', 'divination', 'typhoid rats', 'blade of the sixth pride', 'mana tithe', 'thalia, guardian of thraben', 'daze', 'cave of the frost dragon'}
SETS = ['lea', '7ed', '8ed', 'tsp', 'm15', 'm21']
FONTS = [f'/usr/share/fonts/opentype/urw-base35/{name}.otf' for name in [
    'NimbusRoman-Regular', 'NimbusRoman-Italic', 'P052-Roman', 'P052-Italic',
    'URWBookman-Light', 'URWBookman-LightItalic', 'NimbusSans-Regular', 'NimbusSans-Italic']]
FONT_LICENSE = 'GNU AGPL-3.0 with font embedding exception; URW Base35'


def request(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'ocracle/0.1 (MTG OCR research)', 'Accept': 'application/json'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('request retries exhausted')


def normal(text):
    text = text.translate(str.maketrans({'—':'-', '–':'-', '−':'-', '’':"'", '‘':"'", '“':'"', '”':'"', '•':'*', '…':'...'}))
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().strip()


def split(card):
    # Same name across printings AND all derived crops/fonts/degradations stay together.
    bucket = int(hashlib.sha256(card['name'].casefold().encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket == 0 else ('validation' if bucket == 1 else 'train')


def get_cards(count):
    destination = DATA / 'cards.json'
    snapshot = ROOT / 'training/corpus/cards.json'
    if snapshot.exists() and not destination.exists():
        destination.write_bytes(snapshot.read_bytes())
    if destination.exists():
        return json.loads(destination.read_text())
    cards = []
    for set_code in SETS:
        query = urllib.parse.urlencode({'q': f'set:{set_code} lang:en -is:fullart -is:dfc', 'order': 'name', 'unique':'prints'})
        listing = json.loads(request('https://api.scryfall.com/cards/search?' + query))['data']
        eligible = [c for c in listing if c['name'].casefold() not in EXCLUDED_NAMES and 'image_uris' in c]
        # Spread samples over the available names rather than selecting one color.
        for index in np.linspace(0, len(eligible) - 1, count, dtype=int):
            card = eligible[index]
            cards.append({k: card.get(k, '') for k in ['id','oracle_id','name','set','collector_number','released_at','type_line','oracle_text','flavor_text','image_uris','scryfall_uri','frame']})
        time.sleep(.12)
    destination.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + '\n')
    return cards


def render(text, font_path, rng):
    font = ImageFont.truetype(font_path, size=rng.choice([27, 30, 32, 35]))
    bounds = font.getbbox(text)
    image = Image.new('L', (bounds[2]-bounds[0]+12, bounds[3]-bounds[1]+12), rng.randint(222,255))
    ImageDraw.Draw(image).text((6-bounds[0], 6-bounds[1]), text, font=font, fill=rng.randint(0,40))
    return image


def chunks(text):
    # Never manufacture labels for mana or tap glyphs using brace spelling.
    text = re.sub(r'\{[^}]+\}', ' ', normal(text))
    result, current = [], ''
    for word in text.split():
        if len(current) + len(word) > 44:
            if current:
                result.append(current)
            current = word
        else:
            current = (current + ' ' + word).strip()
    if current:
        result.append(current)
    return [x for x in result if 4 <= len(x) <= 50]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards-per-set',type=int,default=16)
    parser.add_argument('--font-dir',type=Path,help='Additional locally licensed .ttf/.otf fonts')
    parser.add_argument('--real-crops',action='store_true',help='Also acquire scans and bootstrap exact-match title crops')
    args=parser.parse_args()
    DATA.mkdir(exist_ok=True)
    (DATA/'images').mkdir(exist_ok=True)
    (DATA/'scans').mkdir(exist_ok=True)
    cards=get_cards(args.cards_per_set)
    fonts=[Path(x) for x in FONTS if Path(x).exists()]
    if args.font_dir:
        fonts += sorted(p for p in args.font_dir.iterdir() if p.suffix.lower() in ['.ttf','.otf'])
    if not fonts:
        raise SystemExit('Install fonts-urw-base35 or supply --font-dir')
    rng=random.Random(191993)
    records=[]
    for card in cards:
        sources=[('title',normal(card['name'])),('type',normal(card['type_line']))]
        sources += [('rules',x) for x in chunks(card['oracle_text'])]
        sources += [('flavor',x) for x in chunks(card['flavor_text'])]
        for kind,text in sources:
            if not text or len(text)>64:
                continue
            for face in fonts:
                # Flavor/reminder includes italic, but all texts also appear in
                # both styles to prevent network learning style from vocabulary.
                identifier=hashlib.sha256(f"{card['id']}|{kind}|{text}|{face.name}".encode()).hexdigest()[:20]
                path=DATA/'images'/f'{identifier}.png'
                render(text,str(face),rng).save(path)
                records.append({'path':str(path.relative_to(DATA)),'text':text,'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'split':split(card),'source':'synthetic','kind':kind,'font':face.name,'italic':'Italic' in face.name or 'Oblique' in face.name,'set':card['set']})
    if args.real_crops:
        from bootstrap_scans import scan_records
        records += scan_records(cards,DATA)
    (DATA/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    stats={s:sum(r['split']==s for r in records) for s in ['train','validation','test']}
    font_metadata=[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'license':FONT_LICENSE if 'urw-base35' in str(f) else 'user supplied; verify local license'} for f in fonts]
    (DATA/'provenance.json').write_text(json.dumps({'seed':191993,'cards':len(cards),'samples':stats,'excluded_gallery_names':sorted(EXCLUDED_NAMES),'fonts':font_metadata,'data_rights':'Scryfall API card text and scans; Wizards of the Coast card content, used for noncommercial fan OCR research. Not relicensed as crate source.','normalization':'English Latin; curly quotes/dashes and accented letters normalized to pretrained alphabet. Mana glyphs excluded from synthetic labels.','split_policy':'sha256(casefold(card name)) modulo10:0=test,1=validation,2..9=train. All printings/fonts/crops/augmentations grouped.'},indent=2)+'\n')
    print(json.dumps({'cards':len(cards),'samples':stats},indent=2),flush=True)


if __name__=='__main__':
    main()
