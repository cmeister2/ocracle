#!/usr/bin/env python3
"""Separate NFC Latin-accent corpus: one actual printed language per card."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.parse
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont
from language_head import normalized_printed
from network import ALPHABET
from prepare import EXCLUDED_NAMES,FONTS,render,request,split

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/data/latin'
CORPUS=ROOT/'training/latin-corpus'
LANGUAGES=['en','fr','de','it','pt','es']


def fetch_cards():
    frozen=CORPUS/'cards.json'
    if frozen.exists():
        return json.loads(frozen.read_text())
    holdouts=set(EXCLUDED_NAMES)
    evaluation=ROOT/'training/latin-eval/manifest.jsonl'
    if evaluation.exists():
        holdouts.update(json.loads(line)['card_name'].casefold() for line in evaluation.read_text().splitlines())
    cards=[]
    for language in LANGUAGES:
        query=urllib.parse.urlencode({'q':f'set:m21 lang:{language} cn<=274 -type:planeswalker -is:fullart -is:promo -is:dfc','unique':'prints','order':'name','include_multilingual':'true'})
        candidates=json.loads(request('https://api.scryfall.com/cards/search?'+query))['data']
        eligible=[c for c in candidates if c['name'].casefold() not in holdouts and c.get('image_uris') and (language=='en' or c.get('printed_name'))]
        # Explicitly include training, validation and test identities per language.
        chosen=[]
        for grouping,count in [('train',12),('validation',4),('test',4)]:
            options=[c for c in eligible if split(c)==grouping]
            for index in np.linspace(0,len(options)-1,min(count,len(options)),dtype=int):
                c=options[index]
                keep=['id','oracle_id','name','lang','set','collector_number','printed_name','printed_type_line','printed_text','flavor_text','flavor_name','type_line','oracle_text','mana_cost','image_uris','scryfall_uri']
                cards.append({key:c.get(key,'') for key in keep})
        time.sleep(.12)
    frozen.write_text(json.dumps(cards,ensure_ascii=False,indent=2)+'\n')
    return cards


def wrapped(text):
    text=normalized_printed(text)
    lines=[]
    current=''
    for word in text.split():
        if len(current)+len(word)>42:
            if current:lines.append(current)
            current=word
        else:current=(current+' '+word).strip()
    if current:lines.append(current)
    return [line for line in lines if 4<=len(line)<=64]


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    CORPUS.mkdir(exist_ok=True)
    (DATA/'images').mkdir(exist_ok=True)
    (DATA/'scans').mkdir(exist_ok=True)
    cards=fetch_cards()
    frozen_real_path=CORPUS/'real-lines.json'
    frozen_real={record['card_id']:record for record in json.loads(frozen_real_path.read_text())} if frozen_real_path.exists() else {}
    faces=[Path(path) for path in FONTS]
    if (CORPUS/'fonts.json').exists():
        for font in json.loads((CORPUS/'fonts.json').read_text()):
            if hashlib.sha256(Path(font['path']).read_bytes()).hexdigest()!=font['sha256']:
                raise RuntimeError('Frozen Latin corpus font changed: '+font['path'])
    coverage={str(face):set(TTFont(face).getBestCmap()) for face in faces}
    rng=random.Random(20261019)
    records=[]
    characters=set(ALPHABET)
    missing=defaultdict(set)
    for card in cards:
        language=card['lang']
        assert language in LANGUAGES
        title=card.get('flavor_name') or card.get('printed_name') or (card['name'] if language=='en' else '')
        type_line=card.get('printed_type_line') or (card['type_line'] if language=='en' else '')
        rules=card.get('printed_text') or (card['oracle_text'] if language=='en' else '')
        sources=[('title',normalized_printed(title)),('type',normalized_printed(type_line))]
        sources += [('rules',line) for line in wrapped(rules)]
        sources += [('flavor',line) for line in wrapped(card.get('flavor_text',''))]
        for kind,text in sources:
            if not text:continue
            characters.update(text)
            for face in faces:
                absent={c for c in text if ord(c) not in coverage[str(face)]}
                if absent:
                    missing[face.name].update(absent)
                    continue
                identifier=hashlib.sha256(f"{card['id']}|{kind}|{text}|{face.name}".encode()).hexdigest()[:20]
                destination=DATA/'images'/f'{identifier}.png'
                render(text,str(face),rng).save(destination)
                records.append({'path':str(destination.relative_to(DATA)),'text':text,'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'split':split(card),'language':language,'source':'synthetic','kind':kind,'font':face.name,'italic':'Italic' in face.name})
        scan=DATA/'scans'/f"{card['id']}.jpg"
        if not scan.exists():
            scan.write_bytes(request(card['image_uris']['large']))
            time.sleep(.12)
        # M21 standard title: printed name left, mana cost at the right. A later
        # visual review freezes each box; crop discovery does not consult OCR.
        symbols=len(re.findall(r'\{[^}]+\}',card['mana_cost']))
        bbox=frozen_real[card['id']]['bbox'] if card['id'] in frozen_real else [53,49,max(220,620-symbols*31-8),99]
        if card['id'] in frozen_real and hashlib.sha256(scan.read_bytes()).hexdigest()!=frozen_real[card['id']]['scan_sha256']:
            raise RuntimeError('Frozen localized source scan changed: '+card['id'])
        destination=DATA/'images'/f"scan-{card['id']}.png"
        Image.open(scan).convert('L').resize((672,936)).crop(bbox).save(destination)
        text=normalized_printed(title)
        records.append({'path':str(destination.relative_to(DATA)),'text':text,'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'split':split(card),'language':language,'source':'real-scan','kind':'title','font':'printed-m21-title','italic':False,'bbox':bbox,'image_url':card['image_uris']['large'],'scan_sha256':hashlib.sha256(scan.read_bytes()).hexdigest(),'label_provenance':'Localized printed_name; standard M21 title box awaiting contact-sheet review'})
    for record in records:
        if record['source']=='real-scan' and record['card_id'] in frozen_real:
            record.update(frozen_real[record['card_id']])
    alphabet=ALPHABET+''.join(sorted(characters-set(ALPHABET),key=ord))
    (DATA/'manifest.jsonl').write_text(''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in records))
    (CORPUS/'alphabet.json').write_text(json.dumps({'alphabet':alphabet,'languages':LANGUAGES,'normalization':'NFC; preserve accents, punctuation and case; remove brace-encoded mana spelling from synthetic lines only','unrendered_characters':{face:sorted(chars) for face,chars in missing.items()}},ensure_ascii=False,indent=2)+'\n')
    summary={'prints':len(cards),'card_identities':len({c['name'].casefold() for c in cards}),'lines':len(records),'alphabet_characters':len(alphabet),'languages':{lang:{'lines':sum(r['language']==lang for r in records),'actual_titles':sum(r['language']==lang and r['source']=='real-scan' for r in records)} for lang in LANGUAGES},'scope':'Latin-accent training corpus; actual title crops require visual review before validation claims.'}
    frozen_summary_path=CORPUS/'summary.json'
    if frozen_summary_path.exists() and frozen_real:
        expected=json.loads(frozen_summary_path.read_text())
        actual=hashlib.sha256((DATA/'manifest.jsonl').read_bytes()).hexdigest()
        if expected.get('manifest_sha256') and actual!=expected['manifest_sha256']:
            raise RuntimeError('Frozen localized corpus manifest changed')
        summary.update({key:expected[key] for key in ['manifest_sha256','title_boxes_visually_reviewed','scope'] if key in expected})
    (CORPUS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
