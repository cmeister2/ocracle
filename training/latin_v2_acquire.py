#!/usr/bin/env python3
"""Acquire unfiltered accent-bearing MTG titles across historic Latin frames.

Selection uses only metadata, glyph coverage and canonical identity splits;
recognizer predictions never filter the physical training/validation candidates.
"""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import random
from PIL import Image,ImageDraw,ImageFont
from prepare import EXCLUDED_NAMES,request,split
from network import ALPHABET

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'training/latin-v2-corpus'
DATA=ROOT/'training/data/latin-v2'
BULK=ROOT/'training/data/scryfall/all-cards-20261019091831.jsonl.gz'
LANGUAGES=['fr','de','it','pt','es']
COHORTS={'7ed':350,'8ed':350,'m15':269,'m21':274,'tsp':301,'plc':165,'fut':180}
CRITICAL=set('ÄïÊÈù')
FUTURE_RESERVED={
    "Nature's Lore":'78826359-fe63-44ad-adc4-a17ffcd710e4',
    'Wrath of God':'34515b16-c9a4-4f98-8c77-416a7a523407',
    'Gamble':'a54f0869-94c8-42af-9080-166efb9486a4',
    'Decree of Pain':'932668fa-d6e3-41c0-ad0c-8e0a00e68d11',
    'Preordain':'ac641490-ca14-48d7-8cc4-b69ce984befa',
}


def orthography():
    result={}
    for language in ['en']+LANGUAGES:
        url=f'https://raw.githubusercontent.com/unicode-org/cldr/release-48/common/main/{language}.xml'
        data=urllib.request.urlopen(url).read()
        xml=ET.fromstring(data)
        exemplars=[]
        for element in xml.findall('./characters/exemplarCharacters'):
            if element.get('type') in [None,'auxiliary']:
                exemplars.append({'type':element.get('type','main'),'set':element.text})
        # The frozen six exemplars use literal characters, not escaped ranges.
        letters={c for item in exemplars for c in item['set'] if unicodedata.category(c).startswith('L')}
        letters.update(c.upper() for c in list(letters) if len(c.upper())==1)
        if language=='de':letters.add('ẞ')
        result[language]={'source':url,'source_sha256':hashlib.sha256(data).hexdigest(),'exemplars':exemplars,'letters':''.join(sorted(letters,key=ord))}
    result['punctuation']={'characters':'‘’“”„«»‹›–—…¡¿','status':'Typography targets retained alongside ASCII punctuation'}
    (CORPUS/'orthography.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def excluded():
    names=set(EXCLUDED_NAMES)|{name.casefold() for name in ['Glory','Raging Kavu','Cleopatra, Exiled Pharaoh','Stone-Tongue Basilisk','Fungal Shambler','Opt']}
    names.update(name.casefold() for name in FUTURE_RESERVED)
    oracles=set(FUTURE_RESERVED.values())
    for path in [ROOT/'training/corpus/cards.json',ROOT/'training/latin-corpus/cards.json']:
        for card in json.loads(path.read_text()):
            names.add(card['name'].casefold());oracles.add(card['oracle_id'])
    evaluation=ROOT/'training/latin-eval/manifest.jsonl'
    for line in evaluation.read_text().splitlines():
        card=json.loads(line);names.add(card['card_name'].casefold());oracles.add(card['oracle_id'])
    return names,oracles


def acquire_metadata(orth):
    if (CORPUS/'cards.json').exists():return json.loads((CORPUS/'cards.json').read_text())
    names,oracles=excluded()
    candidates=defaultdict(list)
    letter_examples=defaultdict(dict)
    keys=['id','oracle_id','name','lang','set','collector_number','printed_name','printed_type_line','printed_text','flavor_text','type_line','oracle_text','mana_cost','image_uris','scryfall_uri','frame','layout','border_color','image_status','highres_image','multiverse_ids']
    # Stream the complete dated snapshot; no API alphabetical truncation.
    with gzip.open(BULK,'rt',encoding='utf-8') as source:
        for line in source:
            card=json.loads(line)
            lang=card.get('lang')
            if lang not in LANGUAGES:continue
            if card['name'].casefold() in names or card.get('oracle_id') in oracles:continue
            title=card.get('printed_name','')
            text=' '.join(card.get(field,'') or '' for field in ['printed_name','printed_type_line','printed_text','flavor_text'])
            if split(card)=='train':
                for char in set(text):
                    if char in orth[lang]['letters'] and char not in letter_examples[lang]:
                        letter_examples[lang][char]={key:card.get(key,'') for key in keys}
            if card.get('set') not in COHORTS or not card.get('image_uris') or (card.get('image_status')=='placeholder' and not card.get('multiverse_ids')):continue
            if card.get('layout')!='normal' or card.get('border_color')=='borderless':continue
            try:number=int(card['collector_number'])
            except ValueError:continue
            if number>COHORTS[card['set']]:continue
            if not title or not any(ord(c)>127 and unicodedata.category(c).startswith('L') for c in title):continue
            candidates[(lang,split(card))].append({key:card.get(key,'') for key in keys})
    chosen=[]
    for language in LANGUAGES:
        for grouping,count in [('train',20),('validation',8),('test',8)]:
            options=candidates[(language,grouping)]
            # Prioritize broad glyph/cohort coverage; break ties deterministically.
            random.Random(20261020).shuffle(options)
            seen_letters=Counter();seen_cohorts=Counter();seen_names=set()
            for _ in range(count):
                pool=[card for card in options if card['name'].casefold() not in seen_names]
                if not pool:break
                def score(card):
                    letters={c for c in card['printed_name'] if ord(c)>127}
                    return sum((12 if c in CRITICAL else 3)/(1+seen_letters[c]) for c in letters)+4/(1+seen_cohorts[card['set']])
                card=max(pool,key=score)
                chosen.append(card);seen_names.add(card['name'].casefold())
                seen_letters.update(set(card['printed_name']));seen_cohorts[card['set']]+=1
    # Add real-card metadata containing orthographic characters not yet covered
    # in acquired training fields. These supply synthetic labels, not extra gold
    # scans. No held-out identity is repurposed for a rare character.
    seen_ids={card['id'] for card in chosen}
    additions=[]
    for language in LANGUAGES:
        covered={c for card in chosen if card['lang']==language and split(card)=='train' for field in ['printed_name','printed_type_line','printed_text','flavor_text'] for c in (card.get(field) or '')}
        for char in orth[language]['letters']:
            if char in covered:continue
            card=letter_examples[language].get(char)
            if card and card['id'] not in seen_ids:
                additions.append(dict(card,synthetic_only=True));seen_ids.add(card['id'])
                covered.update(''.join(card.get(field) or '' for field in ['printed_name','printed_type_line','printed_text','flavor_text']))
    (CORPUS/'cards.json').write_text(json.dumps(chosen+additions,ensure_ascii=False,indent=2)+'\n')
    (CORPUS/'selection.json').write_text(json.dumps({'physical_titles':len(chosen),'supplementary_training_prints':len(additions),'candidate_counts':{f'{language}:{grouping}':len(cards) for (language,grouping),cards in candidates.items()},'selection':'Metadata glyph/cohort coverage only; no OCR filtering','bulk_source':'https://data.scryfall.io/all-cards/all-cards-20261019091831.jsonl.gz','bulk_sha256':hashlib.sha256(BULK.read_bytes()).hexdigest()},indent=2)+'\n')
    return chosen+additions


def title_box(card):
    if card['frame']=='future':return [109,48,621,107]
    import re
    symbols=len(re.findall(r'\{[^}]+\}',card['mana_cost']))
    right=max(250,620-symbols*31-5)
    if card['frame'] in ['1993','1997']:return [43,39,right,88]
    return [50,43,right,99]


def main():
    CORPUS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    (DATA/'scans-v2').mkdir(exist_ok=True)
    (DATA/'titles-v2').mkdir(exist_ok=True)
    (DATA/'review').mkdir(exist_ok=True)
    orth=orthography()
    cards=acquire_metadata(orth)
    records=[]
    for index,card in enumerate(cards):
        if card.get('synthetic_only'):continue
        source_url=card['image_uris']['large']
        provider='Scryfall'
        prefix='scryfall'
        if card.get('image_status')=='placeholder':
            if not card.get('multiverse_ids'):
                raise RuntimeError('Localized placeholder has no verified source: '+card['id'])
            source_url=f"https://gatherer.wizards.com/Handlers/Image.ashx?multiverseid={card['multiverse_ids'][0]}&type=card"
            provider='Wizards Gatherer'
            prefix='gatherer'
        scan=DATA/'scans-v2'/f"{prefix}-{card['id']}.img"
        if not scan.exists():scan.write_bytes(request(source_url))
        box=title_box(card)
        crop=Image.open(scan).convert('L').resize((672,936)).crop(box)
        path=DATA/'titles-v2'/f"{card['id']}.png"
        crop.save(path)
        record={'path':str(path.relative_to(DATA)),'text':unicodedata.normalize('NFC',card['printed_name']),'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'language':card['lang'],'split':split(card),'source':'real-scan','kind':'title','font':f"printed-{card['set']}-title",'italic':False,'set':card['set'],'frame':card['frame'],'bbox':box,'image_url':source_url,'source_provider':provider,'metadata_image_status':card.get('image_status'),'source_dimensions':list(Image.open(scan).size),'scan_sha256':hashlib.sha256(scan.read_bytes()).hexdigest(),'crop_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'review_status':'pending','label_provenance':'Localized printed_name; unfiltered acquisition, title box pending visual review'}
        records.append(record)
        print(f"{index+1} {card['lang']} {record['split']} {card['set']} {record['text']}",flush=True)
    (CORPUS/'titles-pending.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    for language in LANGUAGES:
        subset=[record for record in records if record['language']==language]
        for page in range((len(subset)+19)//20):
            sheet=Image.new('RGB',(1260,840),'white');draw=ImageDraw.Draw(sheet)
            for offset,record in enumerate(subset[page*20:page*20+20]):
                x=(offset//10)*630;y=(offset%10)*84
                draw.text((x,y),f"{page*20+offset}: {record['split']} {record['set']} {record['text']}",fill='black',font=font)
                sheet.paste(Image.open(DATA/record['path']),(x,y+22))
            sheet.save(DATA/'review'/f'{language}-{page+1}.png')
    print(f'Prepared {len(records)} unfiltered title crops for review',flush=True)


if __name__=='__main__':main()
