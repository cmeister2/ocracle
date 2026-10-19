#!/usr/bin/env python3
"""Freeze reviewed physical Latin V2 lines and a reproducible glyph curriculum."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from fontTools.ttLib import TTFont
from prepare import FONTS,render,split
from latin_prepare import wrapped
from latin_v2_acquire import ROOT,CORPUS,DATA,excluded

SEED=20261020
LANGUAGES=['en','fr','de','it','pt','es']
FACES=list(FONTS)+[f'/usr/share/fonts/truetype/dejavu/{name}.ttf' for name in ['DejaVuSerif','DejaVuSerif-Italic','DejaVuSans','DejaVuSans-Oblique']]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(destination=DATA,verify=False):
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'synthetic-v2').mkdir(exist_ok=True)
    cards=json.loads((CORPUS/'cards.json').read_text())
    reviewed=json.loads((CORPUS/'titles-reviewed.json').read_text())
    bodies=json.loads((CORPUS/'body-lines.json').read_text())
    records=[]
    # Immutable V1 corpus is retained, with its original split for every identity.
    for line in (ROOT/'training/data/latin/manifest.jsonl').read_text().splitlines():
        record=json.loads(line)
        record['path']=str((ROOT/'training/data/latin'/record['path']).relative_to(ROOT/'training/data'))
        record['path']='../'+record['path']
        record['corpus_generation']='latin-v1'
        records.append(record)
    # English clean anchors retain the original card-level split and source font.
    for line in (ROOT/'training/data/manifest.jsonl').read_text().splitlines():
        record=json.loads(line);record['path']='../'+record['path'];record['language']='en';record['corpus_generation']='english-v1'
        records.append(record)
    for record in reviewed:
        if record['review_status']!='accepted':continue
        record=dict(record,corpus_generation='latin-v2-physical')
        if digest(DATA/record['path'])!=record['crop_sha256']:raise RuntimeError('Reviewed title pixels changed')
        records.append(record)
    for record in bodies:
        if record['review_status']!='accepted':continue
        record=dict(record,corpus_generation='latin-v2-reviewed-body')
        if digest(DATA/record['path'])!=record['image_sha256']:raise RuntimeError('Reviewed body pixels changed')
        records.append(record)
    coverage={face:set(TTFont(face).getBestCmap()) for face in FACES}
    rng=random.Random(SEED)
    def add(text,language,kind,card=None):
        text=unicodedata.normalize('NFC',text)
        for face in FACES:
            if any(ord(c) not in coverage[face] for c in text):continue
            identity=card['id'] if card else 'procedural:'+language
            name=Path(face).name
            identifier=hashlib.sha256(f'{identity}|{kind}|{text}|{name}'.encode()).hexdigest()[:24]
            relative=f'synthetic-v2/{identifier}.png'
            render(text,face,rng).save(destination/relative)
            records.append({'path':relative,'text':text,'language':language,'card_id':identity,'oracle_id':card['oracle_id'] if card else identity,'card_name':card['name'] if card else identity,'split':split(card) if card else 'train','source':'synthetic' if card else 'glyph-curriculum','kind':kind,'font':name,'italic':'Italic' in name or 'Oblique' in name,'corpus_generation':'latin-v2-synthetic','set':card.get('set') if card else None})
    for card in cards:
        sources=[('title',card['printed_name'])]
        for field,kind in [('printed_type_line','type'),('printed_text','rules'),('flavor_text','flavor')]:
            sources.extend((kind,line) for line in wrapped(card.get(field) or ''))
        for kind,text in sources:
            if text and len(text)<=64:add(text,card['lang'],kind,card)
        # Curly punctuation variants preserve the same card identity and split.
        title=card['printed_name']
        variant=title.replace("'",'’').replace(' - ',' – ')
        if variant!=title:add(variant,card['lang'],'title-typography',card)
    orth=json.loads((CORPUS/'orthography.json').read_text())
    # Explicit alphabet-practice lines cover exemplar glyphs with no card-text
    # examples. They are training-only primitives, never physical evaluation.
    for language in LANGUAGES:
        chars=orth[language]['letters']+orth['punctuation']['characters']
        for start in range(0,len(chars),12):
            letters=chars[start:start+12]
            add(' '.join(letters),language,'orthographic-glyphs')
            add(' '.join('a'+c+'e' for c in letters),language,'orthographic-context')
    evaluation=[json.loads(l) for l in (ROOT/'training/latin-eval/manifest.jsonl').read_text().splitlines()]
    reserved_names={r['card_name'].casefold() for r in evaluation}|{n.casefold() for n in ['Glory','Raging Kavu','Cleopatra, Exiled Pharaoh','Stone-Tongue Basilisk','Fungal Shambler','Opt']}
    reserved_oracles={r['oracle_id'] for r in evaluation}|{'3853ef6a-6771-4708-8ed5-d667f646de5a','d126bb08-be60-4046-91f1-65672ef42c63','e7a80464-1522-48d7-bf82-f9894f5c3ae0','c38efbe5-7ace-41fa-b874-72e3702a833c','9991941e-436f-42d4-8b0c-1ee84234774b','713332c1-5bd8-400f-bfff-c1ca0697a043'}
    excluded_records=[r for r in records if r['card_name'].casefold() in reserved_names or r['oracle_id'] in reserved_oracles]
    records=[r for r in records if r['card_name'].casefold() not in reserved_names and r['oracle_id'] not in reserved_oracles]
    alphabet=json.loads((ROOT/'training/latin-corpus/alphabet.json').read_text())['alphabet']
    allchars={c for r in records for c in r['text']}
    alphabet+=''.join(sorted(allchars-set(alphabet),key=ord))
    for record in records:
        location=destination/record['path'] if record['corpus_generation']=='latin-v2-synthetic' else DATA/record['path']
        record['image_sha256']=digest(location)
    manifest=''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in records)
    (destination/'manifest.jsonl').write_text(manifest)
    identity_splits=defaultdict(set)
    for record in records:identity_splits[record['oracle_id']].add(record['split'])
    if any(len(s)!=1 for s in identity_splits.values()):raise RuntimeError('Card identity split leakage')
    if any(r['card_name'].casefold() in reserved_names or r['oracle_id'] in reserved_oracles for r in records):raise RuntimeError('Independent holdout identity leaked')
    train_chars=Counter(c for r in records if r['split']=='train' for c in r['text'])
    if not set('ÄïÊÈù‘’“”–—')<=train_chars.keys():raise RuntimeError('Required orthography has no training examples')
    summary={'seed':SEED,'lines':len(records),'manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),'alphabet_characters':len(alphabet),'split':dict(Counter(r['split'] for r in records)),'generation':dict(Counter(r['corpus_generation'] for r in records)),'physical_titles':dict(Counter(r['language'] for r in records if r['corpus_generation']=='latin-v2-physical')),'physical_bodies':sum(r['corpus_generation']=='latin-v2-reviewed-body' for r in records),'training_character_counts':dict(sorted(train_chars.items())),'evaluation_excluded_names':sorted(reserved_names),'evaluation_excluded_oracle_ids':sorted(reserved_oracles),'late_holdout_parent_rows_excluded':len(excluded_records),'late_holdout_parent_identities':sorted({r['card_name'] for r in excluded_records}),'policy':'Canonical-name hash split shared across all languages and printings. No OCR-filtered new titles. Reviewed physical labels only; procedural glyph curriculum explicitly separate.'}
    if verify:
        expected=json.loads((CORPUS/'summary.json').read_text())
        if summary['lines']!=expected['lines'] or summary['manifest_sha256']!=expected['manifest_sha256']:raise RuntimeError('Frozen V2 corpus manifest mismatch')
    else:
        (CORPUS/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        (CORPUS/'alphabet.json').write_text(json.dumps({'alphabet':alphabet,'languages':LANGUAGES,'normalization':'NFC; accents and typography preserved'},ensure_ascii=False,indent=2)+'\n')
        fonts=[{'path':face,'sha256':digest(Path(face)),'license':'AGPL-3.0 with font exception' if 'urw-base35' in face else 'DejaVu fonts license (Bitstream Vera-derived; see /usr/share/doc/fonts-dejavu-core/copyright)'} for face in FACES]
        (CORPUS/'fonts.json').write_text(json.dumps(fonts,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DATA);parser.add_argument('--verify-frozen',action='store_true');args=parser.parse_args()
    print(json.dumps(materialize(args.output,args.verify_frozen),ensure_ascii=False,indent=2))
