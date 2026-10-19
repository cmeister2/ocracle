#!/usr/bin/env python3
"""Propose localized body labels by exact accent-folded metadata alignment.

Accents are restored from the localized printed text, never guessed from English
Oracle text. Accepted candidates still need source/label contact-sheet review.
This writes a new corpus revision and preserves the first Latin experiment.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import numpy as np
from PIL import Image
import torch
from bootstrap_scans import recognize
from network import load_model
from language_head import normalized_printed
from prepare import split
from latin_prepare import DATA,CORPUS,ROOT


def folded_with_indices(text):
    output=[]
    indices=[]
    replacements=str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','—':'-','–':'-'})
    for index,character in enumerate(text):
        normalized=unicodedata.normalize('NFD',character).translate(replacements).casefold()
        for part in normalized:
            if unicodedata.category(part)=='Mn':continue
            output.append(part)
            indices.append(index)
    return ''.join(output),indices


def restored_label(prediction,printed):
    normalized=normalized_printed(printed)
    text,indices=folded_with_indices(normalized)
    target,_=folded_with_indices(' '.join(prediction.strip().split()))
    if len(target)<10:return None
    match=re.search(r'(?<![A-Za-z0-9])'+re.escape(target)+r'(?![A-Za-z0-9])',text)
    if not match:return None
    return normalized[indices[match.start()]:indices[match.end()-1]+1]


def main():
    torch.set_num_threads(1)
    model=load_model(ROOT/'training/checkpoints/base.pt').eval()
    destination=ROOT/'training/data/latin-v2'
    destination.mkdir(exist_ok=True)
    images=destination/'images'
    images.mkdir(exist_ok=True)
    # Copy paths as references to immutable source samples rather than changing
    # the frozen first-stage image files or manifest.
    initial=[json.loads(x) for x in (DATA/'manifest.jsonl').read_text().splitlines()]
    records=[]
    for record in initial:
        record=dict(record,path='../latin/'+record['path'])
        records.append(record)
    proposals=[]
    for card in json.loads((CORPUS/'cards.json').read_text()):
        scan=DATA/'scans'/f"{card['id']}.jpg"
        image=Image.open(scan).convert('L').resize((672,936))
        region=image.crop((54,590,617,836))
        ink=(np.asarray(region)<90).sum(1)>8
        rows=np.flatnonzero(ink)
        if not len(rows):continue
        groups=np.split(rows,np.flatnonzero(np.diff(rows)>3)+1)
        for index,group in enumerate(groups):
            if not 8<=len(group)<=43:continue
            top=max(0,int(group[0])-4)
            bottom=min(region.height,int(group[-1])+5)
            crop=region.crop((0,top,region.width,bottom))
            prediction,_,_=recognize(model,crop)
            candidates=[('flavor',card.get('flavor_text','')),('rules',card.get('printed_text') or (card['oracle_text'] if card['lang']=='en' else ''))]
            restored=None
            for kind,text in candidates:
                restored=restored_label(prediction,text)
                if restored is not None:break
            if restored is None:continue
            filename=f"body-{card['id']}-{index}.png"
            crop.save(images/filename)
            entry={'path':'images/'+filename,'text':restored,'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'split':split(card),'language':card['lang'],'source':'real-scan','kind':kind,'font':'printed-m21-body','italic':kind=='flavor','bbox':[54,590+top,617,590+bottom],'image_url':card['image_uris']['large'],'scan_sha256':hashlib.sha256(scan.read_bytes()).hexdigest(),'label_provenance':'Exact whole-word accent-folded localized printed-text match; original Unicode restored; review candidate','discovery_prediction':prediction.strip()}
            records.append(entry)
            proposals.append(entry)
            print(card['lang'],restored,flush=True)
    (destination/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    proposal_file=ROOT/'training/latin-corpus/body-proposals.json'
    proposal_file.write_text(json.dumps(proposals,ensure_ascii=False,indent=2)+'\n')
    print(f'Proposed {len(proposals)} localized body lines for review',flush=True)


if __name__=='__main__':
    main()
