"""Find actual printed rule/flavor lines with exact metadata-supported labels.

This intentionally rejects OCR mistakes rather than teaching the model its own
mistakes. Metadata is checked as contiguous normalized text; layout is never
inferred from modern Oracle line wrapping. The accepted subset is biased toward
already legible lines, and is not used to claim an independent font benchmark.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from bootstrap_scans import recognize
from network import load_model
from prepare import normal,split


def compact(text):
    return ' '.join(normal(text).split())


def main():
    data=Path(__file__).resolve().parent/'data'
    torch.set_num_threads(1)
    model=load_model(data.parent/'checkpoints/base.pt').eval()
    records=[]
    for card in json.loads((data/'cards.json').read_text()):
        path=data/'scans'/f"{card['id']}.jpg"
        if not path.exists():
            continue
        image=Image.open(path).convert('L').resize((672,936))
        legacy=card['set'] in ['lea','7ed']
        x0,x1,y0,y1=(79,592,574,833) if legacy else (57,614,592,834)
        region=image.crop((x0,y0,x1,y1))
        array=np.array(region)
        dark=(array<90).sum(1)>8
        rows=np.flatnonzero(dark)
        if len(rows)==0:
            continue
        groups=np.split(rows,np.flatnonzero(np.diff(rows)>3)+1)
        flavor=compact(card['flavor_text'])
        rules=compact(card['oracle_text'])
        for index,group in enumerate(groups):
            if not 8<=len(group)<=43:
                continue
            top=max(0,int(group[0])-4)
            bottom=min(region.height,int(group[-1])+5)
            crop=region.crop((0,top,region.width,bottom))
            prediction,_,_=recognize(model,crop)
            prediction=compact(prediction)
            if len(prediction)<10:
                continue
            source=None
            for kind,text in [('flavor',flavor),('rules',rules)]:
                # Match printed text punctuation/case exactly; do not silently
                # correct OCR using a nearby Oracle sentence.
                if re.search(r'(?<![A-Za-z0-9])'+re.escape(prediction)+r'(?![A-Za-z0-9])',text):
                    source=kind
                    break
            if source is None:
                continue
            italic=source=='flavor'
            if source=='rules':
                location=rules.index(prediction)
                opening=rules.rfind('(',0,location+1)
                closing=rules.rfind(')',0,location+1)
                italic=opening>closing
            identifier=f"body-{card['id']}-{index}"
            destination=data/'images'/f'{identifier}.png'
            crop.save(destination)
            records.append({'path':str(destination.relative_to(data)),'text':prediction,'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'split':split(card),'source':'real-scan','kind':source,'font':f"printed-{card['set']}-body",'italic':italic,'set':card['set'],'image_url':card['image_uris']['large'],'scan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bbox':[x0,y0+top,x1,y0+bottom],'label_provenance':'Exact contiguous metadata substring accepted from printed scan; baseline-readable selection bias'})
            print(f'Accepted {source} {card["set"]}:{prediction}',flush=True)
    manifest=data/'manifest.jsonl'
    existing=[json.loads(x) for x in manifest.read_text().splitlines() if not json.loads(x)['path'].startswith('images/body-')]
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in existing+records))
    (data/'body-crops.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Accepted {len(records)} body crops, {sum(r["italic"] for r in records)} italic',flush=True)


if __name__=='__main__':
    main()
