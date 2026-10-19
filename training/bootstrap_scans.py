"""Acquire actual card-title crops, accepting only metadata-exact recognitions.

The Scryfall card name supplies the label. Model predictions are used only to
locate the printed name and reject ambiguous crops. Accepted crops therefore
represent a baseline-readable subset, not an unbiased historic-font benchmark.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from network import ALPHABET,decode,load_model


def recognize(model,image):
    width=round(image.width*64/image.height)
    pixels=torch.from_numpy(np.array(image.resize((width,64)),dtype=np.float32)/255-.5)[None,None]
    with torch.inference_mode():
        sequence=model(pixels)[:,0].argmax(-1).tolist()
    return decode(sequence),sequence,width


def scan_records(cards,data):
    from prepare import request,normal,split
    torch.set_num_threads(1)
    model=load_model(data.parent/'checkpoints/base.pt').eval()
    records=[]
    for card in cards:
        target=normal(card['name'])
        if len(target)>48:
            continue
        scan=data/'scans'/f"{card['id']}.jpg"
        if not scan.exists():
            scan.write_bytes(request(card['image_uris']['large']))
            time.sleep(.11)
        image=Image.open(scan).convert('L')
        # Standard large Scryfall scan, normalized to canonical card dimensions.
        image=image.resize((672,936))
        accepted=None
        for y0,y1 in [(43,80),(39,79),(41,84),(44,78)]:
            region=image.crop((40,y0,627,y1))
            text,sequence,width=recognize(model,region)
            position=text.casefold().find(target.casefold())
            if position<0:
                continue
            # Trace the CTC run locations for the matched prefix, avoiding mana.
            chars=[]
            prior=-1
            for i,value in enumerate(sequence):
                if value and value!=prior:
                    chars.append((ALPHABET[value-1],i))
                prior=value
            first=chars[position][1]
            last=chars[position+len(target)-1][1]
            scale=region.width/width
            x0=max(0,int(first*4*scale)-8)
            x1=min(region.width,int(last*4*scale)+14)
            candidate=region.crop((x0,0,x1,region.height))
            pred,_,_=recognize(model,candidate)
            if pred.strip().casefold()==target.casefold():
                accepted=(candidate,(40+x0,y0,40+x1,y1))
                break
        if accepted is None:
            print(f'Skipped unmatched scan {card["set"]}:{target}',flush=True)
            continue
        candidate,bbox=accepted
        identifier=f"scan-{card['id']}"
        path=data/'images'/f'{identifier}.png'
        candidate.save(path)
        records.append({'path':str(path.relative_to(data)),'text':target,'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'split':split(card),'source':'real-scan','kind':'title','font':f"printed-{card['set']}-title",'italic':False,'set':card['set'],'image_url':card['image_uris']['large'],'scan_sha256':hashlib.sha256(scan.read_bytes()).hexdigest(),'bbox':bbox,'label_provenance':'Scryfall name; exact-match baseline filtered crop'})
        print(f'Accepted scan {card["set"]}:{target}',flush=True)
    return records


if __name__=='__main__':
    data=Path(__file__).resolve().parent/'data'
    records=scan_records(json.loads((data/'cards.json').read_text()),data)
    manifest=data/'manifest.jsonl'
    synthetic=[json.loads(x) for x in manifest.read_text().splitlines() if json.loads(x)['source']!='real-scan']
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in synthetic+records))
    (data/'scan-crops.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Accepted {len(records)} actual printed title crops',flush=True)
