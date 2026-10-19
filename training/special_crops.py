"""Reviewed frame-specific title boxes, including difficult unreadable fonts.

Unlike the baseline-exact bootstrap these titles are not filtered by OCR score.
The bounding templates have been visually reviewed on the frozen30-print corpus.
"""
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw
from prepare import normal,split

MP2_TITLE_X={'Aggravated Assault':(101,513),'Attrition':(225,454),'Aven Mindcensor':(164,508),'Boil':(287,382),'Bontu the Glorified':(74,497),'Capsize':(257,413),'Chain Lightning':(143,510),'Choke':(269,390),'Armageddon':(211,457),'Austere Command':(123,478)}

TEMPLATES={'fut':(112,52,619,102),'plc':(53,53,510,100),'mp2':(46,49,529,96)}


def main():
    data=Path(__file__).resolve().parent/'data'
    records=[]
    review=Image.new('RGB',(1620,750),'white')
    draw=ImageDraw.Draw(review)
    for card in json.loads((data/'cards.json').read_text()):
        if card['set'] not in TEMPLATES:
            continue
        scan=data/'scans'/f"{card['id']}.jpg"
        bbox=TEMPLATES[card['set']]
        if card['set']=='mp2':
            left,right=MP2_TITLE_X[card['name']]
            bbox=(left,bbox[1],right,bbox[3])
        crop=Image.open(scan).convert('L').resize((672,936)).crop(bbox)
        target=normal(card['name'])
        if card['set']=='mp2':
            target=target.upper()
        destination=data/'images'/f"special-{card['id']}.png"
        crop.save(destination)
        index=len(records)
        x=(index//10)*540
        y=(index%10)*75
        draw.text((x,y),f"{card['set']} {target}",fill='black')
        review.paste(crop,(x,y+18))
        records.append({'path':str(destination.relative_to(data)),'text':target,'card_id':card['id'],'oracle_id':card['oracle_id'],'card_name':card['name'],'split':split(card),'source':'real-scan','kind':'title','font':f"printed-{card['set']}-special-title",'italic':False,'set':card['set'],'image_url':card['image_uris']['large'],'scan_sha256':hashlib.sha256(scan.read_bytes()).hexdigest(),'bbox':bbox,'label_provenance':'Scryfall card identity; frame-specific title box and printed casing visually reviewed; not filtered by OCR confidence'})
    manifest=data/'manifest.jsonl'
    existing=[json.loads(x) for x in manifest.read_text().splitlines() if not json.loads(x)['path'].startswith('images/special-')]
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in existing+records))
    (data/'special-crops.json').write_text(json.dumps(records,indent=2)+'\n')
    review.save('/tmp/ocracle-special-review.jpg')
    print(f'Added {len(records)} special titles')


if __name__=='__main__':
    main()
