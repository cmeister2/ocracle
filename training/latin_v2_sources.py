#!/usr/bin/env python3
"""Resolve reviewed V2 sources without using English placeholder imagery."""
import gzip
import json
from latin_v2_acquire import BULK,CORPUS,excluded


def main():
    path=CORPUS/'cards.json'
    cards=json.loads(path.read_text())
    names,oracles=excluded()
    cards=[card for card in cards if card['name'].casefold() not in names and card['oracle_id'] not in oracles]
    by_id={card['id']:card for card in cards}
    remaining=set(by_id)
    with gzip.open(BULK,'rt',encoding='utf-8') as source:
        for line in source:
            card=json.loads(line)
            if card.get('id') not in remaining:continue
            target=by_id[card['id']]
            for key in ['image_status','highres_image','multiverse_ids','released_at']:
                target[key]=card.get(key)
            remaining.remove(card['id'])
            if not remaining:break
    if remaining:raise RuntimeError('Selected card missing from frozen bulk snapshot')
    path.write_text(json.dumps(cards,ensure_ascii=False,indent=2)+'\n')
    print('Resolved source metadata for',len(cards),'prints',flush=True)
    print('Physical placeholders with localized Gatherer image IDs:',sum(card.get('image_status')=='placeholder' and bool(card.get('multiverse_ids')) and not card.get('synthetic_only') for card in cards),flush=True)


if __name__=='__main__':main()
