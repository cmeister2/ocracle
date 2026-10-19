#!/usr/bin/env python3
"""Add unfiltered physical rare-letter titles outside the initial set cohorts."""
from collections import Counter,defaultdict
import gzip
import json
import unicodedata
from latin_v2_acquire import BULK,CORPUS,CRITICAL,LANGUAGES,excluded
from prepare import split


def main():
    cards=json.loads((CORPUS/'cards.json').read_text())
    names,oracles=excluded()
    selected_ids={c['id'] for c in cards}
    selected_oracles={c['oracle_id'] for c in cards}
    candidates=defaultdict(list)
    keys=['id','oracle_id','name','lang','set','collector_number','printed_name','printed_type_line','printed_text','flavor_text','type_line','oracle_text','mana_cost','image_uris','scryfall_uri','frame','layout','border_color','image_status','highres_image','multiverse_ids']
    with gzip.open(BULK,'rt',encoding='utf-8') as source:
        for line in source:
            card=json.loads(line)
            if card.get('lang') not in LANGUAGES or not card.get('image_uris') or card.get('layout')!='normal' or (card.get('image_status')=='placeholder' and not card.get('multiverse_ids')):continue
            if card.get('frame') not in ['1993','1997','2003','2015'] or card.get('border_color')=='borderless':continue
            if any(effect in card.get('frame_effects',[]) for effect in ['showcase','extendedart']):continue
            if card['name'].casefold() in names or card.get('oracle_id') in oracles:continue
            for char in CRITICAL.intersection(card.get('printed_name','')):
                candidates[(char,split(card))].append({key:card.get(key,'') for key in keys})
    for char in sorted(CRITICAL):
        for grouping,target in [('train',2),('validation',1)]:
            current=[c for c in cards if char in c.get('printed_name','') and not c.get('synthetic_only') and split(c)==grouping]
            for card in sorted(candidates[(char,grouping)],key=lambda c:(c['released_at'] if 'released_at' in c else '',c['id'])):
                if len(current)>=target:break
                if card['oracle_id'] in selected_oracles:continue
                card['rare_title_reason']=char
                cards.append(card);current.append(card);selected_ids.add(card['id']);selected_oracles.add(card['oracle_id'])
            # Promote a supplementary training print if it contains the exact
            # rare physical title and no new independent identity is available.
            if len(current)<target:
                for card in cards:
                    if len(current)>=target:break
                    if char in card.get('printed_name','') and split(card)==grouping and card.get('synthetic_only'):
                        card.pop('synthetic_only');card['rare_title_reason']=char;current.append(card)
            print(char,grouping,[(c['lang'],c['printed_name'],c['name'],c['set']) for c in current],flush=True)
    (CORPUS/'cards.json').write_text(json.dumps(cards,ensure_ascii=False,indent=2)+'\n')
    (CORPUS/'rare-title-audit.json').write_text(json.dumps({char:{grouping:[{'id':c['id'],'name':c['name'],'title':c['printed_name'],'language':c['lang'],'set':c['set']} for c in cards if not c.get('synthetic_only') and char in c.get('printed_name','') and split(c)==grouping] for grouping in ['train','validation','test']} for char in sorted(CRITICAL)},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
