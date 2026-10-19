#!/usr/bin/env python3
"""Build a language-specific Unicode alphabet and initialize a neural head.

This prepares weights for training; it never claims that newly initialized
classes recognize their glyphs. Each input card must belong to one language.
"""
import argparse
import json
from pathlib import Path
import re
import unicodedata
import torch
from network import ALPHABET,adapt_model


def normalized_printed(text):
    """Preserve Unicode orthography; remove API mana spelling from line labels."""
    text=unicodedata.normalize('NFC',text)
    text=re.sub(r'\{[^}]+\}',' ',text)
    return ' '.join(text.split())


def alphabet_for_cards(cards,language):
    """Collect printed characters for one whole-card language profile.

    English Oracle fallback is deliberately forbidden for localized text. The
    Latin base alphabet remains available for collector and artist credit lines.
    """
    characters=set(ALPHABET)
    lines=[]
    for card in cards:
        if card.get('lang')!=language:
            raise ValueError(f"Expected language {language}, found {card.get('lang')}")
        faces=card.get('card_faces',[card])
        for face in faces:
            for field in ['flavor_name','printed_name','printed_type_line','printed_text','flavor_text']:
                if face.get(field):
                    line=normalized_printed(face[field])
                    if line:
                        characters.update(line)
                        lines.append({'card_id':card['id'],'oracle_id':card.get('oracle_id'),'language':language,'field':field,'text':line})
    if not lines:
        raise ValueError('No localized printed labels; English Oracle fallback is not permitted')
    # Stable ordering preserves the original Latin rows, then adds codepoints.
    alphabet=ALPHABET+''.join(sorted(characters-set(ALPHABET),key=ord))
    return alphabet,lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards',type=Path,required=True,help='Scryfall card JSON list with lang and localized printed fields')
    parser.add_argument('--language',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    cards=json.loads(args.cards.read_text())
    alphabet,lines=alphabet_for_cards(cards,args.language)
    model=adapt_model(args.checkpoint,alphabet)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state':model.state_dict(),'alphabet':alphabet,'language':args.language,'trained':False},args.output_dir/'initialized.pt')
    config={'alphabet':alphabet,'input_height':64,'model_name':f'ocracle-mtg-{args.language}-untrained','language':args.language,'trained':False,'scope':'Initialized classifier; actual language training and held-out validation required'}
    (args.output_dir/'recognizer-config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (args.output_dir/'printed-lines.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in lines))
    print(f'Initialized {len(alphabet)} character classes for {args.language}; training is still required')


if __name__=='__main__':
    main()
