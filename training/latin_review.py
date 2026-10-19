#!/usr/bin/env python3
"""Contact sheets for reviewing the exact localized title pixels and labels."""
import json
from PIL import Image,ImageDraw,ImageFont
from latin_prepare import DATA,LANGUAGES


def main():
    records=[json.loads(x) for x in (DATA/'manifest.jsonl').read_text().splitlines()]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    for language in LANGUAGES:
        choices=[r for r in records if r['language']==language and r['source']=='real-scan']
        sheet=Image.new('RGB',(1080,760),'white')
        draw=ImageDraw.Draw(sheet)
        for index,record in enumerate(choices):
            x=(index//10)*540
            y=(index%10)*76
            draw.text((x,y),record['text'],fill='black',font=font)
            sheet.paste(Image.open(DATA/record['path']),(x,y+19))
        destination=DATA/f'review-{language}.png'
        sheet.save(destination)
        print(destination)


if __name__=='__main__':
    main()
