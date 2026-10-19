#!/usr/bin/env python3
"""Replay frozen Latin V2 data hashes before optionally repeating training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image
from download_models import fetch
from latin_v2_acquire import ROOT,CORPUS,DATA
from latin_v2_prepare import materialize


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-train',action='store_true');parser.add_argument('--skip-parent-replay',action='store_true',help='Use already-materialized frozen English/V1 data; V2 verifies every referenced PNG hash');parser.add_argument('--steps',type=int,default=1000);args=parser.parse_args()
    source=json.loads((ROOT/'models/training/latin-v1-source.json').read_text())
    if hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()!=source['sha256']:raise RuntimeError('Pinned V1 initialization checkpoint changed')
    if not args.skip_parent_replay:
        subprocess.run([sys.executable,str(ROOT/'training/reproduce.py'),'--skip-train'],check=True)
        subprocess.run([sys.executable,str(ROOT/'training/latin_prepare.py')],check=True)
    for entry in json.loads((CORPUS/'fonts.json').read_text()):
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Pinned V2 font differs: '+entry['path'])
    records=json.loads((CORPUS/'titles-reviewed.json').read_text())
    bodies=json.loads((CORPUS/'body-lines.json').read_text())
    for record in [r for r in records if r['review_status']=='accepted']+bodies:
        scan=DATA/'replay-scans'/f"{record['card_id']}.img";scan.parent.mkdir(exist_ok=True)
        if not scan.exists():
            candidates=[DATA/'scans-v2'/f"{provider}-{record['card_id']}.img" for provider in ['gatherer','scryfall']]
            candidates.append(ROOT/'training/data/latin/scans'/f"{record['card_id']}.jpg")
            for cached in candidates:
                if cached.exists() and hashlib.sha256(cached.read_bytes()).hexdigest()==record['scan_sha256']:
                    shutil.copyfile(cached,scan);break
        fetch(record['image_url'],scan,record['scan_sha256'])
        destination=DATA/record['path'];destination.parent.mkdir(parents=True,exist_ok=True)
        Image.open(scan).convert('L').resize((672,936)).crop(record['bbox']).save(destination)
        expected=record.get('crop_sha256',record.get('image_sha256'))
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:raise RuntimeError('Pinned physical crop differs: '+record['card_id'])
    summary=materialize(verify=True)
    print(f"Replayed {summary['lines']} records; complete labels and image hashes match {summary['manifest_sha256']}",flush=True)
    if args.skip_train:return
    subprocess.run([sys.executable,str(ROOT/'training/latin_v2_train.py'),'--steps',str(args.steps)],check=True)
    subprocess.run([sys.executable,str(ROOT/'training/latin_v2_validate.py')],check=True)


if __name__=='__main__':main()
