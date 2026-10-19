#!/usr/bin/env python3
"""Materialize the frozen corpus and reproduce the domain fine-tuning run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
from download_models import fetch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-train',action='store_true',help='Materialize data and verify provenance only')
    parser.add_argument('--steps',type=int,default=360)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--eval-limit',type=int,default=96,help='0 evaluates every held-out line in7conditions')
    args=parser.parse_args()
    subprocess.run([sys.executable,str(ROOT/'training/download_models.py'),'--checkpoint'],check=True)
    corpus=ROOT/'training/corpus'
    data=ROOT/'training/data'
    data.mkdir(exist_ok=True)
    (data/'cards.json').write_bytes((corpus/'cards.json').read_bytes())
    # Exact font provenance is required for the named reproducibility run.
    provenance=json.loads((corpus/'provenance.json').read_text())
    for font in provenance['fonts']:
        path=Path(font['path'])
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=font['sha256']:
            raise RuntimeError(f'Frozen font unavailable or changed: {path}; install matching fonts-urw-base35 before reproducing')
    subprocess.run([sys.executable,str(ROOT/'training/prepare.py')],check=True)
    records=[json.loads(x) for x in (data/'manifest.jsonl').read_text().splitlines()]
    for record in json.loads((corpus/'real-lines.json').read_text()):
        scan=data/'scans'/f"{record['card_id']}.jpg"
        fetch(record['image_url'],scan,record['scan_sha256'])
        destination=data/record['path']
        with Image.open(scan) as image:
            image.convert('L').resize((672,936)).crop(record['bbox']).save(destination)
        records.append(record)
    (data/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    summary=json.loads((corpus/'summary.json').read_text())
    manifest_hash=hashlib.sha256((data/'manifest.jsonl').read_bytes()).hexdigest()
    listing=''.join(f"{r['path']} {hashlib.sha256((data/r['path']).read_bytes()).hexdigest()}\n" for r in sorted(records,key=lambda r:r['path']))
    images_hash=hashlib.sha256(listing.encode()).hexdigest()
    if len(records)!=summary['lines'] or manifest_hash!=summary['manifest_sha256'] or images_hash!=summary['images_sha256']:
        raise RuntimeError('Frozen corpus reproduction mismatch in count, manifest, or image hashes')
    print(f'Materialized and verified {len(records)} labeled lines: {manifest_hash}',flush=True)
    if args.skip_train:
        return
    subprocess.run([sys.executable,str(ROOT/'training/train.py'),'--steps',str(args.steps),'--threads',str(args.threads),'--eval-limit',str(args.eval_limit)],check=True)
    subprocess.run([str(Path(sys.executable).parent/'rten-convert'),str(ROOT/'models/ocracle-mtg.onnx'),str(ROOT/'models/ocracle-mtg.rten')],check=True)
    print('Exported models/ocracle-mtg.rten; verify model hashes and evaluate held-out gallery before replacing the distributed artifact.',flush=True)


if __name__=='__main__':
    main()
