#!/usr/bin/env python3
"""Validate and package a selected V2 checkpoint; never selects model weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    def run(args):subprocess.run(args,cwd=ROOT,check=True)
    python=sys.executable
    run([python,'training/latin_v2_score.py'])
    run([str(Path(python).parent/'rten-convert'),'models/ocracle-mtg-latin-v2.onnx','models/ocracle-mtg-latin-v2.rten'])
    rows=[json.loads(line) for line in (ROOT/'training/data/latin-v2/manifest.jsonl').read_text().splitlines()]
    for language in ['en','fr','de','it','pt','es']:
        if language=='en':fixture=ROOT/'docs/gallery/assets/beleren-title.png'
        else:
            record=next(r for r in rows if r['language']==language and r['split']=='test' and r['corpus_generation']=='latin-v2-physical')
            fixture=ROOT/'training/data/latin-v2'/record['path']
        run([python,'training/check_export.py','--checkpoint','training/checkpoints/ocracle-mtg-latin-v2.pt','--onnx','models/ocracle-mtg-latin-v2.onnx','--model','models/ocracle-mtg-latin-v2.rten','--config',f'models/latin-v2-profiles/{language}.json','--fixture',str(fixture),'--report',f'training/latin-v2-export-validation-{language}.json'])
    model_hash=hashlib.sha256((ROOT/'models/ocracle-mtg-latin-v2.rten').read_bytes()).hexdigest()
    run([python,'training/latin-eval/score.py','--recognition-model','models/ocracle-mtg-latin-v2.rten','--model-config-dir','models/latin-v2-profiles','--expected-sha256',model_hash,'--output','training/latin-eval/results-v2.json'])
    run([python,'training/latin_v2_gallery.py'])
    run([python,'training/latin_v2_package.py'])


if __name__=='__main__':main()
