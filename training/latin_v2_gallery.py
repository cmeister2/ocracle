#!/usr/bin/env python3
"""Measure frozen English font preservation without registering V2 profiles."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unicodedata
from train import distance

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(text):return ' '.join(unicodedata.normalize('NFC',text).split())


def main():
    fixtures=ROOT/'docs/gallery/assets/fixtures.json'
    fixture_hash=sha(fixtures);samples=json.loads(fixtures.read_text())['samples']
    model=ROOT/'models/ocracle-mtg-latin-v2.rten';configuration=ROOT/'models/latin-v2-profiles/en.json';binary=ROOT/'target/release/ocracle'
    identities={'recognition_model_sha256':sha(model),'model_config_sha256':sha(configuration),'executable_sha256':sha(binary),'fixture_manifest_sha256':fixture_hash}
    groups=defaultdict(lambda:{'observations':0,'character_edits':0,'reference_characters':0,'exact_matches':0});records=[];images={}
    with tempfile.TemporaryDirectory(prefix='ocracle-v2-gallery-') as directory:
        snapshot=Path(directory)
        for original,name in [(model,'model.rten'),(configuration,'config.json'),(binary,'ocracle')]:shutil.copy2(original,snapshot/name)
        for sample in samples:
            for variant in sample['variants']:
                image=fixtures.parent/variant['image'];images[image]=sha(image)
                inference=json.loads(subprocess.check_output([str(snapshot/'ocracle'),'recognize',str(image),'--line','--passes','all','--recognition-model',str(snapshot/'model.rten'),'--model-config',str(snapshot/'config.json')],text=True))
                if inference['model']['sha256']!=identities['recognition_model_sha256']:raise RuntimeError('Wrong inference weights')
                if not math.isfinite(inference['confidence']) or not 0<=inference['confidence']<=1:raise RuntimeError('Invalid neural confidence')
                expected=normalize(sample['ground_truth']);prediction=normalize(inference['text']);edits=distance(expected,prediction)
                records.append({'sample_id':sample['id'],'variant_id':variant['id'],'image_sha256':images[image],'reference':sample['ground_truth'],'character_edits':edits,'reference_characters':len(expected),'exact_match':expected==prediction,'inference':inference})
                for group in ['overall','severity:'+variant['id']]:
                    s=groups[group];s['observations']+=1;s['character_edits']+=edits;s['reference_characters']+=len(expected);s['exact_matches']+=int(expected==prediction)
    if sha(fixtures)!=fixture_hash or any(sha(path)!=value for path,value in images.items()):raise RuntimeError('Frozen evaluation inputs changed')
    for s in groups.values():s['character_error_rate']=s['character_edits']/max(1,s['reference_characters'])
    result={'schema':1,'generated_at':datetime.now(timezone.utc).isoformat(),**identities,'passes':'all','purpose':'Post-selection English physical-font preservation check; experimental V2 remains unregistered','normalization':'NFC and collapsed whitespace; case, punctuation and accents retained','confidence_note':'Uncalibrated emitted-character posterior score, not probability of exact transcription','groups':dict(groups),'records':records}
    output=ROOT/'training/latin-v2-english-gallery.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['groups'],indent=2))


if __name__=='__main__':main()
