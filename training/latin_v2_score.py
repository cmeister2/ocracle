#!/usr/bin/env python3
"""Score fixed post-selection V2 test identities against both source models."""
import hashlib
import json
import torch
import train as common
from network import load_model
from latin_v2_acquire import ROOT,DATA
from latin_v2_train import evaluate


def main():
    torch.set_num_threads(4);common.DATA=DATA
    report_path=ROOT/'training/latin-v2-report.json'
    report=json.loads(report_path.read_text())
    checkpoint=ROOT/'training/checkpoints/ocracle-mtg-latin-v2.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['checkpoint_sha256']:raise RuntimeError('Selected checkpoint changed')
    records=[json.loads(line) for line in (DATA/'manifest.jsonl').read_text().splitlines()]
    selection=[r for r in records if r['split']=='test' and r['corpus_generation']=='latin-v2-physical']
    report['v2_test']['card_identities']=len({r['oracle_id'] for r in selection})
    report['v2_test']['print_identities']=len({r['card_id'] for r in selection})
    results={}
    for name,path in [('generic_test',ROOT/'training/checkpoints/base.pt'),('v1_test',ROOT/'models/training/latin-v1-source.pt')]:
        model=load_model(path).eval()
        results[name]=evaluate(model,selection,('clean','severe','combined'))
        del model
    report.update(results)
    report['test_comparison_policy']='Identical reviewed test card identities and deterministic degradations, scored only after final validation checkpoint selection.'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({name:result['groups']['overall'] for name,result in results.items()},indent=2))


if __name__=='__main__':main()
