#!/usr/bin/env python3
"""Train a separate NFC Latin profile while preserving the English baseline."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import torch
import train as common
from network import adapt_model,decode,load_model
from latin_prepare import DATA,CORPUS,LANGUAGES,ROOT


@torch.inference_mode()
def evaluate(model,records,per_language=20):
    model.eval()
    rng=random.Random(20261019)
    selection=[]
    for language in LANGUAGES:
        choices=[r for r in records if r['language']==language]
        rng.shuffle(choices)
        scans=[r for r in choices if r['source']=='real-scan']
        synthetic=[r for r in choices if r['source']!='real-scan']
        selection+=scans[:4]+synthetic[:max(0,per_language-min(4,len(scans)))]
    stats=defaultdict(lambda:{'lines':0,'characters':0,'errors':0,'exact_lines':0})
    examples=[]
    for severity in ['clean','severe','combined']:
        for index,record in enumerate(selection):
            image=common.tensor(record,severity,20261019+index)
            prediction=decode(model(image[None])[:,0].argmax(-1).tolist(),model.alphabet).strip()
            errors=common.distance(record['text'],prediction)
            for group in ['overall','language:'+record['language'],'severity:'+severity,'source:'+record['source']]:
                value=stats[group]
                value['lines']+=1
                value['characters']+=len(record['text'])
                value['errors']+=errors
                value['exact_lines']+=int(record['text']==prediction)
            if record['source']=='real-scan' and severity=='clean':
                examples.append({'language':record['language'],'card_name':record['card_name'],'target':record['text'],'prediction':prediction})
    for value in stats.values():
        value['cer']=value['errors']/max(1,value['characters'])
        value['exact_line_accuracy']=value['exact_lines']/max(1,value['lines'])
    return {'groups':dict(stats),'examples':examples,'unique_lines':len(selection),'conditions':['clean','severe','combined'],'normalization':'NFC labels; strip predicted boundary whitespace only'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261019)
    rng=random.Random(20261019)
    common.DATA=DATA
    records=[json.loads(x) for x in (DATA/'manifest.jsonl').read_text().splitlines()]
    identities=defaultdict(set)
    for record in records:
        identities[record['oracle_id']].add(record['split'])
    assert all(len(splits)==1 for splits in identities.values()),'Cross-language card identity leakage'
    train=[r for r in records if r['split']=='train']
    validation=[r for r in records if r['split']=='validation']
    test=[r for r in records if r['split']=='test']
    base=ROOT/'training/checkpoints/base.pt'
    alphabet=json.loads((CORPUS/'alphabet.json').read_text())['alphabet']
    baseline=load_model(base)
    print('Scoring English baseline against localized labels',flush=True)
    baseline_validation=evaluate(baseline,validation,12)
    baseline_test=evaluate(baseline,test)
    del baseline
    model=adapt_model(base,alphabet,20261019)
    for parameter in list(model.conv.parameters())+list(model.gru.parameters()):
        parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.output.parameters(),lr=.0003,weight_decay=.00001)
    teacher_weight=model.output[0].weight[:97].detach().clone()
    teacher_bias=model.output[0].bias[:97].detach().clone()
    ctc=torch.nn.CTCLoss()
    indices={c:i+1 for i,c in enumerate(alphabet)}
    by_language={lang:[r for r in train if r['language']==lang] for lang in LANGUAGES}
    scans={lang:[r for r in by_language[lang] if r['source']=='real-scan'] for lang in LANGUAGES}
    history=[]
    start=time.time()
    # Preserve the pretrained BN running statistics and visual representation.
    model.eval()
    best_score=float('inf')
    best_state=None
    validation_history=[]
    for step in range(args.steps):
        chosen=[]
        for _ in range(args.batch_size):
            language=rng.choice(LANGUAGES)
            pool=scans[language] if scans[language] and rng.random()<.25 else by_language[language]
            chosen.append(rng.choice(pool))
        tensors=[common.tensor(r,rng.choice(['clean','clean','gaussian','severe','motion','downsample','jpeg','combined']),rng.randrange(2**32)) for r in chosen]
        images,lengths=common.batch(tensors)
        targets=torch.tensor([indices[c] for r in chosen for c in r['text']],dtype=torch.long)
        target_lengths=torch.tensor([len(r['text']) for r in chosen],dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss=ctc(model(images),targets,lengths,target_lengths)
        loss=loss+.01*((model.output[0].weight[:97]-teacher_weight).square().mean()+(model.output[0].bias[:97]-teacher_bias).square().mean())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite Unicode CTC loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),4)
        optimizer.step()
        history.append(float(loss.detach()))
        if (step+1)%20==0:print(json.dumps({'step':step+1,'mean_loss_last_20':sum(history[-20:])/20,'elapsed_seconds':round(time.time()-start)}),flush=True)
        if (step+1)%200==0 or step+1==args.steps:
            result=evaluate(model,validation,12)
            score=result['groups']['overall']['cer']
            validation_history.append({'step':step+1,'cer':score})
            print(json.dumps({'validation_step':step+1,'cer':score}),flush=True)
            if score<best_score:
                best_score=score
                best_state={key:value.detach().clone() for key,value in model.state_dict().items()}
    if best_state is None:raise RuntimeError('No validation checkpoint selected')
    model.load_state_dict(best_state)
    training_seconds=round(time.time()-start)
    destination=ROOT/'training/checkpoints/ocracle-mtg-latin.pt'
    torch.save({'model_state':model.state_dict(),'alphabet':alphabet,'languages':LANGUAGES,'steps':args.steps,'seed':20261019},destination)
    after_validation=evaluate(model,validation,12)
    after_test=evaluate(model,test)
    report={'model_name':'ocracle-mtg-latin-v1','languages':LANGUAGES,'alphabet_characters':len(alphabet),'seed':20261019,'steps':args.steps,'batch_size':args.batch_size,'backbone_frozen':True,'batch_norm_running_statistics_frozen':True,'head_learning_rate':.0003,'latin_row_preservation_coefficient':.01,'validation_history':validation_history,'selection':'Lowest fixed multilingual validation CER; test never used for checkpoint selection','training_seconds':training_seconds,'training_lines':len(train),'loss_history':history,'source_checkpoint_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'baseline_english_validation':baseline_validation,'baseline_english_test':baseline_test,'latin_validation':after_validation,'latin_test':after_test,'limitations':['Shared Latin profile only; other printed scripts remain unfinished.','Bounded held-out subset and one M21 title treatment; not all fonts or physical camera conditions.','Localized Oracle-like printed_text is used for synthetic lines; actual scans are separately scored.','Posterior confidence remains uncalibrated.']}
    (ROOT/'training/latin-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    model.eval()
    output=ROOT/'models/ocracle-mtg-latin.onnx'
    torch.onnx.export(model,torch.zeros(1,1,64,256),str(output),input_names=['image'],output_names=['log_probs'],dynamic_axes={'image':{0:'batch',3:'width'},'log_probs':{0:'time',1:'batch'}},opset_version=17,dynamo=False)
    config={'alphabet':alphabet,'input_height':64,'model_name':'ocracle-mtg-latin-v1','language':'mul-latin','languages':['en','es','fr','de','it','pt']}
    (ROOT/'models/latin-recognizer-config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    profile_directory=ROOT/'models/latin-profiles'
    profile_directory.mkdir(exist_ok=True)
    for language in LANGUAGES:
        profile=dict(config,language=language)
        (profile_directory/f'{language}.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    print('Latin test',json.dumps(after_test['groups']['overall']),flush=True)
    print('Exported '+str(output),flush=True)


if __name__=='__main__':
    main()
