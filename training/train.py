#!/usr/bin/env python3
"""Fine-tune the CRNN on MTG lines with blur, motion, low resolution and JPEG.

Evaluation uses separate card-name groups and fixed degradations. No database
lookup or nearest-card correction is involved in either training or scoring.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image, ImageFilter
import torch
from network import ALPHABET, decode, load_model

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/data'


def degraded(image, severity, seed):
    rng=random.Random(seed)
    if severity=='clean':
        return image
    if severity=='gaussian':
        return image.filter(ImageFilter.GaussianBlur(rng.uniform(.4,1.0)))
    if severity=='severe':
        return image.filter(ImageFilter.GaussianBlur(rng.uniform(1.6,2.8)))
    if severity=='motion':
        kernel=rng.choice([3,5,7])
        weights=[0.]*(kernel*kernel)
        for i in range(kernel):
            weights[(kernel//2)*kernel+i]=1/kernel
        return image.filter(ImageFilter.Kernel((kernel,kernel),weights)) if kernel<=5 else motion_numpy(image,7)
    if severity=='downsample':
        factor=rng.uniform(.25,.55)
        return image.resize((max(1,int(image.width*factor)),max(1,int(image.height*factor))),Image.Resampling.BILINEAR).resize(image.size,Image.Resampling.BICUBIC)
    if severity=='jpeg':
        buffer=io.BytesIO()
        image.save(buffer,format='JPEG',quality=rng.randint(18,45))
        buffer.seek(0)
        return Image.open(buffer).convert('L')
    if severity=='combined':
        image=degraded(image,'downsample',seed)
        image=degraded(image,'gaussian',seed+1)
        return degraded(image,'jpeg',seed+2)
    raise ValueError(severity)


def motion_numpy(image,kernel):
    array=np.asarray(image,dtype=np.float32)
    padded=np.pad(array,((0,0),(kernel//2,kernel//2)),mode='edge')
    result=sum(padded[:,i:i+image.width] for i in range(kernel))/kernel
    return Image.fromarray(np.uint8(result.clip(0,255)))


def tensor(record,severity,seed):
    with Image.open(DATA/record['path']) as image:
        image=degraded(image.convert('L'),severity,seed)
        width=max(16,round(image.width*64/image.height))
        image=image.resize((width,64),Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(image,dtype=np.float32)/255-.5).unsqueeze(0)


def batch(items):
    lengths=[x.shape[-1] for x in items]
    images=torch.full((len(items),1,64,max(lengths)),.5)
    for i,x in enumerate(items):
        images[i,:,:,:x.shape[-1]]=x
    return images,torch.tensor([n//4+1 for n in lengths],dtype=torch.long)


def distance(left,right):
    prior=list(range(len(right)+1))
    for i,a in enumerate(left):
        current=[i+1]
        for j,b in enumerate(right):
            current.append(min(current[-1]+1,prior[j+1]+1,prior[j]+(a!=b)))
        prior=current
    return prior[-1]


@torch.inference_mode()
def evaluate(model,records,limit,seed):
    model.eval()
    rng=random.Random(seed)
    records=list(records)
    rng.shuffle(records)
    if limit:
        scans=[r for r in records if r['source']=='real-scan']
        synthetic=[r for r in records if r['source']!='real-scan']
        # Preserve actual printed-font examples in the bounded evaluation.
        count=min(len(scans),max(1,limit//3))
        records=scans[:count]+synthetic[:limit-count]
    severities=['clean','gaussian','severe','motion','downsample','jpeg','combined']
    stats=defaultdict(lambda:{'lines':0,'characters':0,'character_errors':0,'exact_lines':0})
    examples=[]
    for severity in severities:
        for index,record in enumerate(records):
            item=tensor(record,severity,seed+index)
            result=model(item.unsqueeze(0))[:,0].argmax(-1).tolist()
            raw_prediction=decode(result)
            prediction=raw_prediction.strip()
            error=distance(record['text'],prediction)
            raw=stats['raw-untrimmed']
            raw['lines']+=1
            raw['characters']+=len(record['text'])
            raw['character_errors']+=distance(record['text'],raw_prediction)
            raw['exact_lines']+=int(record['text']==raw_prediction)
            groups=['overall',f'severity:{severity}',f"style:{'italic' if record['italic'] else 'regular'}",f"font:{record['font']}",f"source:{record['source']}"]
            for key in groups:
                entry=stats[key]
                entry['lines']+=1
                entry['characters']+=len(record['text'])
                entry['character_errors']+=error
                entry['exact_lines']+=int(record['text']==prediction)
            if index<3:
                examples.append({'severity':severity,'target':record['text'],'prediction':prediction,'card_name':record['card_name'],'font':record['font']})
    for entry in stats.values():
        entry['cer']=entry['character_errors']/max(1,entry['characters'])
        entry['exact_line_accuracy']=entry['exact_lines']/max(1,entry['lines'])
    return {'card_identities':len({r['card_name'].casefold() for r in records}),'unique_lines':len(records),'degradations_per_line':len(severities),'groups':dict(stats),'examples':examples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'training/checkpoints/base.pt')
    parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--learning-rate',type=float,default=.00003)
    parser.add_argument('--eval-limit',type=int,default=48,help='Unique validation/test lines, each scored in seven degradation conditions; 0 means all')
    parser.add_argument('--seed',type=int,default=191993)
    parser.add_argument('--output',type=Path,default=ROOT/'models/ocracle-mtg.onnx')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    records=[json.loads(line) for line in (DATA/'manifest.jsonl').read_text().splitlines()]
    # Fail closed if any oracle identity/name crosses split boundaries.
    groups=defaultdict(set)
    for record in records:
        groups[record['card_name'].casefold()].add(record['split'])
    assert all(len(s)==1 for s in groups.values()),'card-name split leakage'
    train=[r for r in records if r['split']=='train']
    validation=[r for r in records if r['split']=='validation']
    test=[r for r in records if r['split']=='test']
    model=load_model(args.checkpoint)
    print('Baseline validation start',flush=True)
    baseline_validation=evaluate(model,validation,args.eval_limit,args.seed)
    print('baseline validation',json.dumps(baseline_validation['groups']['overall']),flush=True)
    # Test is evaluated only before and after the fixed training schedule. It is
    # never used for checkpoint/learning-rate selection.
    baseline_test=evaluate(model,test,args.eval_limit,args.seed+1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.00001)
    loss_function=torch.nn.CTCLoss(zero_infinity=False)
    char_indices={c:i+1 for i,c in enumerate(ALPHABET)}
    history=[]
    start=time.time()
    model.train()
    severities=['clean','clean','gaussian','severe','motion','downsample','jpeg','combined']
    for step in range(args.steps):
        scans=[r for r in train if r['source']=='real-scan']
        selection=[rng.choice(scans if scans and rng.random()<.25 else train) for _ in range(args.batch_size)]
        # Keep batches modest by avoiding excessively wide synthetic lines.
        items=[tensor(r,rng.choice(severities),rng.randrange(2**32)) for r in selection]
        images,lengths=batch(items)
        targets=torch.tensor([char_indices[c] for r in selection for c in r['text']],dtype=torch.long)
        target_lengths=torch.tensor([len(r['text']) for r in selection],dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        predictions=model(images)
        loss=loss_function(predictions,targets,lengths,target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite CTC loss on step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),4)
        optimizer.step()
        history.append(float(loss.detach()))
        if (step+1)%10==0:
            print(json.dumps({'step':step+1,'loss_last_10':sum(history[-10:])/10,'elapsed_seconds':round(time.time()-start)}),flush=True)
    training_seconds=round(time.time()-start)
    output_checkpoint=ROOT/'training/checkpoints/ocracle-mtg.pt'
    output_checkpoint.parent.mkdir(exist_ok=True)
    torch.save({'model_state':model.state_dict(),'steps':args.steps,'seed':args.seed},output_checkpoint)
    after_validation=evaluate(model,validation,args.eval_limit,args.seed)
    after_test=evaluate(model,test,args.eval_limit,args.seed+1)
    report={'architecture':'CRNN: convolutional visual features, 2-layer bidirectional GRU, CTC','checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'seed':args.seed,'steps':args.steps,'batch_size':args.batch_size,'learning_rate':args.learning_rate,'threads':args.threads,'elapsed_training_seconds':training_seconds,'elapsed_training_and_final_evaluation_seconds':round(time.time()-start),'evaluation_normalization':'Strip leading/trailing whitespace for primary metrics; raw-untrimmed group also recorded.', 'real_scan_sampling_probability':0.25,'training_real_scan_lines':sum(r['source']=='real-scan' for r in train),'training_lines':len(train),'training_card_identities':len({r['card_name'].casefold() for r in train}),'validation_lines_available':len(validation),'test_lines_available':len(test),'eval_unique_line_limit':args.eval_limit,'loss_history':history,'baseline_validation':baseline_validation,'fine_tuned_validation':after_validation,'baseline_test':baseline_test,'fine_tuned_test':after_test,'limitations':['Synthetic font faces are open substitutes, not proprietary Magic fonts. See font-coverage.json.','English/Latin alphabet; mana symbols and non-Latin scripts not trained.','Fixed held-out subset, not a comprehensive card benchmark.','Blur may destroy information; neural posteriors are not calibrated correctness probabilities.']}
    (ROOT/'training/report.json').write_text(json.dumps(report,indent=2)+'\n')
    args.output.parent.mkdir(exist_ok=True)
    model.eval()
    dummy=torch.zeros(1,1,64,256)
    torch.onnx.export(model,dummy,str(args.output),input_names=['image'],output_names=['log_probs'],dynamic_axes={'image':{0:'batch',3:'width'},'log_probs':{0:'time',1:'batch'}},opset_version=17,dynamo=False)
    print('Fine-tuned test',json.dumps(after_test['groups']['overall']),flush=True)
    print('Exported',args.output,flush=True)


if __name__=='__main__':
    main()
