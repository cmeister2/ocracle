#!/usr/bin/env python3
"""Validation-selected Latin V2 visual fine-tuning; V1 artifacts stay immutable."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import random
import time
import unicodedata
import torch
from torch import nn
import train as common
from network import ALPHABET,adapt_model,decode,load_model
from latin_v2_acquire import ROOT,CORPUS,DATA
from latin_v2_prepare import LANGUAGES,SEED


def fixed_bn(model):
    """Train visual/recurrent parameters without changing running statistics."""
    model.train()
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()


def initialize(alphabet):
    checkpoint=ROOT/'models/training/latin-v1-source.pt'
    source_metadata=json.loads((ROOT/'models/training/latin-v1-source.json').read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source_metadata['sha256']:raise RuntimeError('Pinned V1 initialization checkpoint changed')
    source=load_model(checkpoint)
    model=adapt_model(checkpoint,alphabet,SEED)
    # Accent rows start near a known base-letter classifier. This is a prior,
    # not a claim that an untrained row recognizes the accented glyph.
    rows={c:i+1 for i,c in enumerate(source.alphabet)}
    with torch.no_grad():
        for index,char in enumerate(alphabet):
            if char in rows:continue
            plain=''.join(c for c in unicodedata.normalize('NFD',char) if not unicodedata.combining(c))
            if len(plain)==1 and plain in rows:
                model.output[0].weight[index+1].copy_(source.output[0].weight[rows[plain]])
                model.output[0].bias[index+1].copy_(source.output[0].bias[rows[plain]]-.75)
    return model


def selected_validation(records):
    result=[r for r in records if r['split']=='validation' and r['corpus_generation']=='latin-v2-physical']
    english=[r for r in records if r['split']=='validation' and r['language']=='en']
    random.Random(SEED).shuffle(english)
    real=[r for r in english if r['source']=='real-scan']
    synthetic=[r for r in english if r['source']!='real-scan']
    return result+real[:8]+synthetic[:8]


@torch.inference_mode()
def evaluate(model,selection,conditions=('clean','severe')):
    model.eval();stats=defaultdict(lambda:{'lines':0,'characters':0,'errors':0,'exact_lines':0});examples=[]
    for severity in conditions:
        for index,record in enumerate(selection):
            item=common.tensor(record,severity,SEED+index)
            raw=decode(model(item[None])[:,0].argmax(-1).tolist(),model.alphabet)
            prediction=raw.strip();errors=common.distance(record['text'],prediction)
            groups=['overall','language:'+record['language'],'severity:'+severity,'source:'+record['source'],'generation:'+record['corpus_generation']]
            if record['corpus_generation']=='latin-v2-physical':groups+=['new-physical','new-physical:'+severity]
            if record['language']=='en':groups+=['english-preservation']
            for group in groups:
                s=stats[group];s['lines']+=1;s['characters']+=len(record['text']);s['errors']+=errors;s['exact_lines']+=int(record['text']==prediction)
            examples.append({'target':record['text'],'prediction':prediction,'raw_prediction':raw,'language':record['language'],'severity':severity,'card_name':record['card_name'],'source':record['source']})
    for s in stats.values():s.update(cer=s['errors']/max(1,s['characters']),exact_line_accuracy=s['exact_lines']/max(1,s['lines']))
    return {'groups':dict(stats),'examples':examples,'unique_lines':len(selection),'conditions':list(conditions),'normalization':'NFC labels; full CTC output; strip predicted boundary whitespace'}


def selection_score(result):
    physical=result['groups']['new-physical'];english=result['groups']['english-preservation']
    return .75*physical['cer']+.15*(1-physical['exact_line_accuracy'])+.1*english['cer']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--threads',type=int,default=4);parser.add_argument('--batch-size',type=int,default=4);args=parser.parse_args()
    if args.steps<1 or args.threads<1 or args.batch_size<2:parser.error('steps and threads must be positive; batch-size must be at least 2 for English and localized samples')
    torch.set_num_threads(args.threads);torch.manual_seed(SEED);rng=random.Random(SEED);common.DATA=DATA
    records=[json.loads(line) for line in (DATA/'manifest.jsonl').read_text().splitlines()]
    summary=json.loads((CORPUS/'summary.json').read_text())
    if hashlib.sha256((DATA/'manifest.jsonl').read_bytes()).hexdigest()!=summary['manifest_sha256']:raise RuntimeError('Training input differs from frozen manifest')
    alphabet=json.loads((CORPUS/'alphabet.json').read_text())['alphabet']
    validation=selected_validation(records)
    train=[r for r in records if r['split']=='train']
    bylang={language:[r for r in train if r['language']==language] for language in LANGUAGES}
    physical={language:[r for r in bylang[language] if r['source']=='real-scan'] for language in LANGUAGES}
    glyphs={language:[r for r in bylang[language] if r['source']=='glyph-curriculum'] for language in LANGUAGES}
    clean_english=[r for r in train if r['language']=='en' and all(c in ALPHABET for c in r['text'])]
    teacher=load_model(ROOT/'training/checkpoints/base.pt').eval()
    for p in teacher.parameters():p.requires_grad_(False)
    model=initialize(alphabet)
    initial_bn={key:value.detach().clone() for key,value in model.state_dict().items() if 'running_' in key or 'num_batches_tracked' in key}
    initial_features={key:model.state_dict()[key].detach().clone() for key in ['conv.13.weight','gru.weight_ih_l0']}
    sampled_characters=Counter()
    print('Scoring frozen V1 and generic source on fixed validation only',flush=True)
    v1=load_model(ROOT/'models/training/latin-v1-source.pt')
    before=evaluate(v1,validation);del v1
    generic=evaluate(teacher,validation)
    for p in model.conv.parameters():p.requires_grad_(False)
    for index,module in enumerate(model.conv):
        if index>=13:
            for p in module.parameters():p.requires_grad_(True)
    visual=[p for p in model.conv.parameters() if p.requires_grad]+list(model.gru.parameters())
    optimizer=torch.optim.AdamW([{'params':visual,'lr':.00001},{'params':model.output.parameters(),'lr':.0002}],weight_decay=.00001)
    ctc=nn.CTCLoss();indices={c:i+1 for i,c in enumerate(alphabet)}
    oldrows=torch.tensor([0]+[indices[c] for c in ALPHABET],dtype=torch.long)
    history=[];validation_history=[];best_score=float('inf');best_state=None;best_step=None;start=time.time()
    for step in range(args.steps):
        fixed_bn(model)
        # Each update includes one clean English anchor, then balanced languages.
        chosen=[rng.choice(clean_english)]
        for _ in range(args.batch_size-1):
            language=rng.choice(LANGUAGES);chance=rng.random()
            pool=physical[language] if chance<.6 and physical[language] else glyphs[language] if chance<.75 and glyphs[language] else bylang[language]
            chosen.append(rng.choice(pool))
        tensors=[common.tensor(r,'clean' if i==0 else rng.choice(['clean','clean','clean','gaussian','severe','motion','downsample','jpeg','combined']),rng.randrange(2**32)) for i,r in enumerate(chosen)]
        images,lengths=common.batch(tensors)
        targets=torch.tensor([indices[c] for r in chosen for c in r['text']],dtype=torch.long);target_lengths=torch.tensor([len(r['text']) for r in chosen],dtype=torch.long)
        sampled_characters.update(c for r in chosen for c in r['text'])
        optimizer.zero_grad(set_to_none=True);pred=model(images)
        loss=ctc(pred,targets,lengths,target_lengths)
        # Teacher sees the identical clean unpadded English line. Student row
        # probabilities are not renormalized: new accent mass is penalized too.
        with torch.no_grad():teacher_scores=teacher(tensors[0][None])[:,0]
        student_scores=model(tensors[0][None])[:,0].index_select(-1,oldrows)
        preservation=(teacher_scores.exp()*(teacher_scores-student_scores)).sum(-1).mean()
        total=loss+.5*preservation
        if not torch.isfinite(total):raise RuntimeError('Nonfinite V2 loss')
        total.backward()
        # Head warm-up avoids disrupting visual features with initially random rows.
        if step<150:
            for p in visual:p.grad=None
        nn.utils.clip_grad_norm_(model.parameters(),4);optimizer.step()
        history.append({'ctc':float(loss.detach()),'teacher_kl':float(preservation.detach()),'total':float(total.detach())})
        if (step+1)%20==0:print(json.dumps({'step':step+1,'loss':sum(x['total'] for x in history[-20:])/20,'elapsed_seconds':round(time.time()-start)}),flush=True)
        if (step+1)%200==0 or step+1==args.steps:
            result=evaluate(model,validation);score=selection_score(result)
            row={'step':step+1,'score':score,'groups':result['groups']};validation_history.append(row);print(json.dumps({'validation':row}),flush=True)
            if score<best_score:best_score=score;best_step=step+1;best_state={key:value.detach().clone() for key,value in model.state_dict().items()}
    model.load_state_dict(best_state)
    if any(not torch.equal(value,model.state_dict()[key]) for key,value in initial_bn.items()):raise RuntimeError('Frozen batch-normalization running statistics changed')
    checkpoint=ROOT/'training/checkpoints/ocracle-mtg-latin-v2.pt'
    torch.save({'model_state':model.state_dict(),'alphabet':alphabet,'languages':LANGUAGES,'selected_step':best_step,'seed':SEED},checkpoint)
    after=evaluate(model,validation)
    # Test labels enter only after checkpoint selection is finished.
    test=[r for r in records if r['split']=='test' and r['corpus_generation']=='latin-v2-physical']
    test_result=evaluate(model,test,('clean','severe','combined'))
    report={'model_name':'ocracle-mtg-latin-v2','languages':LANGUAGES,'seed':SEED,'steps':args.steps,'selected_step':best_step,'batch_size':args.batch_size,'training_seconds':round(time.time()-start),'head_learning_rate':.0002,'visual_learning_rate':.00001,'head_warmup_steps':150,'teacher_kl_coefficient':.5,'teacher':'Official generic English base; clean training inputs only','batch_norm_running_statistics_frozen':True,'trainable_backbone':'conv modules index >=13 and all GRU parameters','selection':'0.75*new-physical validation CER +0.15*(1-new-physical exact accuracy)+0.10*English validation CER; clean and severe; test never selected','manifest_sha256':summary['manifest_sha256'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'validation_history':validation_history,'v1_validation':before,'generic_validation':generic,'v2_validation':after,'v2_test':test_result,'loss_history':history,'limitations':['Latin V2 CRNN experiment only; unregistered pending independent physical evaluation.','Exemplar glyph curriculum establishes training coverage, not physical-font accuracy.','All printed languages, arbitrary fonts and degradation conditions remain unproven.']}
    report['source_checkpoint']=json.loads((ROOT/'models/training/latin-v1-source.json').read_text())
    report['sampled_training_character_counts']=dict(sorted(sampled_characters.items()))
    report['feature_parameter_l2_changes']={key:float((model.state_dict()[key]-value).norm()) for key,value in initial_features.items()}
    report['unchanged_batch_norm_buffers']=len(initial_bn)
    (ROOT/'training/latin-v2-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    model.eval();onnx=ROOT/'models/ocracle-mtg-latin-v2.onnx'
    torch.onnx.export(model,torch.zeros(1,1,64,256),str(onnx),input_names=['image'],output_names=['log_probs'],dynamic_axes={'image':{0:'batch',3:'width'},'log_probs':{0:'time',1:'batch'}},opset_version=17,dynamo=False)
    config={'alphabet':alphabet,'input_height':64,'model_name':'ocracle-mtg-latin-v2','language':'mul-latin','languages':LANGUAGES}
    (ROOT/'models/latin-v2-recognizer-config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    directory=ROOT/'models/latin-v2-profiles';directory.mkdir(exist_ok=True)
    for language in LANGUAGES:(directory/f'{language}.json').write_text(json.dumps(dict(config,language=language),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'selected_step':best_step,'test':test_result['groups'],'export':str(onnx)}),flush=True)


if __name__=='__main__':main()
