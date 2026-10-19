#!/usr/bin/env python3
"""Bounded real-MTG benchmark, with a gated validation-selected pilot mode."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import paddle
from paddle.nn import functional as F
from PIL import Image, ImageFilter
from native import HERE, load_native
from adapt import expand_vocabulary
from inputs import fingerprint, tensor
from mtg_prepare import MANIFEST, SUMMARY, SEED, check
from mtg_augment import CONDITIONS, degrade
from ppocr.losses.rec_ctc_loss import CTCLoss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(scores, characters):
    previous = 0
    text = []
    for index in scores[0].argmax(-1):
        if index and index != previous:
            text.append(characters[index-1])
        previous = index
    return ''.join(text)


def distance(left, right):
    row = list(range(len(right)+1))
    for i, char in enumerate(left, 1):
        following = [i]
        for j, other in enumerate(right, 1):
            following.append(min(row[j]+1, following[-1]+1, row[j-1]+(char != other)))
        row = following
    return row[-1]


def input_for(record, condition='clean', seed=SEED):
    with Image.open(HERE/record['path']) as source:
        image, parameters = degrade(source, condition, seed)
    values = tensor(image)
    return paddle.to_tensor(values), parameters, fingerprint(values)


def supervised_loss(model, ctc, image, text, indices):
    logits = model.logits(image)
    minimum = len(text)+sum(a == b for a, b in zip(text, text[1:]))
    if logits.shape[1] < minimum:
        raise RuntimeError('Physical line cannot align to full CTC sequence')
    labels = paddle.to_tensor([[indices[char] for char in text]], dtype='int32')
    lengths = paddle.to_tensor([len(text)], dtype='int64')
    return ctc(logits, [image, labels, lengths])['loss']


def anchor_evaluation(student, teacher, anchors, old_classes):
    result = []
    with paddle.no_grad():
        for record in anchors:
            image, _, sha = input_for(record)
            old = F.log_softmax(teacher.logits(image), axis=-1)
            current = F.log_softmax(student.logits(image), axis=-1)
            teacher_text = decode(old.exp().numpy(), teacher.characters)
            student_text = decode(current.exp().numpy(), student.characters)
            kl = float((old.exp()*(old-current[:, :, :old_classes])).sum(-1).mean())
            result.append({'id': record['id'], 'tensor_sha256': sha, 'teacher_text': teacher_text,
                           'student_text': student_text, 'same_text': teacher_text == student_text,
                           'teacher_kl': kl})
    return {'training_anchor_lines': len(result), 'same_teacher_text': sum(x['same_text'] for x in result),
            'mean_teacher_kl': sum(x['teacher_kl'] for x in result)/len(result), 'examples': result,
            'interpretation': 'Prediction preservation on clean training anchors, not gold-label accuracy.'}


def evaluate(student, records):
    groups = {}
    examples = []
    with paddle.no_grad():
        for index, record in enumerate(records):
            conditions = ['clean', 'gaussian', 'downsample', 'jpeg']
            for condition in conditions:
                with Image.open(HERE/record['path']) as source:
                    image = source.convert('RGB')
                if condition == 'gaussian':
                    image = image.filter(ImageFilter.GaussianBlur(2.4))
                elif condition == 'downsample':
                    size = image.size
                    image = image.resize((max(1, round(size[0]*.3)), max(1, round(size[1]*.3))), Image.Resampling.BILINEAR)
                    image = image.resize(size, Image.Resampling.BILINEAR)
                elif condition == 'jpeg':
                    encoded = io.BytesIO()
                    image.save(encoded, format='JPEG', quality=35, subsampling=2)
                    encoded.seek(0)
                    image = Image.open(encoded).convert('RGB')
                values = tensor(image)
                prediction = decode(student(paddle.to_tensor(values)).numpy(), student.characters).strip()
                target = record['text']
                errors = distance(target, prediction)
                for key in ['all', record['origin'], 'language:'+record['language'],
                            'condition:'+record.get('condition', condition),
                            'italic:'+str(record.get('italic', False)).lower()]:
                    value = groups.setdefault(key, {'lines': 0, 'characters': 0, 'edits': 0, 'exact': 0})
                    value['lines'] += 1
                    value['characters'] += len(target)
                    value['edits'] += errors
                    value['exact'] += prediction == target
                examples.append({'id': record['id'], 'condition': condition, 'target': target, 'prediction': prediction,
                                 'tensor_sha256': fingerprint(values)})
    for value in groups.values():
        value['cer'] = value['edits']/max(1, value['characters'])
    return {'groups': groups, 'examples': examples,
            'normalization': 'Literal NFC labels; complete CTC output; strip predicted boundary whitespace only.'}


def selection_score(evaluated, kept):
    return (.9*evaluated['groups']['v2-title']['cer']+
            .1*(1-kept['same_teacher_text']/kept['training_anchor_lines']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['benchmark', 'pilot'], default='benchmark')
    parser.add_argument('--pilot-confirmed', action='store_true', help='Use only after explicit corpus/readiness confirmation')
    args = parser.parse_args()
    records, dataset = check()
    if args.mode == 'pilot' and (not args.pilot_confirmed or not dataset['selection_ready']):
        parser.error('Pilot requires explicit confirmation and the hash-frozen V2 validation corpus')
    steps = 5 if args.mode == 'benchmark' else 100
    paddle.set_device('cpu')
    paddle.seed(SEED)
    rng = random.Random(SEED)
    teacher, source = load_native()
    model, _ = load_native()
    for parameter in teacher.parameters():
        parameter.stop_gradient = True
    old_classes = len(teacher.characters)+1
    additions = expand_vocabulary(model, {'¡': '!', '„': '“', 'ẞ': 'ß'})
    indices = {char: i+1 for i, char in enumerate(model.characters)}
    physical = [r for r in records if r['role'] == 'supervised']
    synthetic = [r for r in records if r['role'] == 'synthetic']
    anchors = [r for r in records if r['role'] == 'teacher-anchor']
    validation = [r for r in records if r['role'] == 'validation']
    for row in physical+synthetic:
        if any(char not in indices for char in row['text']):
            raise RuntimeError('Reviewed label has an unconfigured Unicode class')
    pools = {origin: [r for r in physical if r['origin'] == origin]
             for origin in ['v2-title', 'v2-body', 'punctuation']}
    glyphs = [r for r in synthetic if r['kind'] == 'glyph-practice']
    initial_bn = {name: value.numpy().copy() for name, value in model.state_dict().items()
                  if name.endswith(('._mean', '._variance'))}
    groups = {'head': [], 'neck': [], 'upper_backbone': []}
    initial_parameters = {}
    for name, parameter in model.named_parameters():
        group = ('head' if name.startswith('ctc_head.') else 'neck' if name.startswith('ctc_encoder.') else
                 'upper_backbone' if name.startswith('backbone.blocks5.') else None)
        parameter.stop_gradient = group is None
        if group:
            groups[group].append(parameter)
            initial_parameters[name] = parameter.numpy().copy()
    optimizer = paddle.optimizer.Adam(learning_rate=1e-4,
        parameters=[{'params': groups['head']}, {'params': groups['neck'], 'learning_rate': .1},
                    {'params': groups['upper_backbone'], 'learning_rate': .05}],
        grad_clip=paddle.nn.ClipGradByGlobalNorm(4))
    fixed_anchors = sorted(anchors, key=lambda r: r['id'])[:8]
    ctc = CTCLoss()
    history, selections = [], []
    directory = HERE/'runs'/('mtg-'+args.mode)
    directory.mkdir(parents=True, exist_ok=True)
    process = {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
               'mode': args.mode, 'steps': steps, 'source_checkpoint_sha256': source['source_sha256'],
               'dataset_manifest_sha256': digest(MANIFEST), 'dataset_summary_sha256': digest(SUMMARY),
               'selection_protocol': dataset.get('selection_protocol')}
    (directory/'process.json').write_text(json.dumps(process, indent=2)+'\n')
    print(json.dumps({'event': 'initialized', **process}), flush=True)
    before = anchor_evaluation(model, teacher, fixed_anchors, old_classes)
    saved_checkpoints = []

    def save_checkpoint(step):
        path = directory/f'step-{step:03d}.pdparams'
        paddle.save(model.state_dict(), str(path))
        saved_checkpoints.append({'step': step, 'path': str(path.relative_to(HERE)), 'sha256': digest(path)})

    best_score, best_step, best_state = float('inf'), None, None
    baseline = None
    if args.mode == 'pilot':
        baseline = {'official_source': evaluate(teacher, validation),
                    'expanded_source': evaluate(model, validation)}
        best_score, best_step = selection_score(baseline['expanded_source'], before), 0
        best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        save_checkpoint(0)
        selections.append({'step': 0, 'score': best_score, 'validation': baseline['expanded_source'], 'clean_anchor_preservation': before})
        print(json.dumps({'event': 'baseline', 'official_source_groups': baseline['official_source']['groups'],
                          'expanded_source_groups': baseline['expanded_source']['groups'], 'step0_score': best_score,
                          'clean_anchors': {key: value for key, value in before.items() if key != 'examples'}}), flush=True)
    start = time.perf_counter()
    # Timing covers every intended real augmentation, all three punctuation
    # examples, a body and a title; it does not fit/select against validation.
    forced = sorted(pools['punctuation'], key=lambda r: r['id'])+[pools['v2-body'][0], pools['v2-title'][0]]
    for step in range(steps):
        step_start = time.perf_counter()
        if args.mode == 'benchmark':
            row = forced[step]
            condition = CONDITIONS[step+1]
        else:
            origin = rng.choices(['v2-title', 'v2-body', 'punctuation'], weights=[.35, .4, .25])[0]
            row = rng.choice(pools[origin])
            condition = rng.choices(CONDITIONS, weights=[.25, .2, .15, .1, .15, .15])[0]
        practice = rng.choice(glyphs if step%2 == 0 else synthetic)
        anchor = rng.choice(anchors)
        physical_input, parameters, physical_sha = input_for(row, condition, rng.randrange(2**32))
        synthetic_input, synthetic_parameters, synthetic_sha = input_for(practice, rng.choice(['clean', 'gaussian', 'downsample']), rng.randrange(2**32))
        anchor_input, _, anchor_sha = input_for(anchor)
        optimizer.clear_grad()
        physical_ctc = supervised_loss(model, ctc, physical_input, row['text'], indices)
        synthetic_ctc = supervised_loss(model, ctc, synthetic_input, practice['text'], indices)
        with paddle.no_grad():
            teacher_log = F.log_softmax(teacher.logits(anchor_input), axis=-1)
        student_log = F.log_softmax(model.logits(anchor_input), axis=-1)[:, :, :old_classes]
        preservation = (teacher_log.exp()*(teacher_log-student_log)).sum(-1).mean()
        loss = physical_ctc+.25*synthetic_ctc+preservation
        if not bool(paddle.isfinite(loss)) or any(layer.training for layer in model.sublayers()):
            raise RuntimeError('Nonfinite loss or unfrozen inference-mode module')
        loss.backward()
        gradients = {group: sum(float(paddle.sum(p.grad*p.grad)) for p in values if p.grad is not None)**.5
                     for group, values in groups.items()}
        if any(not np.isfinite(value) or value <= 0 for value in gradients.values()):
            raise RuntimeError('An adaptation group lost its real-MTG gradient')
        optimizer.step()
        entry = {'step': step+1, 'physical_id': row['id'], 'physical_origin': row['origin'],
                 'physical_text': row['text'], 'augmentation': parameters, 'physical_tensor_sha256': physical_sha,
                 'synthetic_id': practice['id'], 'synthetic_augmentation': synthetic_parameters,
                 'synthetic_tensor_sha256': synthetic_sha, 'anchor_id': anchor['id'],
                 'anchor_tensor_sha256': anchor_sha, 'physical_ctc': float(physical_ctc),
                 'synthetic_ctc': float(synthetic_ctc), 'teacher_kl': float(preservation), 'loss': float(loss),
                 'gradient_l2_before_clipping': gradients, 'update_seconds': time.perf_counter()-step_start}
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if args.mode == 'pilot' and (step+1)%25 == 0:
            save_checkpoint(step+1)
            evaluated = evaluate(model, validation)
            kept = anchor_evaluation(model, teacher, fixed_anchors, old_classes)
            score = selection_score(evaluated, kept)
            selections.append({'step': step+1, 'score': score, 'validation': evaluated, 'clean_anchor_preservation': kept})
            print(json.dumps({'event': 'validation', 'step': step+1, 'score': score,
                              'groups': evaluated['groups'], 'same_teacher_text': kept['same_teacher_text']}), flush=True)
            if score < best_score:
                best_score, best_step = score, step+1
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    training_seconds = time.perf_counter()-start
    if best_state is not None:
        model.set_state_dict(best_state)
    if any(not np.array_equal(value, model.state_dict()[name].numpy()) for name, value in initial_bn.items()):
        raise RuntimeError('Frozen BN running statistics changed')
    changed = {group: sum(float(np.square(model.state_dict()[name].numpy()-value).sum())
                         for name, value in initial_parameters.items()
                         if name.startswith({'head': 'ctc_head.', 'neck': 'ctc_encoder.', 'upper_backbone': 'backbone.blocks5.'}[group]))**.5
               for group in groups}
    after = anchor_evaluation(model, teacher, fixed_anchors, old_classes)
    checkpoint = directory/'recognizer.pdparams'
    paddle.save(model.state_dict(), str(checkpoint))
    (directory/'alphabet.txt').write_text(model.characters)
    result = {'schema_version': 1, 'mode': args.mode, 'steps': steps, 'seed': SEED,
              'source_checkpoint_sha256': source['source_sha256'], 'dataset_manifest_sha256': digest(MANIFEST),
              'dataset_summary_sha256': digest(SUMMARY), 'dataset_roles': dataset['roles'],
              'training_seconds': training_seconds, 'update_seconds': [x['update_seconds'] for x in history],
              'estimated_100_update_compute_seconds_excluding_selection': sum(x['update_seconds'] for x in history)/steps*100,
              'head_lr': 1e-4, 'neck_lr': 1e-5, 'upper_backbone_lr': 5e-6,
              'objective': 'physical CTC +0.25 synthetic CTC +1.0 clean old-teacher KL in full expanded softmax',
              'appended_classes': additions, 'classes': len(model.characters)+1,
              'batch_norm_state_tensors_unchanged': len(initial_bn), 'parameter_l2_change': changed,
              'before_clean_training_anchors': before, 'after_clean_training_anchors': after,
              'history': history, 'selection_history': selections, 'selected_step': best_step,
              'baseline_validation': baseline,
              'saved_checkpoints': saved_checkpoints, 'selection_protocol': dataset.get('selection_protocol'),
              'process': process,
              'validation_inference_performed': args.mode == 'pilot', 'test_inference_performed': False,
              'checkpoint': {'path': str(checkpoint.relative_to(HERE)), 'sha256': digest(checkpoint), 'distributed': False},
              'implementation_sha256': {name: digest(HERE/name) for name in
                                       ['mtg_prepare.py', 'mtg_augment.py', 'mtg_train.py', 'native.py', 'adapt.py', 'inputs.py']},
              'limitations': ['Timing/gradient benchmark is not an MTG accuracy improvement claim.',
                              'Existing normalized English scan labels are omitted; only teacher predictions are used.',
                              'No frozen test labels used for training or checkpoint selection.',
                              'Fresh development corpus is reserved and never used for this checkpoint choice.']}
    (HERE/('mtg-'+args.mode+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['mode', 'steps', 'training_seconds', 'estimated_100_update_compute_seconds_excluding_selection']}))


if __name__ == '__main__':
    main()
