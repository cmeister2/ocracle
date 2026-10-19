#!/usr/bin/env python3
"""Three CPU updates prove native autograd/append-only vocabulary feasibility.

This is deliberately not an accuracy experiment or a production checkpoint.
"""
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import paddle
from paddle.nn import functional as F
from native import HERE, Recognition, load_native
from adapt import expand_vocabulary
from inputs import FONT, FONT_LICENSE, FONT_SHA256, fingerprint, render, tensor
from ppocr.losses.rec_ctc_loss import CTCLoss

SEED = 20261019


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3:
        parser.error('This bounded proof permits 1–3 updates; it is not a training run')
    gate = json.loads((HERE/'parity.json').read_text())
    if gate['status'] != 'exact_weight_forward_parity_passed':
        raise RuntimeError('Run and pass parity.py before smoke training')
    paddle.set_device('cpu')
    paddle.seed(SEED)
    teacher, source = load_native()
    student, _ = load_native()
    if source['source_sha256'] != gate['native_checkpoint']['source_sha256']:
        raise RuntimeError('Parity gate is for a different source checkpoint')
    for parameter in teacher.parameters():
        parameter.stop_gradient = True
    old_classes = len(teacher.characters)+1
    anchor_text = 'An old book rests here.'
    expanded_text = '¡Ähre! „Été“ ẞ'
    anchor_np = tensor(render(anchor_text))
    expanded_np = tensor(render(expanded_text, degraded=True))
    anchor = paddle.to_tensor(anchor_np)
    expanded = paddle.to_tensor(expanded_np)
    with paddle.no_grad():
        original_logits = student.logits(anchor).numpy().copy()
    additions = expand_vocabulary(student, {'¡': '!', '„': '“', 'ẞ': 'ß'})
    with paddle.no_grad():
        expanded_logits = student.logits(anchor).numpy()
    old_error = float(np.max(np.abs(expanded_logits[:, :, :old_classes]-original_logits)))
    if old_error > .00005:
        raise RuntimeError('Expanding the classifier changed old-class logits')
    # Mean/variance tensors are Paddle state entries, not ordinary parameters.
    initial_bn = {name: value.numpy().copy() for name, value in student.state_dict().items()
                  if name.endswith(('._mean', '._variance'))}
    groups = {'head': [], 'neck': [], 'upper_backbone': []}
    initial = {}
    for name, parameter in student.named_parameters():
        group = ('head' if name.startswith('ctc_head.') else
                 'neck' if name.startswith('ctc_encoder.') else
                 'upper_backbone' if name.startswith('backbone.blocks5.') else None)
        parameter.stop_gradient = group is None
        if group:
            groups[group].append(parameter)
            initial[name] = parameter.numpy().copy()
    optimizer = paddle.optimizer.Adam(
        learning_rate=.0001,
        parameters=[{'params': groups['head']},
                    {'params': groups['neck'], 'learning_rate': .1},
                    {'params': groups['upper_backbone'], 'learning_rate': .1}],
        grad_clip=paddle.nn.ClipGradByGlobalNorm(4))
    ctc = CTCLoss()
    indices = {char: index+1 for index, char in enumerate(student.characters)}
    targets = paddle.to_tensor([[indices[char] for char in expanded_text]], dtype='int32')
    lengths = paddle.to_tensor([len(expanded_text)], dtype='int64')
    minimum_steps = len(expanded_text)+sum(a == b for a, b in zip(expanded_text, expanded_text[1:]))
    with paddle.no_grad():
        teacher_log_probs = F.log_softmax(teacher.logits(anchor), axis=-1)
    history = []
    start = time.time()
    for step in range(args.steps):
        if any(layer.training for layer in student.sublayers()):
            raise RuntimeError('An inference-pooling/dropout/BN module entered train mode')
        optimizer.clear_grad()
        logits = student.logits(expanded)
        if logits.shape[1] < minimum_steps:
            raise RuntimeError('Target cannot align to the complete CTC output')
        loss = ctc(logits, [expanded, targets, lengths])['loss']
        student_log_probs = F.log_softmax(student.logits(anchor), axis=-1)[:, :, :old_classes]
        # Full-softmax probabilities, not renormalized old rows: new-class mass
        # is penalized on the same clean input seen by the frozen teacher.
        preservation = (teacher_log_probs.exp()*(teacher_log_probs-student_log_probs)).sum(-1).mean()
        total = loss+.5*preservation
        if not bool(paddle.isfinite(total)):
            raise RuntimeError('Nonfinite CTC/preservation objective')
        total.backward()
        gradients = {group: float(sum(float(paddle.sum(p.grad*p.grad)) for p in params if p.grad is not None)**.5)
                     for group, params in groups.items()}
        if any(not np.isfinite(value) or value <= 0 for value in gradients.values()):
            raise RuntimeError('Missing/nonfinite native gradient in an adaptation group')
        optimizer.step()
        row = {'step': step+1, 'ctc': float(loss), 'teacher_kl': float(preservation),
               'total': float(total), 'gradient_l2_before_clipping': gradients}
        history.append(row)
        print(json.dumps(row), flush=True)
    if any(not np.array_equal(value, student.state_dict()[name].numpy()) for name, value in initial_bn.items()):
        raise RuntimeError('Frozen batch-normalization statistics changed')
    changed = {group: float(sum(float(np.square(student.state_dict()[name].numpy()-value).sum())
                              for name, value in initial.items()
                              if name.startswith({'head': 'ctc_head.', 'neck': 'ctc_encoder.',
                                                  'upper_backbone': 'backbone.blocks5.'}[group]))**.5)
               for group in groups}
    if any(value <= 0 for value in changed.values()):
        raise RuntimeError('An intended native feature group did not update')
    output = HERE/'runs/smoke'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'expanded-small.pdparams'
    paddle.save(student.state_dict(), str(checkpoint))
    (output/'alphabet.txt').write_text(student.characters)
    restored = Recognition(student.characters)
    missing, unexpected = restored.set_state_dict(paddle.load(str(checkpoint)))
    if missing or unexpected:
        raise RuntimeError('Expanded checkpoint cannot reload strictly')
    with paddle.no_grad():
        reload_error = float(paddle.max(paddle.abs(student(expanded)-restored(expanded))))
    if reload_error != 0:
        raise RuntimeError('Expanded checkpoint reload changed native inference')
    report = {'schema_version': 1, 'status': 'native_autograd_and_vocabulary_smoke_passed',
              'seed': SEED, 'steps': args.steps, 'python': platform.python_version(),
              'paddle_version': paddle.__version__, 'source_checkpoint_sha256': source['source_sha256'],
              'parity_report_sha256': hashlib.sha256((HERE/'parity.json').read_bytes()).hexdigest(),
              'original_classes_including_blank': old_classes,
              'expanded_classes_including_blank': len(student.characters)+1,
              'alphabet_sha256': hashlib.sha256(student.characters.encode()).hexdigest(),
              'appended_classes': additions, 'old_classifier_columns_preserved_before_update': True,
              'max_old_logit_error_after_expansion_before_update': old_error,
              'unchanged_batch_norm_state_tensors': len(initial_bn), 'model_mode': 'eval; autograd enabled',
              'learning_rates': {'head': .0001, 'neck': .00001, 'upper_backbone': .00001},
              'parameter_l2_change': changed, 'loss_history': history,
              'checkpoint_reload_max_probability_error': reload_error,
              'implementation_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                        for name in ['native.py', 'adapt.py', 'inputs.py', 'smoke.py']},
              'inputs': [{'text': anchor_text, 'role': 'clean frozen teacher anchor',
                          'tensor_sha256': fingerprint(anchor_np), 'shape': list(anchor_np.shape)},
                         {'text': expanded_text, 'role': 'new classes; italic; Gaussian .8; .7 downsample',
                          'tensor_sha256': fingerprint(expanded_np), 'shape': list(expanded_np.shape)}],
              'font': {'path': str(FONT), 'sha256': FONT_SHA256, 'license': FONT_LICENSE},
              'checkpoint': {'path': str(checkpoint.relative_to(HERE)),
                             'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             'bytes': checkpoint.stat().st_size, 'distributed': False},
              'elapsed_seconds': round(time.time()-start, 3),
              'limitations': ['Fresh procedural inputs only; no MTG adaptation or held-out accuracy claim.',
                              'Original NRTR auxiliary branch/loss omitted.',
                              'PIL bilinear smoke preprocessing; deployment uses image-crate triangle resize.',
                              'No export, runtime change or profile registration.']}
    (HERE/'smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
