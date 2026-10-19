#!/usr/bin/env python3
"""Export the exact three-step smoke checkpoint and prove ONNX/RTen/CLI parity."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import onnx
import onnxruntime as ort
import paddle
import paddle2onnx
from PIL import Image
from native import HERE, ROOT, Recognition, alphabet
from inputs import fingerprint, render, tensor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def greedy(probabilities, characters):
    text = []
    previous = 0
    for index in probabilities[0].argmax(-1):
        if index and index != previous:
            text.append(characters[index-1])
        previous = index
    return ''.join(text)


def snapshot(source, destination):
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def load_smoke(report, checkpoint, characters):
    if digest(checkpoint) != report['checkpoint']['sha256']:
        raise RuntimeError('Smoke weights differ from the frozen three-step result')
    if hashlib.sha256(characters.encode()).hexdigest() != report['alphabet_sha256']:
        raise RuntimeError('Expanded smoke dictionary differs')
    original = alphabet()
    if characters != original+''.join(x['character'] for x in report['appended_classes']):
        raise RuntimeError('Old dictionary order or appended classes changed')
    for item in report['appended_classes']:
        if characters[item['class_index']-1] != item['character']:
            raise RuntimeError('Appended Unicode class has the wrong index')
    for name, sha in report['implementation_sha256'].items():
        if digest(HERE/name) != sha:
            raise RuntimeError('Smoke implementation binding differs: '+name)
    model = Recognition(characters)
    missing, unexpected = model.set_state_dict(paddle.load(str(checkpoint)))
    if missing or unexpected:
        raise RuntimeError('Expanded checkpoint failed strict native loading')
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True,
                        help='Built training/experimental_paddle/runner raw-tensor probe')
    parser.add_argument('--cli', type=Path, default=ROOT/'target/release/ocracle')
    args = parser.parse_args()
    paddle.set_device('cpu')
    report = json.loads((HERE/'smoke.json').read_text())
    if report['status'] != 'native_autograd_and_vocabulary_smoke_passed' or report['steps'] != 3:
        raise RuntimeError('Export proof requires the frozen three-step smoke')
    if digest(HERE/'parity.json') != report['parity_report_sha256']:
        raise RuntimeError('Original exact-weight parity binding changed')
    directory = HERE/'runs/smoke'
    checkpoint = directory/'expanded-small.pdparams'
    characters = (directory/'alphabet.txt').read_text()
    native = load_smoke(report, checkpoint, characters)
    layer = load_smoke(report, checkpoint, characters)
    # Full graph tracing permits symbolic batch/width while keeping height48.
    specification = [paddle.static.InputSpec([None, 3, 48, None], dtype='float32', name='x')]
    layer = paddle.jit.to_static(layer, input_spec=specification, full_graph=True)
    prefix = directory/'inference'
    paddle.jit.save(layer, str(prefix))
    model_path = directory/'expanded-small.onnx'
    # Disable optimizer paths that can implicitly install additional packages.
    paddle2onnx.export(model_filename=str(prefix)+'.json',
                      params_filename=str(prefix)+'.pdiparams', save_file=str(model_path),
                      opset_version=17, auto_upgrade_opset=False,
                      enable_onnx_checker=True, enable_optimize=False, optimize_tool=None)
    graph = onnx.load(str(model_path))
    onnx.checker.check_model(graph, full_check=True)
    shape = graph.graph.input[0].type.tensor_type.shape.dim
    if not shape[0].dim_param or not shape[3].dim_param or shape[1].dim_value != 3 or shape[2].dim_value != 48:
        raise RuntimeError('Export lost the dynamic batch/width input contract')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
    cache = HERE/'cache'
    cache.mkdir(exist_ok=True)
    runner = snapshot(args.runner, cache/'paddle-rten-export-snapshot')
    cli = snapshot(args.cli, cache/'ocracle-export-snapshot')
    environment = {**os.environ, 'RAYON_NUM_THREADS': '2'}
    config = {'alphabet': characters, 'input_height': 48,
              'model_name': 'PP-OCRv6 small native three-step smoke; experimental',
              'language': 'en', 'architecture': 'LCNetV4 + LightSVTR; three-step native vocabulary smoke',
              'output_layout': 'NTC', 'output_values': 'probabilities',
              'input': {'color_mode': 'bgr', 'scale': 2/255, 'offset': -1,
                        'padding_width': 320, 'width_rounding': 'ceil', 'min_content_width': 1}}
    config_path = directory/'recognizer-config.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    rng = np.random.default_rng(20261019)
    probes = [('random-width320', rng.uniform(-1, 1, (1, 3, 48, 320)).astype('float32'), None),
              ('random-width513', rng.uniform(-1, 1, (1, 3, 48, 513)).astype('float32'), None),
              ('random-batch2-width320', rng.uniform(-1, 1, (2, 3, 48, 320)).astype('float32'), None)]
    for case_id, text, width, degraded in [('clean-anchor', 'An old book rests here.', 513, False),
                                          ('blur-expanded', '¡Ähre! „Été“ ẞ', 320, True)]:
        # Exact height48 avoids independent native/CLI resizing. These are new
        # procedural graph probes, not held-out recognition accuracy cases.
        image = render(text, degraded).resize((width, 48), Image.Resampling.BILINEAR)
        path = directory/(case_id+'.png')
        image.save(path)
        probes.append((case_id, tensor(image), path))
    records = []
    with paddle.no_grad():
        for case_id, image, image_path in probes:
            expected = native(paddle.to_tensor(image)).numpy()
            exported = session.run(None, {session.get_inputs()[0].name: image})[0]
            if expected.shape != exported.shape or exported.shape[-1] != len(characters)+1:
                raise RuntimeError('Expanded class count/output shape differs after export')
            error = float(np.max(np.abs(expected-exported)))
            equal = bool(np.array_equal(expected.argmax(-1), exported.argmax(-1)))
            if error > 5e-5 or not equal:
                raise RuntimeError('Adapted native/ONNX outputs differ')
            record = {'id': case_id, 'input_shape': list(image.shape), 'output_shape': list(exported.shape),
                      'tensor_sha256': fingerprint(image), 'native_onnx_max_absolute_probability_error': error,
                      'native_onnx_identical_argmax': equal,
                      'appended_classes_max_absolute_probability_error':
                          float(np.max(np.abs(expected[:, :, -3:]-exported[:, :, -3:]))) }
            if image.shape[0] == 1:
                input_path, output_path = cache/'export-input.f32', cache/'export-output.f32'
                image.tofile(input_path)
                process = subprocess.run([str(runner), str(model_path), str(input_path), str(image.shape[-1]), str(output_path)],
                                         check=True, capture_output=True, text=True, env=environment)
                actual = np.fromfile(output_path, dtype='<f4').reshape(exported.shape)
                rten_error = float(np.max(np.abs(actual-exported)))
                if not np.isfinite(actual).all() or rten_error > 5e-5 or not np.array_equal(actual.argmax(-1), exported.argmax(-1)):
                    raise RuntimeError('Adapted ONNX/RTen outputs differ')
                record.update({'onnx_rten_max_absolute_probability_error': rten_error,
                               'onnx_rten_identical_argmax': True, 'rten_stdout': process.stdout,
                               'rten_row_sum_max_error': float(np.max(np.abs(actual.sum(-1)-1))),
                               'native_text': greedy(expected, characters),
                               'onnx_text': greedy(exported, characters), 'rten_text': greedy(actual, characters)})
            if image_path:
                command = [str(cli), 'recognize', str(image_path), '--recognition-model', str(model_path),
                           '--model-config', str(config_path), '--line', '--passes', 'original']
                process = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
                result = json.loads(process.stdout)
                if result['model']['sha256'] != digest(model_path) or result['model']['language'] != config['language']:
                    raise RuntimeError('Actual CLI did not load the experimental weights/configuration')
                if result['text'] != record['rten_text']:
                    raise RuntimeError('Actual Rust CLI disagrees on a pre-resized procedural input')
                record.update({'image_path': str(image_path.relative_to(HERE)), 'image_sha256': digest(image_path),
                               'actual_cli': result, 'cli_matches_native_onnx_rten': True})
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key not in ['actual_cli']}), flush=True)
    result = {'schema_version': 1, 'status': 'adapted_native_onnx_rten_cli_parity_passed',
              'source_checkpoint_sha256': report['source_checkpoint_sha256'],
              'smoke_report_sha256': digest(HERE/'smoke.json'),
              'smoke_checkpoint_sha256': digest(checkpoint),
              'onnx': {'path': str(model_path.relative_to(HERE)), 'sha256': digest(model_path),
                       'bytes': model_path.stat().st_size, 'opset': 17,
                       'operators': dict(sorted(Counter(x.op_type for x in graph.graph.node).items())),
                       'classes': len(characters)+1, 'dynamic_batch': True, 'dynamic_width': True},
              'configuration_sha256': digest(config_path), 'appended_classes': report['appended_classes'],
              'alphabet_sha256': hashlib.sha256(characters.encode()).hexdigest(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ['paddlepaddle', 'paddle2onnx', 'onnx', 'onnxruntime', 'numpy', 'Pillow']},
              'paddle2onnx_source': {'repository': 'PaddlePaddle/Paddle2ONNX', 'release': 'v2.1.0',
                                    'revision': 'c8b5048c3a0903986bd3ec1cce2af9915b391c49', 'license': 'Apache-2.0'},
              'cli_sha256': digest(cli), 'runner_sha256': digest(runner), 'rten_version': '0.26.0',
              'runner_source_sha256': digest(ROOT/'training/experimental_paddle/runner/src/main.rs'),
              'runner_lock_sha256': digest(ROOT/'training/experimental_paddle/runner/Cargo.lock'),
              'implementation_sha256': digest(Path(__file__)), 'records': records,
              'limitations': ['Only the existing three-step procedural smoke checkpoint; no MTG accuracy claim.',
                              'CLI checks use pre-resized height48 PNGs so shared tensor agreement is meaningful.',
                              'Raw-tensor Rust probe is batch1 only; dynamic batch2 checked in native/ONNX Runtime.',
                              'No production profile/model changes; exported weights remain ignored local outputs.']}
    (HERE/'export.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
