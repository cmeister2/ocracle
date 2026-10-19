#!/usr/bin/env python3
"""Verify dynamic ONNX/PyTorch parity and actual held-out Rust transcripts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from network import decode,load_model

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'target/release/ocracle')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'training/checkpoints/ocracle-mtg.pt')
    parser.add_argument('--onnx',type=Path,default=ROOT/'models/ocracle-mtg.onnx')
    parser.add_argument('--model',type=Path,default=ROOT/'models/ocracle-mtg.rten')
    parser.add_argument('--config',type=Path,default=ROOT/'models/recognizer-config.json')
    parser.add_argument('--report',type=Path,default=ROOT/'training/export-validation.json')
    parser.add_argument('--fixture',type=Path,action='append',help='Actual held-out image; repeat for multiple language profiles')
    args=parser.parse_args()
    torch.set_num_threads(1)
    model=load_model(args.checkpoint).eval()
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    session=ort.InferenceSession(str(args.onnx),options,providers=['CPUExecutionProvider'])
    checks=[]
    for batch,width in [(1,64),(1,273),(2,160)]:
        pixels=np.zeros((batch,1,64,width),dtype=np.float32)
        with torch.inference_mode():
            native=model(torch.from_numpy(pixels)).numpy()
        exported=session.run(None,{'image':pixels})[0]
        error=float(np.max(np.abs(native-exported)))
        assert native.shape==exported.shape
        assert error<.0005,f'ONNX parity error {error}'
        checks.append({'batch':batch,'width':width,'shape':list(native.shape),'max_log_probability_error':error})
    fixtures=[]
    paths=args.fixture or [ROOT/'docs/gallery/assets'/filename for filename in ['beleren-title.png','modern-flavor.png','italic-reminder.png']]
    for path in paths:
        path=path.resolve()
        image=Image.open(path).convert('L')
        width=max(16,round(image.width*64/image.height))
        pixels=np.array(image.resize((width,64),Image.Resampling.BILINEAR),dtype=np.float32)[None,None]/255-.5
        with torch.inference_mode():
            native=model(torch.from_numpy(pixels)).numpy()
        exported=session.run(None,{'image':pixels})[0]
        error=float(np.max(np.abs(native-exported)))
        assert error<.0005
        pytorch_text=decode(native[:,0].argmax(-1).tolist(),model.alphabet)
        onnx_text=decode(exported[:,0].argmax(-1).tolist(),model.alphabet)
        assert pytorch_text==onnx_text
        rust=json.loads(subprocess.check_output([str(args.binary),'recognize',str(path),'--line','--passes','original','--recognition-model',str(args.model),'--model-config',str(args.config)],text=True))
        assert rust['text'].strip()==pytorch_text.strip(),f"Rust/PyTorch mismatch: {rust['text']!r} vs {pytorch_text!r}"
        fixtures.append({'image':str(path.relative_to(ROOT)),'pytorch_text':pytorch_text,'onnx_text':onnx_text,'rust_text':rust['text'],'max_log_probability_error':error,'rust_confidence':rust['confidence'],'input_width':width})
    report={'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'dynamic_tensor_checks':checks,'held_out_gallery_checks':fixtures,'comparison':'ONNX vs PyTorch same tensors; actual Rust image pipeline transcripts equal after boundary-whitespace trim. Pillow/image triangle filters can differ by pixel rounding.'}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
