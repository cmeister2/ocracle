"""Native official LCNetV4/LightSVTR CTC branch, with strict source-weight loading.

The original auxiliary NRTR head is deliberately absent. This is a trainable
native instance of the deployed CTC branch, not a claimed reproduction of the
original full MultiHead training recipe. Eval-mode pooling and fixed BN preserve
the variable-width inference graph while ordinary autograd remains enabled.
"""
import hashlib
import json
from pathlib import Path
import sys
import paddle
from paddle import nn
from paddle.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def checked_sources():
    manifest = json.loads((HERE / "source-files.json").read_text())
    for record in manifest["files"]:
        data = (HERE / "vendor" / record["path"]).read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise RuntimeError("Official source hash differs: " + record["path"])
    sys.path.insert(0, str(HERE / "vendor"))


checked_sources()
from ppocr.modeling.backbones.rec_lcnetv4 import PPLCNetV4
from ppocr.modeling.necks.rnn import SequenceEncoder
from ppocr.modeling.heads.rec_ctc_head import CTCHead


def alphabet():
    dictionary = (HERE / "vendor/ppocr/utils/dict/ppocrv6_dict.txt").read_text().splitlines()
    if any(len(char) != 1 for char in dictionary) or len(set(dictionary)) != len(dictionary):
        raise RuntimeError("Official dictionary is not a unique Unicode scalar list")
    result = "".join(dictionary) + " "
    deployed = json.loads((ROOT / "models/paddle-v6/en.json").read_text())["alphabet"]
    if result != deployed:
        raise RuntimeError("Native training dictionary differs from deployed ONNX alphabet")
    return result


class Recognition(nn.Layer):
    def __init__(self, characters):
        super().__init__()
        self.characters = characters
        self.backbone = PPLCNetV4(model_size="small")
        self.ctc_encoder = SequenceEncoder(self.backbone.out_channels, "lightsvtr", dims=120,
                                          depth=2, mlp_ratio=2.0, local_kernel=7)
        self.ctc_head = CTCHead(120, len(characters) + 1, fc_decay=.00001)
        self.eval()

    def logits(self, image):
        if self.training:
            raise RuntimeError("Use eval-mode dynamic-width pooling/fixed BN during adaptation")
        return self.ctc_head.fc(self.ctc_encoder(self.backbone(image)))

    def forward(self, image):
        return F.softmax(self.logits(image), axis=-1)


def load_native():
    record = json.loads((HERE / "source-checkpoint.json").read_text())
    checkpoint = HERE / record["path"]
    if checkpoint.stat().st_size != record["bytes"] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError("Native source checkpoint hash differs")
    source = paddle.load(str(checkpoint))
    model = Recognition(alphabet())
    state = {}
    consumed = set()
    for name, target in model.state_dict().items():
        original = name if name.startswith("backbone.") else "head." + name
        if original not in source or list(source[original].shape) != list(target.shape):
            raise RuntimeError("Missing/mismatched native parameter: " + original)
        state[name] = source[original]
        consumed.add(original)
    unused = sorted(set(source) - consumed)
    # Full source includes the auxiliary NRTR branch, which inference omits.
    if any(not name.startswith(("head.gtc_head.", "head.before_gtc.")) for name in unused):
        raise RuntimeError("Unexpected unconsumed source parameters: " + str(unused[:12]))
    missing, unexpected = model.set_state_dict(state)
    if missing or unexpected:
        raise RuntimeError("Non-strict native checkpoint load")
    model.eval()
    return model, {"source_parameters": len(source), "loaded_parameters": len(state),
                   "unused_auxiliary_parameters": unused,
                   "source_sha256": record["sha256"]}
