"""Small, fresh procedural inputs; no held-out card transcriptions are used."""
import hashlib
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf')
FONT_SHA256 = '1d7476da447f9455a47c41adfd6ac2b1fc1b65e660f3ea4bf36128506c7e094c'
FONT_LICENSE = 'DejaVu fonts license (Bitstream Vera-derived)'


def render(text, degraded=False):
    if hashlib.sha256(FONT.read_bytes()).hexdigest() != FONT_SHA256:
        raise RuntimeError('Install the pinned fonts-dejavu-extra font before this smoke test')
    font = ImageFont.truetype(str(FONT), 28)
    left, top, right, bottom = font.getbbox(text)
    image = Image.new('RGB', (right-left+16, bottom-top+14), (236, 229, 211))
    ImageDraw.Draw(image).text((8-left, 7-top), text, font=font, fill=(33, 39, 45))
    if degraded:
        image = image.filter(ImageFilter.GaussianBlur(.8))
        size = image.size
        image = image.resize((max(1, round(size[0]*.7)), max(1, round(size[1]*.7))), Image.Resampling.BILINEAR)
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def tensor(image):
    """Official BGR/affine/pad contract; PIL bilinear is used for this proof."""
    image = image.convert('RGB')
    width = math.ceil(48*image.width/image.height)
    result = np.zeros((1, 3, 48, max(320, width)), dtype=np.float32)
    pixels = np.asarray(image.resize((width, 48), Image.Resampling.BILINEAR))
    result[0, :, :, :width] = pixels[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)*(2/255)-1
    return result


def fingerprint(array):
    return hashlib.sha256(array.tobytes()).hexdigest()
