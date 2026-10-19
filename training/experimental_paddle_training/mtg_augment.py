"""Seeded degradations applied before the recognizer's height48 resizing."""
import io
import math
import random
import numpy as np
from PIL import Image, ImageFilter

CONDITIONS = ['clean', 'gaussian', 'downsample', 'jpeg', 'motion', 'combined']


def degrade(image, condition, seed):
    if condition not in CONDITIONS:
        raise ValueError('Unknown MTG degradation')
    rng = random.Random(seed)
    image = image.convert('RGB')
    parameters = {'condition': condition, 'seed': seed}
    if condition in ('gaussian', 'combined'):
        sigma = rng.uniform(1.0, 2.8)
        image = image.filter(ImageFilter.GaussianBlur(sigma))
        parameters['sigma_source_pixels'] = sigma
    if condition in ('downsample', 'combined'):
        factor = rng.uniform(.35, .65)
        size = image.size
        image = image.resize((max(1, round(size[0]*factor)), max(1, round(size[1]*factor))), Image.Resampling.BILINEAR)
        image = image.resize(size, Image.Resampling.BILINEAR)
        parameters['downsample_factor'] = factor
    if condition == 'motion':
        length = rng.choice([5, 9, 13])
        angle = rng.uniform(-math.pi/3, math.pi/3)
        values = np.asarray(image, dtype=np.float32)
        radius = length//2+1
        padded = np.pad(values, ((radius, radius), (radius, radius), (0, 0)), mode='edge')
        result = np.zeros_like(values)
        for offset in range(-length//2+1, length//2+1):
            dx, dy = round(offset*math.cos(angle)), round(offset*math.sin(angle))
            result += padded[radius+dy:radius+dy+image.height, radius+dx:radius+dx+image.width]
        image = Image.fromarray(np.rint(result/length).clip(0, 255).astype(np.uint8))
        parameters.update(motion_length=length, motion_angle_radians=angle)
    if condition in ('jpeg', 'combined'):
        quality = rng.randint(15, 55)
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=quality, subsampling=2)
        stream.seek(0)
        image = Image.open(stream).convert('RGB')
        parameters['jpeg_quality'] = quality
    return image, parameters
