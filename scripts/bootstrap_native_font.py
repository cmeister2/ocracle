#!/usr/bin/env python3
"""Fetch the exact Ubuntu font used by the frozen native rendering proof.

Ubuntu and Debian builds bearing the same upstream version contain different
TTF bytes. Verify both the complete official package and the two selected
members. This extracts data only; it never installs a Debian package or runs
its maintainer scripts. CI copies the verified font to the historical path.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

PACKAGE_URL = 'https://archive.ubuntu.com/ubuntu/pool/main/f/fonts-dejavu/fonts-dejavu-extra_2.37-8_all.deb'
PACKAGE_BYTES = 1947322
PACKAGE_SHA256 = '5502e5ccc416683e5251dc105b37be31158bc2e02133221e56524ea2a8e426fe'
FONT_SHA256 = '1d7476da447f9455a47c41adfd6ac2b1fc1b65e660f3ea4bf36128506c7e094c'
MEMBERS = {
    './usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf':
        ('DejaVuSerif-Italic.ttf', 346544, FONT_SHA256),
    './usr/share/doc/fonts-dejavu-extra/copyright':
        ('copyright', 3859, '63d3ba759d12804c5b31a9d5940d855c1820d1f5999e6b0872eb1c7ff045fbc9'),
}


def verified(data, size, expected, label):
    if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Pinned bytes differ: '+label)
    return data


def fetch_package(path):
    if path.exists():
        verified(path.read_bytes(), PACKAGE_BYTES, PACKAGE_SHA256, 'cached Ubuntu font package')
        return
    request = urllib.request.Request(PACKAGE_URL, headers={'User-Agent': 'ocracle pinned native font bootstrap'})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(PACKAGE_BYTES+1)
    verified(data, PACKAGE_BYTES, PACKAGE_SHA256, 'downloaded Ubuntu font package')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def selected_members(data, members=MEMBERS):
    """Read explicit regular-file members; never extract archive paths to disk."""
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        for name, (output, size, expected) in members.items():
            matches = [member for member in archive.getmembers() if member.name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size != size:
                raise ValueError('Expected one pinned regular-file member: '+name)
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise ValueError('Missing pinned member data: '+name)
            with stream:
                result[output] = verified(stream.read(size+1), size, expected, name)
    return result


def bootstrap(destination):
    destination.mkdir(parents=True, exist_ok=True)
    package = destination/'fonts-dejavu-extra_2.37-8_all.deb'
    fetch_package(package)
    # dpkg-deb handles the package's zstd compression on Ubuntu; unlike dpkg -i,
    # --fsys-tarfile only decodes data and never executes package scripts.
    unpacked = subprocess.run(['dpkg-deb', '--fsys-tarfile', str(package)], check=True, capture_output=True)
    for name, data in selected_members(unpacked.stdout).items():
        temporary = destination/(name+'.part')
        try:
            temporary.write_bytes(data)
            temporary.replace(destination/name)
        finally:
            temporary.unlink(missing_ok=True)
    return {'package_url': PACKAGE_URL, 'package_sha256': PACKAGE_SHA256,
            'font': str(destination/'DejaVuSerif-Italic.ttf'), 'font_sha256': FONT_SHA256,
            'license': str(destination/'copyright')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.destination), indent=2))
