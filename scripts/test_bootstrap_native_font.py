"""Fail closed on package/member drift without requiring network access."""
import ast
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import bootstrap_native_font as fonts


class PinnedNativeFont(unittest.TestCase):
    def test_font_pin_still_matches_frozen_renderer(self):
        source = Path(__file__).resolve().parents[1]/'training/experimental_paddle_training/inputs.py'
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(source.read_text()).body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == 'FONT_SHA256'}
        self.assertEqual(fonts.FONT_SHA256, values['FONT_SHA256'])

    def test_corrupt_cached_package_rejected_without_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'font.deb'
            path.write_bytes(b'different package')
            with patch.object(fonts.urllib.request, 'urlopen') as network:
                with self.assertRaisesRegex(ValueError, 'Pinned bytes differ'):
                    fonts.fetch_package(path)
                network.assert_not_called()

    def test_wrong_download_digest_rejected_before_caching(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'font.deb'
            with patch.object(fonts, 'PACKAGE_BYTES', 3), patch.object(fonts.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad')):
                with self.assertRaisesRegex(ValueError, 'Pinned bytes differ'):
                    fonts.fetch_package(path)
            self.assertFalse(path.exists())

    def test_member_bytes_and_regular_file_type_required(self):
        def archive(data, symbolic=False):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as output:
                member = tarfile.TarInfo('./font.ttf')
                if symbolic:
                    member.type = tarfile.SYMTYPE
                    member.linkname = '/different/font.ttf'
                else:
                    member.size = len(data)
                output.addfile(member, None if symbolic else io.BytesIO(data))
            return stream.getvalue()

        members = {'./font.ttf': ('font.ttf', 4, hashlib.sha256(b'font').hexdigest())}
        self.assertEqual(fonts.selected_members(archive(b'font'), members), {'font.ttf': b'font'})
        for data in [archive(b'FAKE'), archive(b'font', symbolic=True)]:
            with self.assertRaises(ValueError):
                fonts.selected_members(data, members)


if __name__ == '__main__':
    unittest.main()
