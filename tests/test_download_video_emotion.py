"""Offline tests of interrupted-download and integrity handling."""
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'downloader', Path(__file__).resolve().parents[1] / 'scripts' / 'download_video_emotion.py'
)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name)
        self.body = b'example dataset bytes'
        self.item = {
            'key': 'sample.bin', 'size': len(self.body),
            'checksum': 'md5:' + hashlib.md5(self.body).hexdigest(),
            'links': {'self': 'https://example.invalid/sample.bin'},
        }
        self.partial = self.destination / 'sample.bin.part'
        self.final = self.destination / 'sample.bin'

    def test_resume_and_verify(self):
        self.partial.write_bytes(self.body[:7])
        response = Response(self.body[7:], 206, {'Content-Range': f'bytes 7-{len(self.body)-1}/{len(self.body)}'})
        with patch.object(downloader.urllib.request, 'urlopen', return_value=response) as request:
            downloader.download_file(self.item, self.destination, attempts=1)
        self.assertEqual(request.call_args.args[0].get_header('Range'), 'bytes=7-')
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertFalse(self.partial.exists())

    def test_server_ignores_range_restarts_without_appending(self):
        self.partial.write_bytes(self.body[:7])
        with patch.object(downloader.urllib.request, 'urlopen', return_value=Response(self.body)):
            downloader.download_file(self.item, self.destination, attempts=1)
        self.assertEqual(self.final.read_bytes(), self.body)

    def test_corrupt_payload_is_not_marked_complete(self):
        with patch.object(downloader.urllib.request, 'urlopen', return_value=Response(b'x' * len(self.body))):
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                downloader.download_file(self.item, self.destination, attempts=1)
        self.assertFalse(self.final.exists())

    def test_verified_file_skips_network(self):
        self.final.write_bytes(self.body)
        with patch.object(downloader.urllib.request, 'urlopen') as request:
            downloader.download_file(self.item, self.destination, attempts=1)
        request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
