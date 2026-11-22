import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('rgb', Path(__file__).resolve().parents[1] / 'scripts/download_ekman_rgb.py')
rgb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rgb)


class RGBDownloadTests(unittest.TestCase):
    def archive(self, method):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w', compression=method) as z:
            z.writestr('videos/anger/example.mp4', b'fake video payload' * 100)
        return data.getvalue()

    def check_method(self, method):
        body = self.archive(method)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(rgb, 'SIZE', len(body)), patch.object(
                rgb, 'byte_range', side_effect=lambda url, start, end: io.BytesIO(body[start:end + 1])):
                with zipfile.ZipFile(rgb.RemoteArchive('test')) as z:
                    info = z.infolist()[0]
                    end = z.start_dir - 1
                self.assertEqual(rgb.download_member('test', output, info, end)[0], 'downloaded')
                self.assertEqual((output / 'anger/example.mp4').read_bytes(), b'fake video payload' * 100)
                self.assertEqual(rgb.download_member('test', output, info, end)[0], 'verified existing')
                (output / 'anger/example.mp4').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'Existing file'):
                    rgb.download_member('test', output, info, end)

    def test_deflated_member_and_resume(self):
        self.check_method(zipfile.ZIP_DEFLATED)

    def test_stored_member_and_resume(self):
        self.check_method(zipfile.ZIP_STORED)

    def test_archive_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rgb.target_path(Path(tmp), zipfile.ZipInfo('videos/anger/../../escape.mp4'))


if __name__ == '__main__':
    unittest.main()
