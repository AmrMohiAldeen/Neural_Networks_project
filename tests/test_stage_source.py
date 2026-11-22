import argparse
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from eventfer.common import csv_write, digest, lock_directory
from eventfer.stage_source import stage


class StageSourceTests(unittest.TestCase):
    def fixture(self, root):
        raw, frozen = root / 'rgb', root / 'frozen'
        (raw / 'anger').mkdir(parents=True)
        frozen.mkdir()
        included, excluded = raw / 'anger/retained.mp4', raw / 'anger/excluded.mp4'
        included.write_bytes(b'retained source')
        excluded.write_bytes(b'excluded source')
        rows = [{'dataset': 'ekman_rgb', 'modality': 'rgb', 'clip_id': 'anger/retained.mp4',
                 'path': str(included), 'label': 'anger', 'label_id': '0',
                 'split': 'train', 'sha256': digest(included)}]
        target = root / 'heldout.raw'
        target.write_bytes(b'heldout events')
        rows.append({'dataset': 'nefer', 'modality': 'events', 'clip_id': 'heldout',
                     'path': str(target), 'label': 'anger', 'label_id': '0',
                     'split': 'target_test', 'sha256': digest(target)})
        csv_write(frozen / 'manifest.csv', rows)
        lock_directory(frozen, ['manifest.csv'])
        args = argparse.Namespace(manifest=frozen / 'manifest.csv', rgb_root=raw,
                                  output=root / 'staged', source_dataset='ekman_rgb',
                                  expected_count=1, link_mode='hardlink')
        return args, included, excluded

    def test_stages_only_retained_paths_without_moving_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, included, excluded = self.fixture(Path(tmp))
            # Exercise selection/path logic without requiring Windows hard-link privileges.
            with contextlib.redirect_stdout(io.StringIO()), patch(
                'eventfer.stage_source.os.link', side_effect=shutil.copyfile
            ) as link:
                stage(args)
            link.assert_called_once_with(included.resolve(), args.output / 'anger/retained.mp4')
            self.assertEqual((args.output / 'anger/retained.mp4').read_bytes(), included.read_bytes())
            self.assertFalse((args.output / 'anger/excluded.mp4').exists())
            self.assertFalse((args.output / 'heldout.raw').exists())
            self.assertTrue(included.is_file())
            self.assertTrue(excluded.is_file())
            self.assertTrue((args.output / 'staging.json').is_file())

    def test_changed_source_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, included, _ = self.fixture(Path(tmp))
            included.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                stage(args)
            self.assertFalse(args.output.exists())

    def test_incorrect_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _ = self.fixture(Path(tmp))
            args.expected_count = 1632
            with self.assertRaisesRegex(ValueError, 'Expected 1632'):
                stage(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
