import argparse
from collections import Counter
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'cleaner', Path(__file__).resolve().parents[1] / 'scripts/clean_frozen_splits.py')
cleaner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleaner)


class FrozenCleanupTests(unittest.TestCase):
    def fixture(self, root):
        original = root / 'v1'
        original.mkdir()
        data = root / 'recordings'
        data.mkdir()
        rows = []

        def add(name, label, split, payload=None, dataset='ekman_rgb'):
            path = data / name
            path.write_bytes((payload or name).encode())
            rows.append({'dataset': dataset, 'modality': 'rgb' if dataset == 'ekman_rgb' else 'events',
                         'clip_id': name, 'path': str(path), 'label': label,
                         'label_id': str(cleaner.CLASSES.index(label)), 'participant_id': '',
                         'original_video_id': name, 'split': split,
                         'sha256': cleaner.sha256(path), 'expected_label': ''})
        for split in ('train', 'validation', 'test'):
            for label in cleaner.CLASSES:
                add(f'{split}_{label}.mp4', label, split)
        for split in ('train', 'test'):
            add(f'{split}_conflict_disgust.mp4', 'disgust', split, f'{split}_conflict')
            add(f'{split}_conflict_fear.mp4', 'fear', split, f'{split}_conflict')
        add('fear_video.mp4', 'fear', 'test', 'identical_fear')
        add('fear_video-1.mp4', 'fear', 'test', 'identical_fear')
        add('reserve.raw', 'anger', 'target_reserve', dataset='nefer')
        add('heldout.raw', 'happiness', 'target_test', dataset='nefer')
        cleaner.write_csv(original / 'manifest.csv', rows, list(rows[0]))
        cleaner.write_csv(original / 'exclusions.csv', [{'dataset': 'nefer', 'clip_id': 'missing', 'reason': 'missing raw recording'}], ['dataset', 'clip_id', 'reason'])
        classes, participants = cleaner.count_tables(rows)
        cleaner.write_csv(original / 'class_counts.csv', classes, list(classes[0]))
        cleaner.write_csv(original / 'participant_counts.csv', participants, list(participants[0]))
        cleaner.write_json(original / 'audit.json', {'source_protocol': 'fixture groups',
                           'status': 'metadata_splits_frozen_event_integrity_pending',
                           'clip_counts': dict(Counter(r['split'] for r in rows)), 'seed': 476})
        cleaner.write_json(original / 'checksums.json', {p.name: cleaner.sha256(p) for p in original.iterdir()})
        return original, rows

    def test_cleanup_preserves_splits_nefer_originals_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, old_rows = self.fixture(root)
            original_hashes = {p.name: cleaner.sha256(p) for p in original.iterdir()}
            args = argparse.Namespace(input=original, output=root / 'v2', source_dataset='ekman_rgb',
                                      expect_source_counts=[6, 6, 7], verify_recordings=True)
            with contextlib.redirect_stdout(io.StringIO()):
                cleaner.clean(args)
            rows, _ = cleaner.read_csv(root / 'v2/manifest.csv')
            self.assertEqual(len(rows), len(old_rows) - 5)
            self.assertTrue(all(row in old_rows for row in rows))
            self.assertEqual([r for r in rows if r['dataset'] == 'nefer'], [r for r in old_rows if r['dataset'] == 'nefer'])
            self.assertIn('fear_video.mp4', [r['clip_id'] for r in rows])
            self.assertNotIn('fear_video-1.mp4', [r['clip_id'] for r in rows])
            self.assertTrue(all('conflict' not in r['clip_id'] for r in rows))
            self.assertEqual(original_hashes, {p.name: cleaner.sha256(p) for p in original.iterdir()})
            self.assertTrue(all(Path(r['path']).is_file() for r in old_rows))
            exclusions, _ = cleaner.read_csv(root / 'v2/exclusions.csv')
            self.assertEqual(len(exclusions), 6)
            self.assertEqual(cleaner.verify(root / 'v2')['source_duplicates_remaining'], [])
            with self.assertRaisesRegex(ValueError, 'new output'):
                cleaner.clean(args)

    def test_wrong_expected_counts_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, _ = self.fixture(root)
            args = argparse.Namespace(input=original, output=root / 'v2', source_dataset='ekman_rgb',
                                      expect_source_counts=[999, 6, 7], verify_recordings=False)
            with self.assertRaisesRegex(ValueError, 'Unexpected source counts'):
                cleaner.clean(args)
            self.assertFalse(args.output.exists())

    def test_tampered_original_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, _ = self.fixture(Path(tmp))
            (original / 'audit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                cleaner.verify(original)


if __name__ == '__main__':
    unittest.main()
