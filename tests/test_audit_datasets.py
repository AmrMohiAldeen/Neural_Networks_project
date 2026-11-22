import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'audit', Path(__file__).resolve().parents[1] / 'scripts' / 'audit_datasets.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_joy_mapping(self):
        self.assertEqual(audit.label(' Joy '), 'happiness')

    def rows(self):
        return [{'clip_id': str(i), 'original_video_id': str(i), 'participant_id': '',
                 'sha256': str(i), 'split': ''} for i in range(30)]

    def test_grouped_splits_reproducible_and_connected(self):
        rows = self.rows()
        rows[0]['original_video_id'] = rows[1]['original_video_id']
        rows[1]['participant_id'] = rows[2]['participant_id'] = 'known_person'
        rows[2]['sha256'] = rows[3]['sha256']
        again = [dict(row) for row in rows]
        audit.split_source(rows, 476)
        audit.split_source(again, 476)
        self.assertEqual(rows, again)
        self.assertEqual(len({r['split'] for r in rows[:4]}), 1)
        self.assertEqual({r['split'] for r in rows}, {'train', 'validation', 'test'})

    def test_official_identity_leak_rejected(self):
        rows = self.rows()[:3]
        for row, split in zip(rows, ['train', 'validation', 'test']):
            row['split'] = split
        rows[0]['participant_id'] = rows[1]['participant_id'] = 'same'
        with self.assertRaisesRegex(ValueError, 'leak'):
            audit.split_source(rows, 476)

    def test_partial_official_splits_rejected(self):
        rows = self.rows()
        rows[0]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'every clip'):
            audit.split_source(rows, 476)

    def test_nefer_frame_rows_collapse_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.csv'
            path.write_text('user_03,clip,frame1,Happiness\nuser_03,clip,frame2,Happiness\n')
            self.assertEqual(audit.annotation_clips(path), {('03', 'clip'): 'happiness'})
            path.write_text(path.read_text() + 'user_03,clip,frame3,Fear\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                audit.annotation_clips(path)

    def test_ambiguous_file_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for folder in ['a', 'b']:
                (root / folder).mkdir()
                (root / folder / 'clip.mp4').write_bytes(b'video')
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                audit.resolve_file(root, 'missing/clip.mp4', audit.file_index(root, audit.VIDEO))

    def test_full_metadata_build_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, target, ann, out = [root / x for x in ['source', 'target', 'annotations', 'output']]
            src.mkdir()
            target.mkdir()
            (ann / 'reported_emotions').mkdir(parents=True)
            source_csv = root / 'source.csv'
            with source_csv.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'label', 'original_video_id', 'split'])
                writer.writeheader()
                for split in ['train', 'validation', 'test']:
                    for emotion in audit.CLASSES:
                        name = f'{split}_{emotion}.mp4'
                        (src / name).write_bytes(name.encode())
                        writer.writerow({'path': name, 'label': emotion,
                                         'original_video_id': name, 'split': split})
            for part, users in [('train', audit.TRAIN_USERS), ('validation', audit.TARGET_USERS)]:
                with (ann / 'reported_emotions' / f'{part}_event_subjective.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    for user in sorted(users):
                        for emotion in audit.CLASSES + ['None', 'Contempt']:
                            clip = f'user{user}_{emotion}_cd'
                            writer.writerow([f'user_{user}', clip, 'frame0.png', emotion])
                            writer.writerow([f'user_{user}', clip, 'frame1.png', emotion])
                            if emotion in audit.CLASSES:
                                (target / f'{clip}.raw').write_bytes(clip.encode())
            args = argparse.Namespace(source_root=src, source_kind='ekman_rgb', source_csv=source_csv,
                                      ekman_pickle=None, trust_pickle=False, nefer_root=target,
                                      annotations_root=ann, seed=476, output=out)
            audit.build(args)
            report = json.loads((out / 'audit.json').read_text())
            self.assertEqual(report['clip_counts']['target_test'], 36)
            self.assertEqual(report['clip_counts']['target_reserve'], 120)
            self.assertEqual(report['clip_counts']['train'], 6)
            self.assertEqual(report['class_to_idx']['happiness'], 3)
            with (out / 'exclusions.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 52)
            checksums = json.loads((out / 'checksums.json').read_text())
            for name, expected in checksums.items():
                self.assertEqual(audit.sha256(out / name), expected)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                audit.build(args)


if __name__ == '__main__':
    unittest.main()
