import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eventfer.common import CLASSES, csv_write, digest, lock_directory, read_rows, verify_lock
from eventfer.subset import allocate, make_subset, verify

ROOT = Path(__file__).resolve().parents[1]


class AllocationTests(unittest.TestCase):
    def test_proportional_with_minimum_and_exact_total(self):
        alloc = allocate({'train': 1140, 'validation': 159, 'test': 333}, 300, minimum=6)
        self.assertEqual(sum(alloc.values()), 300)
        self.assertEqual(alloc, {'train': 210, 'validation': 29, 'test': 61})
        small = allocate({'a': 100, 'b': 1, 'c': 2}, 6, minimum=1)
        self.assertEqual(sum(small.values()), 6)
        self.assertTrue(all(v >= 1 for v in small.values()))
        self.assertLessEqual(small['b'], 1)

    def test_rejects_impossible_requests(self):
        with self.assertRaises(ValueError):
            allocate({'a': 2}, 3)
        with self.assertRaises(ValueError):
            allocate({'a': 5, 'b': 5}, 1, minimum=1)


class SubsetTests(unittest.TestCase):
    def fixture(self, root):
        rgb, frozen = root / 'rgb', root / 'frozen'
        frozen.mkdir()
        rows = []
        for split, per_class in (('train', 12), ('validation', 3), ('test', 4)):
            for c, label in enumerate(CLASSES):
                for i in range(per_class):
                    video = rgb / label / f'{split}_{i}.mp4'
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.write_bytes(f'{split}{label}{i}'.encode())
                    rows.append({'dataset': 'ekman_rgb', 'modality': 'rgb', 'clip_id': f'{label}/{video.name}',
                                 'path': str(video), 'label': label, 'label_id': str(c), 'split': split,
                                 'participant_id': '', 'sha256': digest(video)})
        target = root / 'user_03.raw'
        target.write_bytes(b'target')
        rows.append({'dataset': 'nefer', 'modality': 'events', 'clip_id': 'u03', 'path': str(target),
                     'label': 'fear', 'label_id': '2', 'split': 'target_test', 'participant_id': '03',
                     'sha256': digest(target)})
        csv_write(frozen / 'manifest.csv', rows)
        lock_directory(frozen, ['manifest.csv'])
        return argparse.Namespace(manifest=frozen / 'manifest.csv', rgb_root=rgb, size=60, seed=476,
                                  source_dataset='ekman_rgb', prefer_converted_root=None,
                                  output=root / 'subset')

    def run_quiet(self, fn, args):
        with contextlib.redirect_stdout(io.StringIO()):
            fn(args)

    def test_stratified_deterministic_and_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            self.run_quiet(make_subset, args)
            verify_lock(args.output / 'manifest.csv')
            rows = read_rows(args.output / 'manifest.csv')
            parent = {(r['dataset'], r['clip_id']): r for r in read_rows(args.manifest)}
            source = [r for r in rows if r['dataset'] == 'ekman_rgb']
            self.assertEqual(len(source), 60)
            self.assertEqual([r for r in rows if r['dataset'] == 'nefer'][0]['clip_id'], 'u03')
            for r in rows:
                self.assertEqual(r, parent[(r['dataset'], r['clip_id'])])  # rows unchanged, splits kept
            for split in ('train', 'validation', 'test'):
                self.assertEqual({r['label'] for r in source if r['split'] == split}, set(CLASSES))
            paths = (args.output / 'source_relative_paths.txt').read_text().split()
            self.assertEqual(len(paths), 60)
            again = argparse.Namespace(**{**vars(args), 'output': Path(tmp) / 'again'})
            self.run_quiet(make_subset, again)
            self.assertEqual(digest(again.output / 'manifest.csv'),
                             digest(args.output / 'manifest.csv'))
            self.run_quiet(verify, argparse.Namespace(directory=args.output))

    def test_prefers_completed_conversions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            events = Path(tmp) / 'events'
            done = events / 'anger/train_11'
            done.mkdir(parents=True)
            (done / 'events.h5').write_bytes(b'12345')
            (done / '.conversion-success.json').write_text(json.dumps({'h5_size': 5}))
            partial = events / 'anger/train_10'
            partial.mkdir(parents=True)
            (partial / 'events.h5').write_bytes(b'partial')  # no completion record
            args.prefer_converted_root = events
            self.run_quiet(make_subset, args)
            chosen = {r['clip_id'] for r in read_rows(args.output / 'manifest.csv')}
            self.assertIn('anger/train_11.mp4', chosen)
            info = json.loads((args.output / 'subset.json').read_text())
            self.assertEqual(info['selected_already_converted'], 1)

    def test_existing_output_and_changed_parent_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            self.run_quiet(make_subset, args)
            with self.assertRaises(FileExistsError):
                make_subset(args)
            with args.manifest.open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify(argparse.Namespace(directory=args.output))


class IncludeListTests(unittest.TestCase):
    def test_launcher_converts_only_listed_videos(self):
        fake = "import argparse\nfrom pathlib import Path\np=argparse.ArgumentParser()\np.add_argument('-i')\n" \
               "p.add_argument('-o')\na,_=p.parse_known_args()\n(Path(a.o)/'events.h5').write_bytes(b'x')\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'in/anger').mkdir(parents=True)
            (root / 'v2e').mkdir()
            (root / 'v2e/v2e.py').write_text(fake)
            for name in ('a', 'b', 'c'):
                (root / f'in/anger/{name}.mp4').write_text('ok')
            (root / 'list.txt').write_text('anger/a.mp4\nanger/c.mp4\n')
            command = [sys.executable, str(ROOT / 'scripts/convert_events_parallel.py'),
                       '--input-root', str(root / 'in'), '--output-root', str(root / 'out'),
                       '--v2e-root', str(root / 'v2e'), '--include-list', str(root / 'list.txt')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("'success': 2", result.stdout)
            self.assertTrue((root / 'out/anger/c/events.h5').is_file())
            self.assertFalse((root / 'out/anger/b').exists())
            (root / 'list.txt').write_text('anger/missing.mp4\n')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
