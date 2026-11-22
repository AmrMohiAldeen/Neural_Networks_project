"""Offline synthetic integration tests; these are not experimental results."""
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from eventfer.common import CLASSES, csv_write, digest, json_write, lock_directory
from eventfer import evaluate, experiment
from eventfer.export_raw import export_one
from eventfer.preprocess import build_counts, load_counts
from eventfer.protocol import MODES, check_evaluation_features, recipe


class ControlledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        manifest_dir = root / 'manifest'
        manifest_dir.mkdir()
        counts_dir = root / 'counts'
        counts_dir.mkdir()
        rows = []
        for split in ('train', 'validation', 'test', 'target_test'):
            for cls, label in enumerate(CLASSES):
                name = f'{split}_{cls}'
                dataset = 'nefer' if split == 'target_test' else 'ekman_rgb'
                path = counts_dir / f'{name}.npz'
                np.savez_compressed(path, counts=np.full((8, 2, 1, 1), cls + 1, dtype='uint32'),
                                    metadata=np.asarray(json.dumps({'start_us': 0, 'end_us': 1000000})))
                rows.append({'dataset': dataset, 'modality': 'events', 'clip_id': name,
                             'path': '/fixture/' + name, 'label': label, 'label_id': str(cls),
                             'participant_id': ['03', '07', '17', '19', '27', '28'][cls] if dataset == 'nefer' else '',
                             'original_video_id': name, 'split': split,
                             'sha256': hashlib.sha256(name.encode()).hexdigest(), 'expected_label': label if dataset == 'nefer' else '',
                             'counts_path': str(path), 'counts_sha256': digest(path)})
        csv_write(manifest_dir / 'manifest.csv', rows)
        lock_directory(manifest_dir, ['manifest.csv'])
        count_config = {'manifest_sha256': digest(manifest_dir / 'manifest.csv'), 'bins': 8, 'size': 224,
                        'crop_policy': 'full-frame', 'crops_sha256': None,
                        'implementation_sha256': digest(Path('eventfer/events.py')), 'splits': ['train', 'validation']}
        statistics = {'fitted_split': 'train', 'compression': 'log1p(count)/log(256); no per-frame rescaling or clipping',
                      'backbone_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]}}
        training = {}
        for mode in MODES:
            output = root / ('train_' + mode)
            output.mkdir()
            selected = []
            for source in rows:
                if source['split'] not in {'train', 'validation'}:
                    continue
                row = dict(source)
                features = np.zeros((8, 384), dtype='float32')
                features[:, int(row['label_id'])] = 3
                path = output / f'{row["clip_id"]}.npy'
                np.save(path, features)
                row.update(features_path=str(path), features_sha256=digest(path))
                selected.append(row)
            csv_write(output / 'index.csv', selected)
            config = {'mode': mode, 'statistics_sha256': 'fixture_stats', 'backbone': 'dinov2_vits14',
                      'backbone_commit': 'fixture', 'weights_sha256': 'fixture_weights',
                      'embedding_dim': 384, 'bins': 8, 'torch_version': str(torch.__version__),
                      'representation_sha256': digest(Path('eventfer/events.py')),
                      'count_config': count_config, 'statistics': statistics}
            json_write(output / 'config.json', config)
            lock_directory(output, ['index.csv', 'config.json'])
            training[mode] = output / 'index.csv'
        options = argparse.Namespace(output=root / 'experiment', seeds=[17, 29, 43],
                                     expected_train_counts=[6, 6], epochs=3, patience=3,
                                     batch_size=6, hidden_dim=8, dropout=0., lr=.03, weight_decay=0.,
                                     **{mode + '_index': path for mode, path in training.items()})
        return options, rows, manifest_dir / 'manifest.csv'

    def test_full_nine_run_freeze_evaluation_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            options, rows, manifest = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.create(options)
                experiment.train(argparse.Namespace(experiment=options.output, device='cpu'))
                # Completed-run resume verifies and skips rather than overwriting.
                experiment.train(argparse.Namespace(experiment=options.output, device='cpu'))
                experiment.freeze(argparse.Namespace(experiment=options.output, manifest=manifest,
                                                    evaluation_crops=None, expected_test_counts=[6, 6]))
            lock = options.output / 'frozen/experiment.json'
            frozen = json.loads(lock.read_text())
            for seed in options.seeds:
                self.assertEqual(len({frozen['models'][mode][str(seed)]['initial_state_sha256'] for mode in MODES}), 1)
            indexes = {}
            for mode in MODES:
                output = root / ('eval_' + mode)
                output.mkdir()
                selected = []
                for source in rows:
                    if source['split'] not in {'test', 'target_test'}:
                        continue
                    row = dict(source)
                    features = np.zeros((8, 384), dtype='float32')
                    features[:, int(row['label_id'])] = 3
                    path = output / f'{row["clip_id"]}.npy'
                    np.save(path, features)
                    row.update(features_path=str(path), features_sha256=digest(path))
                    selected.append(row)
                csv_write(output / 'index.csv', selected)
                config = json.loads((getattr(options, mode + '_index').parent / 'config.json').read_text())
                config['count_config']['splits'] = ['test', 'target_test']
                json_write(output / 'config.json', config)
                lock_directory(output, ['index.csv', 'config.json'])
                indexes[mode] = output / 'index.csv'
            argv = ['evaluate', '--evaluation-lock', str(lock), '--output', str(root / 'results'), '--device', 'cpu']
            for mode, index in indexes.items():
                argv += ['--' + mode.replace('_', '-') + '-index', str(index)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            summary = json.loads((root / 'results/summary.json').read_text())
            self.assertAlmostEqual(summary['primary_target_gain_pp'], 0.)
            self.assertAlmostEqual(summary['primary_source_loss_pp'], 0.)
            self.assertFalse(summary['hypothesis']['point_estimate_criterion_met'])
            with (root / 'results/predictions.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 9 * 12)
            with (root / 'results/participant_omission.csv').open() as f:
                omissions = list(csv.DictReader(f))
                self.assertEqual(len(omissions), 18)
                self.assertTrue(all(row['missing_classes'] for row in omissions))
            self.assertTrue((root / 'results/figures/event_density.png').is_file())
            self.assertEqual(len(json.loads((root / 'results/seed_metrics.json').read_text())), 18)
            # Even a freshly checksummed evaluation cache cannot omit a test clip.
            missing_dir = root / 'missing_clip_cache'
            missing_dir.mkdir()
            with indexes['baseline'].open(newline='') as f:
                incomplete = list(csv.DictReader(f))[1:]
            csv_write(missing_dir / 'index.csv', incomplete)
            json_write(missing_dir / 'config.json', json.loads((indexes['baseline'].parent / 'config.json').read_text()))
            lock_directory(missing_dir, ['index.csv', 'config.json'])
            bad_argv = list(argv)
            bad_argv[bad_argv.index('--baseline-index') + 1] = str(missing_dir / 'index.csv')
            bad_argv[bad_argv.index('--output') + 1] = str(root / 'missing_clip_results')
            with patch.object(sys, 'argv', bad_argv), self.assertRaisesRegex(ValueError, 'omitted'):
                evaluate.main()
            self.assertFalse((root / 'missing_clip_results').exists())
            checkpoint = Path(frozen['models']['baseline']['17']['checkpoint'])
            checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
            argv[argv.index('--output') + 1] = str(root / 'bad_results')
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'changed'):
                evaluate.main()
            self.assertFalse((root / 'bad_results').exists())
            with self.assertRaisesRegex(ValueError, 'training is closed'):
                experiment.train(argparse.Namespace(experiment=options.output, device='cpu'))

    def test_condition_label_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            options, rows, manifest = self.fixture(Path(temp))
            index = options.per_bin_index
            with index.open(newline='') as f:
                selected = list(csv.DictReader(f))
            selected[0]['label'], selected[0]['label_id'] = 'fear', '2'
            csv_write(index, selected)
            lock_directory(index.parent, ['index.csv', 'config.json'])
            with self.assertRaisesRegex(ValueError, 'Conditions differ'):
                experiment.create(options)
            self.assertFalse(options.output.exists())

    def test_sdk_export_preserves_integer_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / 'clip.raw'
            raw.write_bytes(b'fixture raw')
            row = {'dataset': 'nefer', 'clip_id': 'clip', 'path': str(raw), 'sha256': digest(raw)}
            data = np.array([(0, 0, 0, 5000000000), (3, 2, 1, 5000000100)],
                            dtype=[('x', 'u2'), ('y', 'u2'), ('p', 'u1'), ('t', 'i8')])
            class Reader:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                def get_size(self):
                    return 3, 4
                def __iter__(self):
                    yield data
            info = export_one(row, root, Reader)
            self.assertEqual(info['events'], 2)
            self.assertEqual(info['start_us'], 5000000000)
            path = next(root.glob('*.h5'))
            with h5py.File(path) as f:
                np.testing.assert_array_equal(f['events'][:], [[5000000000, 0, 0, 0], [5000000100, 3, 2, 1]])
            # Conda-side counting uses the verified export without importing the SDK.
            row.update(modality='events', label='anger', label_id='0', split='target_test',
                       participant_id='03', original_video_id='clip', expected_label='')
            manifest_dir = root / 'frozen'
            manifest_dir.mkdir()
            csv_write(manifest_dir / 'manifest.csv', [row])
            lock_directory(manifest_dir, ['manifest.csv'])
            args = argparse.Namespace(manifest=manifest_dir / 'manifest.csv', output=root / 'counts',
                                      splits=['target_test'], crop_policy='full-frame', crops=None,
                                      bins=8, size=224, decoded_raw_root=root)
            with contextlib.redirect_stdout(io.StringIO()):
                build_counts(args)
            with (root / 'counts/index.csv').open(newline='') as f:
                counted = next(csv.DictReader(f))
            self.assertEqual(int(load_counts(counted).sum()), 2)
            self.assertEqual(counted['path'], row['path'])

    def test_hypothesis_uses_percentage_points_and_both_margins(self):
        frozen = {'seeds': [17, 29, 43], 'expected_target_participants': ['03', '07', '17', '19', '27', '28'],
                  'density': {'cutoffs': [2.5, 4.5]},
                  'hypothesis': {'target_gain_percentage_points_at_least': 3., 'source_loss_percentage_points_at_most': 2.}}
        def run(source_errors, target_errors):
            rows = []
            for mode in MODES:
                for seed in frozen['seeds']:
                    for split in ('test', 'target_test'):
                        wrong = (source_errors if split == 'test' else target_errors) if mode == 'per_bin' else 2
                        for cls, label in enumerate(CLASSES):
                            for sample in range(50):
                                rows.append({'mode': mode, 'seed': seed, 'split': split,
                                             'label': label, 'label_id': cls,
                                             'prediction_id': (cls + 1) % 6 if sample < wrong else cls,
                                             'participant_id': frozen['expected_target_participants'][cls],
                                             'event_rate': cls + 1, 'expected_label': ''})
            return evaluate.analyze(rows, frozen)[0]
        passed = run(3, 0)
        self.assertAlmostEqual(passed['primary_target_gain_pp'], 4.)
        self.assertAlmostEqual(passed['primary_source_loss_pp'], 2.)
        self.assertTrue(passed['hypothesis']['point_estimate_criterion_met'])
        self.assertFalse(run(4, 0)['hypothesis']['point_estimate_criterion_met'])
        self.assertFalse(run(3, 1)['hypothesis']['point_estimate_criterion_met'])


if __name__ == '__main__':
    unittest.main()
