"""Scientific correctness, leakage controls, and baseline learning smoke checks."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from eventfer.common import csv_write, digest, json_write, lock_directory, verify_lock
from eventfer.events import accumulate, count_clip, represent, scan
from eventfer.features import encode
from eventfer.preprocess import fit_reference
from eventfer.train import TemporalHead, metrics, seed_everything, train_one


class PreprocessingTests(unittest.TestCase):
    def test_bins_endpoint_polarity_and_fixed_crop(self):
        data = np.array([[0, 0, 0, 1], [1, 1, 1, 0], [2, 2, 2, 1],
                         [8, 3, 3, 0], [8, 4, 4, 1]], dtype=np.int64)
        counts = np.zeros((8, 2, 4, 4), dtype=np.uint64)
        accumulate(data, counts, 0, 8, (0, 0, 4, 4), 'binary')
        self.assertEqual(counts.sum(), 4)
        self.assertEqual(counts[0, 0, 0, 0], 1)
        self.assertEqual(counts[1, 1, 1, 1], 1)
        self.assertEqual(counts[2, 0, 2, 2], 1)
        self.assertEqual(counts[7, 1, 3, 3], 1)

    def test_scaling_formula_and_sum_channel_before_log(self):
        counts = np.zeros((3, 2, 1, 1), dtype=np.uint32)
        counts[:, 0, 0, 0] = [0, 10, 100]
        counts[:, 1, 0, 0] = [0, 10, 100]
        frames, scales = represent(counts, 'per_bin', 10)
        np.testing.assert_allclose(scales, [1, .5, .25])
        self.assertAlmostEqual(float(frames[1, 2, 0, 0]), np.log1p(10) / np.log(256), places=6)
        np.testing.assert_allclose(represent(counts, 'baseline')[1], [1, 1, 1])
        clip_scales = represent(counts, 'whole_clip', 10)[1]
        self.assertEqual(len(set(clip_scales)), 1)
        np.testing.assert_array_equal(counts[:, 0, 0, 0], [0, 10, 100])

    def test_h5_stream_scan_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.h5'
            with h5py.File(path, 'w') as f:
                f['events'] = np.array([[100, 0, 0, 1], [200, 3, 3, 0]], dtype='uint32')
            row = {'path': str(path), 'width': '4', 'height': '4'}
            counts, metadata = count_clip(row, None, bins=8, size=4)
            self.assertEqual(int(counts.sum()), 2)
            self.assertEqual(metadata['raw_events'], 2)
            self.assertEqual(counts[0, 0, 0, 0], 1)
            self.assertEqual(counts[7, 1, 3, 3], 1)
            with patch('eventfer.events.chunks', return_value=iter([
                np.array([[10, 0, 0, 1]], dtype='uint32'),
                np.array([[9, 0, 0, 1]], dtype='uint32'),
            ])):
                with self.assertRaisesRegex(ValueError, 'Timestamps'):
                    scan(path, 4, 4)

    def test_reference_ignores_validation_and_target(self):
        from eventfer.common import CLASSES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i, split in enumerate(['train', 'validation', 'target_test']):
                counts = np.zeros((3, 2, 2, 2), dtype=np.uint32)
                counts[0, 0, 0, 0], counts[1, 0, 0, 0] = (2, 4) if i == 0 else (100000, 100000)
                path = root / f'{split}.npz'
                np.savez_compressed(path, counts=counts)
                rows.append({'dataset': 'nefer' if i == 2 else 'source', 'clip_id': str(i),
                             'label': CLASSES[i], 'label_id': i, 'split': split,
                             'counts_path': str(path), 'counts_sha256': digest(path)})
            csv_write(root / 'index.csv', rows)
            json_write(root / 'config.json', {'bins': 3})
            lock_directory(root, ['index.csv', 'config.json'])
            with contextlib.redirect_stdout(io.StringIO()):
                fit_reference(argparse.Namespace(index=root / 'index.csv', output=root / 'statistics'))
            fitted = json.loads((root / 'statistics/statistics.json').read_text())
            self.assertEqual(fitted['reference_count'], 3.5)
            self.assertEqual(fitted['training_clips'], 1)
            (root / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_lock(root / 'index.csv')


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_feature_encoding_batches_and_dimensions(self):
        class FakeBackbone(torch.nn.Module):
            def forward(self, images):
                return images.mean((1, 2, 3))[:, None].expand(-1, 384)
        frames = np.ones((8, 3, 224, 224), dtype=np.float32)
        model = FakeBackbone().eval().requires_grad_(False)
        result = encode(model, frames, torch.device('cpu'), 3)
        self.assertEqual(result.shape, (8, 384))
        self.assertTrue(np.isfinite(result).all())

    def test_seed_repeats_initial_head(self):
        seed_everything(17)
        first = TemporalHead(input_dim=8).state_dict()
        seed_everything(17)
        second = TemporalHead(input_dim=8).state_dict()
        for key in first:
            torch.testing.assert_close(first[key], second[key])

    def test_fixed_six_class_metrics(self):
        result = metrics([0, 1], [0, 1])
        self.assertEqual(result['accuracy'], 1)
        self.assertAlmostEqual(result['macro_f1'], 2 / 6)

    def test_training_learns_tiny_features_and_saves_checkpoint(self):
        from eventfer.common import CLASSES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split in ['train', 'validation']:
                for cls in range(6):
                    values = np.zeros((8, 12), dtype=np.float32)
                    values[:, cls] = 3
                    path = root / f'{split}_{cls}.npy'
                    np.save(path, values)
                    rows.append({'dataset': 'source', 'clip_id': path.stem, 'split': split,
                                 'label': CLASSES[cls], 'label_id': str(cls),
                                 'features_path': str(path), 'features_sha256': digest(path)})
            index = root / 'index.csv'
            csv_write(index, rows)
            args = argparse.Namespace(index=index, output=root / 'run', device='cpu',
                                      hidden_dim=24, dropout=0., lr=.03, weight_decay=0.,
                                      batch_size=6, epochs=50, patience=12, tiny_per_class=0)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train_one(args, rows, {'embedding_dim': 12, 'bins': 8}, 17)
            self.assertEqual(summary['source_validation_macro_f1'], 1.)
            self.assertFalse(summary['test_evaluated'])
            saved = torch.load(root / 'run/seed_17/best.pt', map_location='cpu', weights_only=True)
            restored = TemporalHead(**saved['model_config'])
            restored.load_state_dict(saved['state_dict'])
            self.assertEqual(restored(torch.zeros(2, 8, 12)).shape, (2, 6))


if __name__ == '__main__':
    unittest.main()
