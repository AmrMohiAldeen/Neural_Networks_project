"""Shared recipe, identity, and provenance checks for the controlled experiment."""
import json
from pathlib import Path

import numpy as np

from .common import digest, verify_lock

MODES = ('baseline', 'per_bin', 'whole_clip')
COUNT_KEYS = ('manifest_sha256', 'bins', 'size', 'crop_policy', 'implementation_sha256')
CODE_FILES = ('common.py', 'events.py', 'preprocess.py', 'features.py', 'train.py',
              'protocol.py', 'experiment.py', 'evaluate.py', 'export_raw.py')


def key(row):
    return row['dataset'] + '\0' + row['clip_id']


def identity(row):
    return {field: row.get(field, '') for field in
            ('dataset', 'clip_id', 'split', 'label', 'label_id', 'participant_id',
             'original_video_id', 'sha256', 'expected_label')}


def recipe(config):
    return {
        'statistics_sha256': config['statistics_sha256'],
        'backbone': config['backbone'], 'backbone_commit': config['backbone_commit'],
        'weights_sha256': config['weights_sha256'], 'embedding_dim': config['embedding_dim'],
        'backbone_torch_version': config['torch_version'],
        'bins': config['bins'], 'representation_sha256': config['representation_sha256'],
        'counts': {k: config['count_config'][k] for k in COUNT_KEYS},
        'source_crops_sha256': config['count_config']['crops_sha256'],
        'compression': config['statistics']['compression'],
        'normalization': config['statistics']['backbone_normalization'],
    }


def code_hashes():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in CODE_FILES}


def load_frozen(path):
    verify_lock(path)
    frozen = json.loads(Path(path).read_text(encoding='utf-8'))
    if frozen.get('status') != 'locked_before_test_scoring':
        raise ValueError('A completed pre-evaluation freeze is required')
    verify_lock(frozen['plan_path'])
    if digest(frozen['plan_path']) != frozen['plan_sha256']:
        raise ValueError('Source experiment plan changed since evaluation freeze')
    if frozen['code_sha256'] != code_hashes():
        raise ValueError('Experiment code changed since evaluation freeze; do not silently rescore')
    return frozen


def check_evaluation_counts(config, frozen):
    for name in COUNT_KEYS:
        if config[name] != frozen['recipe']['counts'][name]:
            raise ValueError(f'Evaluation preprocessing differs from locked recipe: {name}')
    if config['crops_sha256'] != frozen['evaluation_crops_sha256']:
        raise ValueError('Evaluation crop annotations differ from the predeclared file')
    if set(config['splits']) != {'test', 'target_test'}:
        raise ValueError('Evaluation counts must contain source test and NEFER target_test only')


def check_evaluation_features(config, frozen, mode):
    if config['mode'] != mode:
        raise ValueError(f'Wrong representation: expected {mode}')
    check_evaluation_counts(config['count_config'], frozen)
    actual = recipe(config)
    expected = dict(frozen['recipe'])
    expected['source_crops_sha256'] = frozen['evaluation_crops_sha256']
    if actual != expected:
        raise ValueError('Evaluation embeddings do not match frozen backbone/statistics/representation recipe')


def event_rate(row):
    """Retained events per second, before scaling, using clip event timestamps."""
    if digest(row['counts_path']) != row['counts_sha256']:
        raise ValueError(f'Count cache changed: {row["clip_id"]}')
    with np.load(row['counts_path'], allow_pickle=False) as f:
        metadata = json.loads(str(f['metadata']))
        duration = max((metadata['end_us'] - metadata['start_us']) / 1e6, 1e-6)
        return float(f['counts'].sum(dtype=np.float64) / duration)
