"""Create, train, and freeze the three-condition, three-seed experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .common import CLASSES, digest, fresh_directory, json_write, lock_directory, read_rows, verify_lock
from .protocol import MODES, code_hashes, event_rate, identity, key, recipe
from .train import train_one

BUDGET_KEYS = ('epochs', 'patience', 'batch_size', 'hidden_dim', 'dropout', 'lr', 'weight_decay')


def load_plan(directory):
    path = Path(directory).resolve() / 'plan.json'
    verify_lock(path)
    plan = json.loads(path.read_text())
    if plan['code_sha256'] != code_hashes():
        raise ValueError('Code changed after experiment creation; preserve this plan and create a new version')
    for mode in MODES:
        index = Path(plan['caches'][mode]['index'])
        verify_lock(index)
        if digest(index) != plan['caches'][mode]['index_sha256']:
            raise ValueError(f'Training cache changed: {mode}')
    return plan


def create(args):
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3 or any(s < 0 or s >= 2 ** 32 for s in args.seeds):
        raise ValueError('Choose exactly three distinct valid seeds')
    if (min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1
            or not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0):
        raise ValueError('Invalid training budget')
    caches, reference_rows, reference_recipe = {}, None, None
    for mode in MODES:
        index = getattr(args, mode + '_index').resolve()
        verify_lock(index)
        rows = read_rows(index)
        config = json.loads((index.parent / 'config.json').read_text())
        if config['mode'] != mode or set(r['split'] for r in rows) != {'train', 'validation'}:
            raise ValueError('Each training cache must have its declared mode and source train/validation only')
        if config['statistics']['fitted_split'] != 'train':
            raise ValueError('Representation statistics must be fitted on source training only')
        signature = [dict(identity(r), counts_sha256=r['counts_sha256']) for r in rows]
        if reference_rows is None:
            reference_rows, reference_recipe = signature, recipe(config)
        elif reference_rows != signature or reference_recipe != recipe(config):
            raise ValueError('Conditions differ in clip order, labels, splits, counts, backbone, or source statistics')
        for row in rows:
            if digest(row['features_path']) != row['features_sha256']:
                raise ValueError(f'Feature file changed: {row["clip_id"]}')
        caches[mode] = {'index': str(index), 'index_sha256': digest(index), 'config': config}
    train_rows = [r for r in rows if r['split'] == 'train']
    validation_rows = [r for r in rows if r['split'] == 'validation']
    if reference_recipe['backbone'] != 'dinov2_vits14' or reference_recipe['embedding_dim'] != 384 or reference_recipe['bins'] != 8:
        raise ValueError('This study requires eight DINOv2 ViT-S/14 embeddings of dimension 384')
    for split_rows in (train_rows, validation_rows):
        if {r['label'] for r in split_rows} != set(CLASSES):
            raise ValueError('All six source classes are required in train and validation')
    if args.expected_train_counts and args.expected_train_counts != [len(train_rows), len(validation_rows)]:
        raise ValueError('Unexpected source training/validation counts')
    rates = [event_rate(r) for r in train_rows]
    cutoffs = np.quantile(rates, [1 / 3, 2 / 3], method='linear').tolist()
    output = fresh_directory(args.output)
    json_write(output / 'plan.json', {
        'status': 'source_development_only', 'seeds': args.seeds, 'modes': list(MODES),
        'caches': caches, 'recipe': reference_recipe,
        'budget': {name: getattr(args, name) for name in BUDGET_KEYS},
        'training_clips': len(train_rows), 'validation_clips': len(validation_rows),
        'density': {'definition': 'retained unscaled events per second over first-to-last clip event; min duration 1 microsecond',
                    'cutoffs': cutoffs, 'fitted_split': 'source_train', 'quantiles': [1 / 3, 2 / 3]},
        'hypothesis': {'target_gain_percentage_points_at_least': 3.,
                       'source_loss_percentage_points_at_most': 2.},
        'class_order': CLASSES, 'code_sha256': code_hashes(),
    })
    lock_directory(output, ['plan.json'])
    print(f'Created controlled experiment: {output}; nine head-training runs planned')


def checked_run(directory, plan, mode, seed):
    run = Path(directory).resolve() / 'models' / mode / f'seed_{seed}'
    summary = json.loads((run / 'summary.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if summary.get('tiny_correctness_run') or summary.get('test_evaluated'):
        raise ValueError('Controlled experiment cannot use a tiny or previously test-selected run')
    expected_training = [r['clip_id'] for r in read_rows(Path(plan['caches'][mode]['index'])) if r['split'] == 'train']
    if config.get('tiny_per_class') != 0 or config['selected_training_clips'] != expected_training:
        raise ValueError('Run did not use the full planned source training set')
    if digest(run / 'best.pt') != summary['checkpoint_sha256'] or digest(run / 'config.json') != summary['config_sha256']:
        raise ValueError('Saved model or run configuration changed')
    if config['seed'] != seed or config['feature_index_sha256'] != plan['caches'][mode]['index_sha256']:
        raise ValueError('Run seed/cache differs from experiment plan')
    if config['trainer_sha256'] != plan['code_sha256']['train.py']:
        raise ValueError('Trainer code differs from experiment plan')
    if any(config[name] != plan['budget'][name] for name in BUDGET_KEYS):
        raise ValueError('Training budget differs from plan')
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    best = max(history, key=lambda record: record['validation']['macro_f1'])
    if summary['best_epoch'] != best['epoch']:
        raise ValueError('Checkpoint was not selected by first best source-validation macro-F1')
    if summary['source_validation_macro_f1'] != best['validation']['macro_f1']:
        raise ValueError('Validation summary differs from source-validation history')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['seed'] != seed or checkpoint['epoch'] != best['epoch'] or checkpoint['validation'] != best['validation']:
        raise ValueError('Checkpoint metadata differs from source validation history')
    if checkpoint['feature_index_sha256'] != config['feature_index_sha256']:
        raise ValueError('Checkpoint input provenance differs from saved config')
    return run, config, summary


def train(args):
    directory = args.experiment.resolve()
    plan = load_plan(directory)
    if (directory / 'frozen').exists():
        raise ValueError('Experiment is frozen for evaluation; training is closed')
    for mode in MODES:
        index = Path(plan['caches'][mode]['index'])
        rows = read_rows(index)
        config = plan['caches'][mode]['config']
        summaries = []
        for seed in plan['seeds']:
            run = directory / 'models' / mode / f'seed_{seed}'
            if run.exists():
                if not (run / 'summary.json').is_file():
                    raise ValueError(f'Incomplete run: {run}. Inspect/archive it before retrying; no overwrite is performed.')
                _, _, summary = checked_run(directory, plan, mode, seed)
                print(f'Verified existing completed run: {mode}, seed {seed}', flush=True)
            else:
                options = argparse.Namespace(index=index, output=directory / 'models' / mode,
                                             device=args.device, tiny_per_class=0, **plan['budget'])
                summary = train_one(options, rows, config, seed)
            summaries.append(summary)
        json_write(directory / 'models' / mode / 'validation_summary.json', summaries)
    print('All nine source-only training runs completed. Freeze before test evaluation.')


def freeze(args):
    directory = args.experiment.resolve()
    plan = load_plan(directory)
    verify_lock(args.manifest)
    if digest(args.manifest) != plan['recipe']['counts']['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from training count-cache manifest')
    rows = read_rows(args.manifest)
    tests = [r for r in rows if r['split'] in {'test', 'target_test'}]
    counts = {split: sum(r['split'] == split for r in tests) for split in ('test', 'target_test')}
    if not all(counts.values()) or (args.expected_test_counts and args.expected_test_counts != [counts['test'], counts['target_test']]):
        raise ValueError('Missing or unexpected test clip counts')
    expected_users = ['03', '07', '17', '19', '27', '28']
    observed_users = sorted({r['participant_id'] for r in tests if r['split'] == 'target_test'})
    if not set(observed_users).issubset(expected_users):
        raise ValueError('Unexpected NEFER target participant')
    crop_policy = plan['recipe']['counts']['crop_policy']
    if crop_policy == 'fixed-boxes':
        if not args.evaluation_crops:
            raise ValueError('Freeze a label-blind evaluation crop CSV for all test clips')
        with args.evaluation_crops.open(newline='', encoding='utf-8-sig') as f:
            crop_rows = list(csv.DictReader(f))
        crops = {}
        for row in crop_rows:
            k = key(row)
            if k in crops:
                raise ValueError('Duplicate evaluation crop entry')
            box = tuple(int(row[field]) for field in ('x0', 'y0', 'x1', 'y1'))
            if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                raise ValueError('Invalid evaluation crop box')
            crops[k] = box
        if any(key(row) not in crops for row in tests):
            raise ValueError('Evaluation crop annotations do not cover every test clip')
        crops_sha256 = digest(args.evaluation_crops)
    else:
        if args.evaluation_crops:
            raise ValueError('Full-frame evaluation must not supply crop annotations')
        crops_sha256 = None
    models, initial_states, selection = {}, {}, None
    for mode in MODES:
        models[mode] = {}
        for seed in plan['seeds']:
            run, config, summary = checked_run(directory, plan, mode, seed)
            if seed in initial_states and initial_states[seed] != config['initial_state_sha256']:
                raise ValueError('Paired conditions did not start from the same head weights')
            initial_states[seed] = config['initial_state_sha256']
            signature = (config['class_weights'], config['selected_training_clips'], config['model'])
            if selection is None:
                selection = signature
            elif signature != selection:
                raise ValueError('Conditions differ in class weights, source clips, or architecture')
            models[mode][str(seed)] = {
                'checkpoint': str(run / 'best.pt'), 'checkpoint_sha256': summary['checkpoint_sha256'],
                'run_config': str(run / 'config.json'), 'run_config_sha256': digest(run / 'config.json'),
                'history': str(run / 'history.jsonl'), 'history_sha256': digest(run / 'history.jsonl'),
                'best_epoch': summary['best_epoch'], 'initial_state_sha256': config['initial_state_sha256'],
            }
    output = fresh_directory(directory / 'frozen')
    json_write(output / 'experiment.json', {
        'status': 'locked_before_test_scoring', 'plan_sha256': digest(directory / 'plan.json'),
        'plan_path': str(directory / 'plan.json'),
        'seeds': plan['seeds'], 'modes': list(MODES), 'models': models, 'recipe': plan['recipe'],
        'density': plan['density'], 'hypothesis': plan['hypothesis'], 'class_order': CLASSES,
        'evaluation_manifest_sha256': digest(args.manifest),
        'evaluation_crops_sha256': crops_sha256,
        'evaluation_rows': {key(r): identity(r) for r in tests}, 'test_counts': counts,
        'expected_target_participants': expected_users, 'observed_target_participants': observed_users,
        'metrics': 'clip-level macro-F1 over fixed six classes, undefined class F1=0; accuracy secondary',
        'uncertainty': 'mean and sample standard deviation across three seeds; paired seed differences; no significance claim',
        'code_sha256': code_hashes(),
    })
    lock_directory(output, ['experiment.json'])
    print(f'Locked all nine checkpoints and evaluation rules: {output / "experiment.json"}')


def status(args):
    plan = load_plan(args.experiment)
    rows = []
    for mode in MODES:
        for seed in plan['seeds']:
            run = args.experiment.resolve() / 'models' / mode / f'seed_{seed}'
            if (run / 'summary.json').is_file():
                _, _, summary = checked_run(args.experiment, plan, mode, seed)
                rows.append({'mode': mode, 'seed': seed, 'status': 'verified_complete',
                             'source_validation_macro_f1': summary['source_validation_macro_f1']})
            else:
                rows.append({'mode': mode, 'seed': seed,
                             'status': 'incomplete_or_running' if run.exists() else 'not_started'})
    print(json.dumps({'frozen': (args.experiment / 'frozen/experiment.json').is_file(), 'runs': rows}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    make = sub.add_parser('create')
    for mode in MODES:
        make.add_argument('--' + mode.replace('_', '-') + '-index', dest=mode + '_index', type=Path, required=True)
    make.add_argument('--output', type=Path, required=True)
    make.add_argument('--seeds', nargs=3, type=int, default=[17, 29, 43])
    make.add_argument('--expected-train-counts', nargs=2, type=int)
    for name, default, kind in [('epochs', 50, int), ('patience', 8, int), ('batch_size', 32, int),
                                ('hidden_dim', 128, int), ('dropout', .2, float), ('lr', .001, float), ('weight_decay', .0001, float)]:
        make.add_argument('--' + name.replace('_', '-'), dest=name, default=default, type=kind)
    run = sub.add_parser('train')
    run.add_argument('--experiment', type=Path, required=True)
    run.add_argument('--device', default='cuda')
    lock = sub.add_parser('freeze')
    lock.add_argument('--experiment', type=Path, required=True)
    lock.add_argument('--manifest', type=Path, required=True)
    lock.add_argument('--evaluation-crops', type=Path)
    lock.add_argument('--expected-test-counts', nargs=2, type=int)
    inspect = sub.add_parser('status')
    inspect.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    {'create': create, 'train': train, 'freeze': freeze, 'status': status}[args.command](args)


if __name__ == '__main__':
    main()
