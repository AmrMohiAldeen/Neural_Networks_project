"""Build count caches and fit scaling statistics from source training only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import csv_write, digest, fresh_directory, json_write, lock_directory, read_rows, uid, verify_lock
from .events import count_clip


def load_counts(row):
    if digest(row['counts_path']) != row['counts_sha256']:
        raise ValueError(f'Count cache changed: {row["clip_id"]}')
    with np.load(row['counts_path'], allow_pickle=False) as f:
        return f['counts']


def build_counts(args):
    verify_lock(args.manifest)
    selected = [r for r in read_rows(args.manifest) if r['split'] in args.splits]
    if not selected or set(args.splits) - {r['split'] for r in selected}:
        raise ValueError('Requested splits missing from manifest')
    crops = {}
    if args.crop_policy == 'fixed-boxes':
        if not args.crops:
            raise ValueError('--crops required for fixed-boxes policy')
        with args.crops.open(newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                key = (r['dataset'], r['clip_id'])
                if key in crops:
                    raise ValueError(f'Duplicate crop entry: {key}')
                crops[key] = tuple(int(r[k]) for k in ('x0', 'y0', 'x1', 'y1'))
    elif args.crops:
        raise ValueError('--crops cannot be used with full-frame policy')
    output = fresh_directory(args.output)
    (output / 'counts').mkdir()
    rows = []
    for i, source in enumerate(selected, 1):
        row = dict(source)
        if row['modality'] != 'events':
            raise ValueError('Link RGB manifest to converted event files before preprocessing')
        if digest(row['path']) != row['sha256']:
            raise ValueError(f'Event file changed: {row["path"]}')
        key = (row['dataset'], row['clip_id'])
        if args.crop_policy == 'fixed-boxes' and key not in crops:
            raise ValueError(f'Missing fixed crop: {key}')
        decoded_root = getattr(args, 'decoded_raw_root', None)
        reader_row = row
        decoded_info = None
        if decoded_root and Path(row['path']).suffix.lower() in {'.raw', '.dat'}:
            sidecar = decoded_root / f'{uid(row)}.json'
            decoded_info = json.loads(sidecar.read_text())
            decoded = decoded_root / f'{uid(row)}.h5'
            if (decoded_info['source_sha256'] != row['sha256']
                    or decoded_info['clip_id'] != row['clip_id']
                    or decoded_info['dataset'] != row['dataset']
                    or digest(decoded) != decoded_info['h5_sha256']):
                raise ValueError('Decoded RAW cache does not match the frozen original recording')
            reader_row = dict(row, path=str(decoded), width=str(decoded_info['width']),
                              height=str(decoded_info['height']), polarity_encoding='binary')
        counts, metadata = count_clip(reader_row, crops.get(key), bins=args.bins, size=args.size)
        if decoded_info:
            metadata['decoded_h5_sha256'] = decoded_info['h5_sha256']
        path = output / 'counts' / f'{uid(row)}.npz'
        np.savez_compressed(path, counts=counts, metadata=np.asarray(json.dumps(metadata)))
        row.update(counts_path=str(path), counts_sha256=digest(path))
        rows.append(row)
        print(f'[{i}/{len(selected)}] {row["clip_id"]}: {int(counts.sum())} retained events', flush=True)
    csv_write(output / 'index.csv', rows)
    json_write(output / 'config.json', {
        'manifest_sha256': digest(args.manifest), 'bins': args.bins, 'size': args.size,
        'crop_policy': args.crop_policy, 'crops_sha256': digest(args.crops) if args.crops else None,
        'splits': args.splits, 'channel_order': ['positive', 'negative'],
        'time_binning': f'{args.bins} equal intervals over first to last full-stream event; final endpoint included',
        'spatial_projection': 'fixed clip crop; floor coordinate scaling to grid; no moving crop',
        'implementation_sha256': digest(Path(__file__).with_name('events.py')),
        'cache_builder_sha256': digest(Path(__file__)),
        'decoded_raw_root': str(args.decoded_raw_root.resolve()) if getattr(args, 'decoded_raw_root', None) else None,
    })
    lock_directory(output, ['index.csv', 'config.json'])


def fit_reference(args):
    verify_lock(args.index)
    rows = [r for r in read_rows(args.index) if r['split'] == 'train']
    if not rows:
        raise ValueError('No source training clips in count cache')
    nonempty = []
    for row in rows:
        totals = load_counts(row).sum(axis=(1, 2, 3), dtype=np.float64)
        nonempty.extend(totals[totals > 0].tolist())
    if not nonempty:
        raise ValueError('Source training crops contain no events')
    output = fresh_directory(args.output)
    json_write(output / 'statistics.json', {
        'reference_count': float(np.percentile(nonempty, 75, method='linear')),
        'percentile': 75, 'percentile_method': 'linear', 'fitted_split': 'train',
        'training_clips': len(rows), 'nonempty_training_bins': len(nonempty),
        'index_sha256': digest(args.index),
        'counts_config': json.loads((args.index.parent / 'config.json').read_text()),
        'compression': 'log1p(count)/log(256); no per-frame rescaling or clipping',
        'backbone_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
    })
    lock_directory(output, ['statistics.json'])
    print(f'Source-only statistics saved: {output / "statistics.json"}')


def preview(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .events import represent
    verify_lock(args.index)
    verify_lock(args.statistics)
    statistics = json.loads(args.statistics.read_text())
    rows = [r for r in read_rows(args.index) if r['split'] == 'train']
    if not rows:
        raise ValueError('Preview requires a source training clip')
    if args.clip_id:
        rows = [r for r in rows if r['clip_id'] == args.clip_id]
        if len(rows) != 1:
            raise ValueError('Specify a unique source training clip_id')
    counts = load_counts(rows[0])
    fig, axes = plt.subplots(3, len(counts), figsize=(2 * len(counts), 6), squeeze=False)
    for row_idx, mode in enumerate(('baseline', 'per_bin', 'whole_clip')):
        frames, scales = represent(counts, mode, statistics['reference_count'])
        for b in range(len(counts)):
            axes[row_idx, b].imshow(np.clip(frames[b].transpose(1, 2, 0), 0, 1))
            axes[row_idx, b].set_title(f'bin {b + 1}; scale={scales[b]:.2f}', fontsize=8)
            axes[row_idx, b].set_xticks([])
            axes[row_idx, b].set_yticks([])
        axes[row_idx, 0].set_ylabel(mode)
    fig.suptitle(f'{rows[0]["clip_id"]}\nR=positive, G=negative, B=sum; display clipped to [0,1] only')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f'Preview saved: {args.output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    counts = sub.add_parser('counts')
    counts.add_argument('--manifest', type=Path, required=True)
    counts.add_argument('--output', type=Path, required=True)
    counts.add_argument('--splits', nargs='+', default=['train', 'validation'])
    counts.add_argument('--crop-policy', choices=['full-frame', 'fixed-boxes'], required=True)
    counts.add_argument('--crops', type=Path)
    counts.add_argument('--bins', type=int, default=8)
    counts.add_argument('--size', type=int, default=224)
    counts.add_argument('--decoded-raw-root', type=Path, help='Verified H5 exports from eventfer.export_raw')
    stats = sub.add_parser('fit')
    stats.add_argument('--index', type=Path, required=True)
    stats.add_argument('--output', type=Path, required=True)
    view = sub.add_parser('preview')
    view.add_argument('--index', type=Path, required=True)
    view.add_argument('--statistics', type=Path, required=True)
    view.add_argument('--clip-id')
    view.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    {'counts': build_counts, 'fit': fit_reference, 'preview': preview}[args.command](args)


if __name__ == '__main__':
    main()
