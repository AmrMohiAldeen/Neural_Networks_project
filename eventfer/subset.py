"""Freeze a deterministic, split- and class-stratified source subset of a frozen manifest.

Source rows keep their existing split assignment (no resplitting), so the
group-aware separation of the parent split is preserved. Every NEFER row is
kept unchanged. The result is a new locked metadata version; the parent is
only read.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .common import CLASSES, csv_write, digest, fresh_directory, json_write, lock_directory, read_rows, verify_lock

SOURCE_SPLITS = ('train', 'validation', 'test')


def allocate(sizes, total, minimum=0):
    """Largest-remainder proportional allocation with a per-key minimum and cap."""
    available = sum(sizes.values())
    if total > available:
        raise ValueError(f'Requested {total} clips but only {available} are available')
    if minimum * len(sizes) > total:
        raise ValueError(f'{total} clips cannot give {minimum} to each of {len(sizes)} groups')
    quota = {k: total * n / available for k, n in sizes.items()}
    alloc = {k: min(sizes[k], max(minimum, math.floor(q))) for k, q in quota.items()}
    while sum(alloc.values()) < total:
        k = max((k for k in sizes if alloc[k] < sizes[k]), key=lambda k: (quota[k] - alloc[k], k))
        alloc[k] += 1
    while sum(alloc.values()) > total:
        k = min((k for k in sizes if alloc[k] > minimum), key=lambda k: (quota[k] - alloc[k], k))
        alloc[k] -= 1
    return alloc


def already_converted(row, rgb_root, events_root):
    """True if convert_events_parallel.py left a completion record matching the H5 size."""
    relative = Path(row['path']).resolve().relative_to(rgb_root)
    output = events_root / relative.with_suffix('')
    marker, h5 = output / '.conversion-success.json', output / 'events.h5'
    try:
        return json.loads(marker.read_text(encoding='utf-8'))['h5_size'] == h5.stat().st_size
    except (OSError, ValueError, KeyError):
        return False


def make_subset(args):
    verify_lock(args.manifest)
    rows = read_rows(args.manifest)
    rgb_root = args.rgb_root.resolve()
    source = [r for r in rows if r['dataset'] == args.source_dataset and r['split'] in SOURCE_SPLITS]
    if not source:
        raise ValueError('No source rows in manifest')
    for row in source:
        if not Path(row['path']).resolve().is_relative_to(rgb_root):
            raise ValueError(f'Source path outside --rgb-root: {row["path"]}')
    cells = {}
    for row in source:
        cells.setdefault((row['split'], row['label']), []).append(row)
    for split in SOURCE_SPLITS:
        missing = set(CLASSES) - {label for s, label in cells if s == split}
        if missing:
            raise ValueError(f'Parent {split} split lacks classes: {sorted(missing)}')
    split_sizes = {s: sum(len(v) for (sp, _), v in cells.items() if sp == s) for s in SOURCE_SPLITS}
    split_alloc = allocate(split_sizes, args.size, minimum=len(CLASSES))
    events_root = args.prefer_converted_root.resolve() if args.prefer_converted_root else None

    def order(row):
        # Label-blind deterministic order; optionally reuse finished conversions first.
        rank = hashlib.sha256(f'{args.seed}\0{row["clip_id"]}'.encode()).hexdigest()
        done = events_root is not None and already_converted(row, rgb_root, events_root)
        return (not done, rank)

    selected, table, reused = set(), [], 0
    for split in SOURCE_SPLITS:
        class_alloc = allocate({c: len(cells[(split, c)]) for c in CLASSES}, split_alloc[split], minimum=1)
        for label in CLASSES:
            pool = sorted(cells[(split, label)], key=order)
            chosen = pool[:class_alloc[label]]
            if events_root is not None:
                reused += sum(already_converted(r, rgb_root, events_root) for r in chosen)
            selected.update((r['dataset'], r['clip_id']) for r in chosen)
            table.append({'split': split, 'label': label, 'parent_clips': len(pool), 'subset_clips': len(chosen)})
    kept = [r for r in rows if (r['dataset'], r['clip_id']) in selected
            or not (r['dataset'] == args.source_dataset and r['split'] in SOURCE_SPLITS)]
    relative = sorted(Path(r['path']).resolve().relative_to(rgb_root).as_posix()
                      for r in kept if (r['dataset'], r['clip_id']) in selected)
    output = fresh_directory(args.output)
    csv_write(output / 'manifest.csv', kept)
    csv_write(output / 'class_counts.csv', table)
    (output / 'source_relative_paths.txt').write_text('\n'.join(relative) + '\n', encoding='utf-8')
    counts = {s: sum(r['split'] == s for r in kept) for s in sorted({r['split'] for r in kept})}
    json_write(output / 'subset.json', {
        'parent_manifest': str(args.manifest.resolve()), 'parent_sha256': digest(args.manifest),
        'source_dataset': args.source_dataset, 'requested_source_clips': args.size,
        'seed': args.seed, 'rgb_root': str(rgb_root),
        'split_allocation': split_alloc, 'clip_counts': counts,
        'preferred_converted_root': str(events_root) if events_root else None,
        'selected_already_converted': reused if events_root else None,
        'policy': ('Proportional largest-remainder allocation over source splits, then over the six classes '
                   'within each split (at least one clip per class). Within a split/class cell clips are '
                   'ordered by sha256(seed, clip_id); if preferred_converted_root is set, clips with a '
                   'completed conversion record come first. Labels select only the stratum, never a clip. '
                   'Split assignments unchanged; all NEFER rows kept.'),
    })
    lock_directory(output, ['manifest.csv', 'class_counts.csv', 'source_relative_paths.txt', 'subset.json'])
    print(f'Source subset: {split_alloc} (total {len(selected)})')
    if events_root is not None:
        print(f'Already converted and reused: {reused}; still to convert: {len(selected) - reused}')
    print(f'All rows by split: {counts}')
    print(f'Locked subset metadata: {output}')


def verify(args):
    path = args.directory / 'manifest.csv'
    verify_lock(path)
    info = json.loads((args.directory / 'subset.json').read_text(encoding='utf-8'))
    verify_lock(Path(info['parent_manifest']))
    if digest(info['parent_manifest']) != info['parent_sha256']:
        raise ValueError('Parent manifest changed since the subset was made')
    rows = read_rows(path)
    counts = {s: sum(r['split'] == s for r in rows) for s in sorted({r['split'] for r in rows})}
    if counts != info['clip_counts']:
        raise ValueError('Subset manifest counts differ from subset.json')
    for split in SOURCE_SPLITS:
        labels = {r['label'] for r in rows if r['split'] == split and r['dataset'] == info['source_dataset']}
        if labels != set(CLASSES):
            raise ValueError(f'{split} lacks classes: {sorted(set(CLASSES) - labels)}')
    print(f'PASS: {counts}')
    with (args.directory / 'class_counts.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            print(f'{row["split"]:<11}{row["label"]:<10}{row["subset_clips"]:>4} of {row["parent_clips"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    make = sub.add_parser('create')
    make.add_argument('--manifest', type=Path, required=True)
    make.add_argument('--rgb-root', type=Path, required=True)
    make.add_argument('--size', type=int, required=True, help='Total source clips across train/validation/test')
    make.add_argument('--seed', type=int, default=476)
    make.add_argument('--source-dataset', default='ekman_rgb')
    make.add_argument('--prefer-converted-root', type=Path,
                      help='Output root of convert_events_parallel.py; finished clips are chosen first in each cell')
    make.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    {'create': make_subset, 'verify': verify}[args.command](args)


if __name__ == '__main__':
    main()
