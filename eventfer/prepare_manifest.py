"""Link converted v2e files to existing frozen RGB splits without resplitting."""
import argparse
from pathlib import Path

from .common import csv_write, digest, fresh_directory, json_write, lock_directory, read_rows, verify_lock


def prepare(args):
    verify_lock(args.manifest)
    rows = read_rows(args.manifest)
    source = args.source_rgb_root.resolve()
    events = args.source_events_root.resolve()
    linked = []
    for original in rows:
        row = dict(original)
        if row['modality'] == 'rgb':
            video = Path(row['path']).resolve()
            if not video.is_relative_to(source):
                raise ValueError(f'RGB file outside provided root: {video}')
            if digest(video) != row['sha256']:
                raise ValueError(f'RGB file changed since split freeze: {video}')
            path = events / video.relative_to(source).with_suffix('') / 'events.h5'
            if not path.is_file():
                raise ValueError(f'Converted stream missing: {path}')
            row.update(path=str(path), modality='events', sha256=digest(path),
                       width=str(args.source_width), height=str(args.source_height))
        else:
            path = Path(row['path'])
            if digest(path) != row['sha256']:
                raise ValueError(f'Event file changed since freeze: {path}')
            row.setdefault('width', '')
            row.setdefault('height', '')
            if row['split'] in {'train', 'validation', 'test'}:
                row['width'] = row['width'] or str(args.source_width)
                row['height'] = row['height'] or str(args.source_height)
            elif args.target_width and args.target_height:
                row['width'], row['height'] = str(args.target_width), str(args.target_height)
        row['polarity_encoding'] = row.get('polarity_encoding') or 'binary'
        linked.append(row)
    output = fresh_directory(args.output)
    csv_write(output / 'manifest.csv', linked)
    json_write(output / 'provenance.json', {
        'parent_manifest': str(args.manifest.resolve()), 'parent_sha256': digest(args.manifest),
        'source_rgb_root': str(source), 'source_events_root': str(events),
        'note': 'Existing labels, clip IDs, groups, and splits retained; locally simulated events linked.',
    })
    lock_directory(output, ['manifest.csv', 'provenance.json'])
    print(f'Linked {len(linked)} clips: {output / "manifest.csv"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--source-rgb-root', type=Path, required=True)
    p.add_argument('--source-events-root', type=Path, required=True)
    p.add_argument('--source-width', type=int, default=336)
    p.add_argument('--source-height', type=int, default=224)
    p.add_argument('--target-width', type=int)
    p.add_argument('--target-height', type=int)
    p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
