"""Stage only frozen source RGB clips for the official batch converter."""
import argparse
import os
from pathlib import Path

from .common import digest, fresh_directory, json_write, read_rows, verify_lock


def stage(args):
    if args.output.exists():
        raise FileExistsError(f'Staging output already exists: {args.output}')
    verify_lock(args.manifest)
    root = args.rgb_root.resolve()
    rows = [r for r in read_rows(args.manifest)
            if r['dataset'] == args.source_dataset and r['split'] in {'train', 'validation', 'test'}]
    if not rows:
        raise ValueError('No selected source recordings')
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} source recordings; found {len(rows)}')
    links, relative_paths = [], set()
    for i, row in enumerate(rows, 1):
        video = Path(row['path']).resolve()
        if row['modality'] != 'rgb' or video.suffix.lower() != '.mp4' or not video.is_relative_to(root):
            raise ValueError(f'Expected source MP4 inside RGB root: {video}')
        relative = video.relative_to(root)
        if relative in relative_paths:
            raise ValueError(f'Duplicate staging path: {relative}')
        relative_paths.add(relative)
        if digest(video) != row['sha256']:
            raise ValueError(f'Source recording changed since split freeze: {video}')
        links.append((video, relative, row))
        if i % 100 == 0:
            print(f'Verified source recordings: {i}/{len(rows)}', flush=True)
    output = fresh_directory(args.output)
    for video, relative, row in links:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.link_mode == 'symlink':
            destination.symlink_to(video)
        else:
            os.link(video, destination)
    json_write(output / 'staging.json', {
        'manifest': str(args.manifest.resolve()), 'manifest_sha256': digest(args.manifest),
        'rgb_root': str(root), 'source_dataset': args.source_dataset,
        'source_clips': len(rows), 'link_mode': args.link_mode,
        'clips': [{'clip_id': row['clip_id'], 'split': row['split'],
                   'relative_path': relative.as_posix(), 'sha256': row['sha256']}
                  for video, relative, row in links],
        'note': 'Only frozen source RGB clips staged. No recordings copied, renamed, or deleted; splits preserved.',
    })
    print(f'Staged {len(rows)} verified source videos: {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-dataset', default='ekman_rgb')
    p.add_argument('--expected-count', type=int)
    p.add_argument('--link-mode', choices=['symlink', 'hardlink'], default='symlink')
    stage(p.parse_args())


if __name__ == '__main__':
    main()
