"""Check the pinned v2e HDF5 layout: uint32 [t_us, x, y, polarity_0_or_1]."""
import argparse
import json
from pathlib import Path
import sys


def validate(path, width, height):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as stream:
        events = stream['events']
        if events.ndim != 2 or events.shape[1] != 4 or events.dtype != np.dtype('uint32'):
            raise ValueError(f'Unexpected event shape/dtype: {events.shape}, {events.dtype}')
        if not len(events):
            raise ValueError('No events')
        previous, first, polarities = None, None, set()
        for start in range(0, len(events), 1_000_000):
            chunk = events[start:start + 1_000_000]
            if first is None:
                first = int(chunk[0, 0])
            if (previous is not None and int(chunk[0, 0]) < previous) or np.any(chunk[1:, 0] < chunk[:-1, 0]):
                raise ValueError('Timestamps decrease (or uint32 time wrapped)')
            if np.any(chunk[:, 1] >= width) or np.any(chunk[:, 2] >= height):
                raise ValueError('Event coordinates outside configured sensor dimensions')
            polarities.update(int(x) for x in np.unique(chunk[:, 3]))
            if not polarities.issubset({0, 1}):
                raise ValueError(f'Unexpected polarity codes: {sorted(polarities)}')
            previous = int(chunk[-1, 0])
        return {'events': len(events), 'duration_seconds': (previous - first) / 1e6,
                'first_timestamp_us': first, 'last_timestamp_us': previous,
                'polarities': sorted(polarities)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--width', type=int, default=336)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--expected-input-root', type=Path, help='Require a matching events.h5 for every RGB MP4')
    parser.add_argument('--include-list', type=Path,
                        help='Only check these input-relative MP4 paths (one per line) under --root')
    parser.add_argument('--output', type=Path, default=Path('artifacts/event_integrity.json'))
    args = parser.parse_args()
    wanted = None
    if args.include_list:
        wanted = sorted({line.strip() for line in args.include_list.read_text(encoding='utf-8').splitlines() if line.strip()})
        if not wanted:
            raise SystemExit('Empty include list')
    expected_paths = [args.root / Path(rel).with_suffix('') / 'events.h5' for rel in wanted] if wanted else None
    paths = [p for p in expected_paths if p.is_file()] if wanted else sorted(args.root.rglob('events.h5'))
    report = {'format': 'v2e-1.7.0 H5', 'timestamp_unit': 'microseconds',
              'width': args.width, 'height': args.height, 'valid': [], 'invalid': [], 'missing': []}
    for path in paths:
        try:
            report['valid'].append({'path': str(path), **validate(path, args.width, args.height)})
        except Exception as error:
            report['invalid'].append({'path': str(path), 'error': str(error)})
    if wanted:
        report['missing'] = [str(p) for p in expected_paths if not p.is_file()]
        report['include_list'] = str(args.include_list)
        if args.expected_input_root:
            absent = [rel for rel in wanted if not (args.expected_input_root / rel).is_file()]
            if absent:
                raise SystemExit(f'{len(absent)} listed videos are absent from the input root, e.g. {absent[0]}')
    elif args.expected_input_root:
        videos = sorted(p for p in args.expected_input_root.rglob('*') if p.suffix.lower() == '.mp4')
        if not videos:
            raise SystemExit('No input MP4 videos found')
        for video in videos:
            expected = args.root / video.relative_to(args.expected_input_root).with_suffix('') / 'events.h5'
            if not expected.is_file():
                report['missing'].append(str(expected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Valid={len(report["valid"])} Invalid={len(report["invalid"])} Missing={len(report["missing"])} Report={args.output}')
    if not paths or report['invalid'] or report['missing']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
