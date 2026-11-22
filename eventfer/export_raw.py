"""Export locked target RAW files with an SDK-compatible Python interpreter.

Run under the interpreter that can import metavision_core. The resulting H5
cache can then be read from the PyTorch Conda environment without that SDK.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .common import digest, fresh_directory, json_write, read_rows, uid, verify_lock
from .events import scan


def export_one(row, output, iterator_factory):
    if digest(row['path']) != row['sha256']:
        raise ValueError(f'RAW changed since manifest freeze: {row["path"]}')
    reader = iterator_factory(input_path=row['path'], mode='n_events', n_events=1_000_000,
                              relative_timestamps=False)
    height, width = reader.get_size()
    if not width or not height:
        raise ValueError('Reader did not report sensor dimensions')
    path = output / f'{uid(row)}.h5'
    partial = output / f'{uid(row)}.part.h5'
    with h5py.File(partial, 'w') as f:
        events = f.create_dataset('events', shape=(0, 4), maxshape=(None, 4),
                                  dtype='int64', chunks=True, compression='gzip')
        for raw in reader:
            if not len(raw):
                continue
            data = np.column_stack([raw[k].astype(np.int64) for k in ('t', 'x', 'y', 'p')])
            start = len(events)
            events.resize(start + len(data), axis=0)
            events[start:] = data
        f.attrs['timestamp_unit'] = 'microseconds'
        f.attrs['width'], f.attrs['height'] = int(width), int(height)
    first, last, count = scan(partial, int(width), int(height))
    partial.replace(path)
    metadata = {'dataset': row['dataset'], 'clip_id': row['clip_id'],
                'source_sha256': row['sha256'], 'source_path': row['path'],
                'h5_sha256': digest(path), 'width': int(width), 'height': int(height),
                'events': count, 'start_us': first, 'end_us': last,
                'timestamp_unit': 'microseconds', 'polarity_encoding': 'binary'}
    json_write(output / f'{uid(row)}.json', metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    verify_lock(args.manifest)
    from metavision_core.event_io import EventsIterator
    rows = [r for r in read_rows(args.manifest) if r['split'] == 'target_test']
    if not rows:
        raise ValueError('No target_test recordings')
    if args.output.exists():
        output = args.output.resolve()
        metadata = json.loads((output / 'export.json').read_text())
        if metadata['manifest_sha256'] != digest(args.manifest):
            raise ValueError('Export directory belongs to another frozen manifest')
    else:
        output = fresh_directory(args.output)
        json_write(output / 'export.json', {'manifest_sha256': digest(args.manifest),
                                          'exporter_sha256': digest(Path(__file__))})
    for i, row in enumerate(rows, 1):
        sidecar, path = output / f'{uid(row)}.json', output / f'{uid(row)}.h5'
        if sidecar.is_file() and path.is_file():
            info = json.loads(sidecar.read_text())
            if info['source_sha256'] == row['sha256'] and digest(path) == info['h5_sha256']:
                print(f'[{i}/{len(rows)}] verified existing {row["clip_id"]}', flush=True)
                continue
            raise ValueError('Existing RAW export failed integrity check')
        info = export_one(row, output, EventsIterator)
        print(f'[{i}/{len(rows)}] {row["clip_id"]}: {info["events"]} events', flush=True)
    print('All target RAW exports completed and validated.', flush=True)


if __name__ == '__main__':
    main()
