"""Bounded-memory event readers and fixed, clip-level spatial/temporal binning."""
from pathlib import Path

import h5py
import numpy as np


def chunks(path, chunk_size=1_000_000):
    path = Path(path)
    if path.suffix.lower() in {'.h5', '.hdf5'}:
        with h5py.File(path, 'r') as f:
            data = f['events']
            if data.ndim != 2 or data.shape[1] != 4:
                raise ValueError('H5 requires events[N,4] columns [t_us,x,y,p]')
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
    elif path.suffix.lower() == '.npz':
        # NPZ is loaded into memory; use H5 for large recordings.
        with np.load(path, allow_pickle=False) as f:
            arrays = [f[key] for key in ('t', 'x', 'y', 'p')]
            if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays):
                raise ValueError('NPZ requires equal-length 1D t_us,x,y,p arrays (timestamp key: t)')
            for i in range(0, len(arrays[0]), chunk_size):
                yield np.column_stack([a[i:i + chunk_size] for a in arrays])
    elif path.suffix.lower() in {'.raw', '.dat'}:
        try:
            from metavision_core.event_io import EventsIterator
        except ImportError as error:
            raise RuntimeError('NEFER RAW/DAT needs the Metavision/OpenEB Python event reader; install its SDK or export canonical H5/NPZ first.') from error
        iterator = EventsIterator(input_path=str(path), mode='n_events', n_events=chunk_size, relative_timestamps=False)
        for data in iterator:
            if len(data):
                yield np.column_stack([data[k].astype(np.int64) for k in ('t', 'x', 'y', 'p')])
    else:
        raise ValueError(f'Unsupported raw event format {path.suffix}; TBR images are not polarity counts')


def dimensions(row):
    if row.get('width') and row.get('height'):
        return int(row['width']), int(row['height'])
    if Path(row['path']).suffix.lower() in {'.raw', '.dat'}:
        from metavision_core.event_io import EventsIterator
        reader = EventsIterator(input_path=row['path'], mode='n_events', n_events=1)
        height, width = reader.get_size()
        return int(width), int(height)
    raise ValueError(f'Explicit width,height required for {row["path"]}; do not infer sensor size from active pixels')


def scan(path, width, height, encoding='binary'):
    first, previous, n = None, None, 0
    allowed = {0, 1} if encoding == 'binary' else {-1, 1}
    for data in chunks(path):
        if not len(data):
            continue
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError('Events must use integer microsecond timestamps, coordinates, and polarity')
        if np.any(data[:, :3] < 0) or np.any(data[:, 1] >= width) or np.any(data[:, 2] >= height):
            raise ValueError('Negative timestamps or out-of-bounds coordinates')
        if (previous is not None and int(data[0, 0]) < previous) or np.any(data[1:, 0] < data[:-1, 0]):
            raise ValueError('Timestamps must be nondecreasing, including chunk boundaries')
        if not set(int(x) for x in np.unique(data[:, 3])).issubset(allowed):
            raise ValueError(f'Unexpected polarity for {encoding} encoding')
        first = int(data[0, 0]) if first is None else first
        previous = int(data[-1, 0])
        n += len(data)
    if not n:
        raise ValueError('Empty event stream; audit explicitly rather than silently dropping clip')
    return first, previous, n


def accumulate(data, counts, start, end, crop, encoding):
    bins, _, size, _ = counts.shape
    x0, y0, x1, y1 = crop
    inside = (data[:, 1] >= x0) & (data[:, 1] < x1) & (data[:, 2] >= y0) & (data[:, 2] < y1)
    data = data[inside]
    if not len(data):
        return
    x = np.floor((data[:, 1].astype(np.float64) - x0) * size / (x1 - x0)).astype(np.int64)
    y = np.floor((data[:, 2].astype(np.float64) - y0) * size / (y1 - y0)).astype(np.int64)
    # Subtract in float64 to avoid unsigned arithmetic; timestamps are microseconds.
    b = np.zeros(len(data), dtype=np.int64) if end == start else np.minimum(
        ((data[:, 0].astype(np.float64) - start) * bins / (end - start)).astype(np.int64), bins - 1)
    polarity = np.where(data[:, 3] == 1, 0, 1)  # channel 0 positive, 1 negative
    indices = ((b * 2 + polarity) * size + y) * size + x
    counts += np.bincount(indices, minlength=counts.size).reshape(counts.shape).astype(np.uint64)


def count_clip(row, crop, bins=8, size=224):
    width, height = dimensions(row)
    if width < 1 or height < 1 or bins < 1 or size < 1:
        raise ValueError('Dimensions, bins and size must be positive')
    if crop is None:
        crop = (0, 0, width, height)
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'Invalid fixed crop {crop} for sensor {width}x{height}')
    encoding = row.get('polarity_encoding', 'binary')
    if encoding not in {'binary', 'signed'}:
        raise ValueError('polarity_encoding must be binary or signed')
    start, end, n = scan(row['path'], width, height, encoding)
    counts = np.zeros((bins, 2, size, size), dtype=np.uint64)
    for data in chunks(row['path']):
        accumulate(data, counts, start, end, crop, encoding)
    if counts.max() > np.iinfo(np.uint32).max:
        raise OverflowError('Per-pixel counts exceed uint32')
    return counts.astype(np.uint32), {'start_us': start, 'end_us': end, 'raw_events': n,
                                     'crop': list(crop), 'width': width, 'height': height}


def represent(counts, mode='baseline', reference=None):
    """Apply scaling before log compression; no per-image renormalization."""
    counts = counts.astype(np.float32)
    totals = counts.sum(axis=(1, 2, 3), dtype=np.float64)
    scales = np.ones(len(counts), dtype=np.float32)
    if mode != 'baseline':
        if reference is None or reference <= 0:
            raise ValueError('Positive source-training reference required')
        activity = totals if mode == 'per_bin' else np.full(len(counts), totals.mean()) if mode == 'whole_clip' else None
        if activity is None:
            raise ValueError(f'Unknown representation: {mode}')
        scales = np.maximum(.25, reference / np.maximum(reference, activity)).astype(np.float32)
    counts *= scales[:, None, None, None]
    three = np.concatenate([counts, counts.sum(axis=1, keepdims=True)], axis=1)
    return np.log1p(three) / np.log(256.0), scales
