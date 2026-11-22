"""Inventory datasets and create reproducible, clip-level metadata splits.

No training or preprocessing statistics are fitted here. Native event decoding
and timestamp/polarity checks belong to the format-specific preprocessing audit.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import zipfile

CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
TARGET_USERS = {'03', '07', '17', '19', '27', '28'}
TRAIN_USERS = {'01', '02', '04', '05', '06', '08', '09', '10', '11', '12',
               '13', '14', '15', '16', '21', '22', '23', '24', '25', '26'}
VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpeg', '.mpg'}
EVENT = {'.raw', '.dat', '.aedat', '.aedat4', '.h5', '.hdf5', '.npz', '.npy'}
FIELDS = ['dataset', 'modality', 'clip_id', 'path', 'label', 'label_id',
          'participant_id', 'original_video_id', 'split', 'sha256', 'expected_label']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def label(value):
    value = str(value).strip().lower()
    return 'happiness' if value == 'joy' else value


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inventory(roots, output):
    result = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f'Dataset directory not found: {root}')
        counts, total, samples, metadata, archives = Counter(), 0, defaultdict(list), [], []
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            ext = path.suffix.lower() or '<no suffix>'
            counts[ext] += 1
            total += path.stat().st_size
            if len(samples[ext]) < 8:
                samples[ext].append(path.relative_to(root).as_posix())
            if ext in {'.csv', '.json', '.txt', '.yaml', '.yml'}:
                entry = {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size}
                if ext in {'.csv', '.txt'}:
                    with path.open(encoding='utf-8', errors='replace') as f:
                        entry['preview'] = [f.readline()[:500].rstrip() for _ in range(3)]
                metadata.append(entry)
            if ext == '.zip':
                try:
                    with zipfile.ZipFile(path) as z:
                        members = z.infolist()
                        archives.append({'path': str(path), 'members': len(members),
                                         'uncompressed_bytes': sum(x.file_size for x in members),
                                         'sample': [x.filename for x in members[:15]]})
                except zipfile.BadZipFile:
                    archives.append({'path': str(path), 'error': 'Invalid or incomplete ZIP'})
        result.append({'root': str(root), 'bytes': total, 'extensions': dict(counts),
                       'samples': dict(samples), 'metadata': metadata, 'archives': archives})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Inventory saved: {output}')
    for item in result:
        print(item['root'], json.dumps(item['extensions']))


def file_index(root, extensions):
    index = defaultdict(list)
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in extensions:
            index[path.stem].append(path.resolve())
    return index


def resolve_file(root, value, index):
    # Metadata from another OS can contain Windows separators.
    value = str(value).replace('\\', '/')
    path = Path(value)
    direct = path if path.is_absolute() else root / path
    if direct.is_file() and direct.resolve().is_relative_to(root):
        return direct.resolve()
    matches = index.get(path.stem, [])
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f'Missing or ambiguous recording: {value}; found {len(matches)} stem matches')


def source_rows(args):
    root = args.source_root.expanduser().resolve()
    extensions = VIDEO if args.source_kind == 'ekman_rgb' else EVENT
    index = file_index(root, extensions)
    if args.source_csv:
        with args.source_csv.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            required = {'path', 'label', 'original_video_id'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'Source CSV requires columns {sorted(required)}')
            records = list(reader)
    else:
        if not args.trust_pickle:
            raise ValueError('Use --trust-pickle only for the trusted publisher file; pickle executes code.')
        import pickle
        with args.ekman_pickle.open('rb') as f:
            data = pickle.load(f)
        import numpy as np
        samples = data['samples']
        if isinstance(samples, dict):
            samples = list(samples.items())
        elif isinstance(samples, list):
            samples = [(str(i), sample) for i, sample in enumerate(samples)]
        else:
            raise ValueError('Unsupported pickle samples layout')
        records = []
        for key, sample in samples:
            if 'label' not in sample or 'video_path' not in sample:
                raise ValueError('Unsupported pickle sample layout: need video_path and label')
            vector = np.asarray(sample['label']).reshape(-1)
            if vector.size != 6 or not np.all((vector == 0) | (vector == 1)) or vector.sum() != 1:
                raise ValueError(f'Invalid one-hot label for {key}')
            idx = int(vector.argmax())
            mapping = data['idx_to_label']
            mapped = mapping[idx] if idx in mapping else mapping[str(idx)]
            video = sample['video_path']
            records.append({'path': video, 'label': mapped,
                            'original_video_id': Path(str(video).replace('\\', '/')).stem})
    rows, exclusions = [], []
    for record in records:
        name = label(record['label'])
        if name not in CLASSES:
            raise ValueError(f'Unexpected source label {name!r}; review label mapping')
        original = record['original_video_id'].strip()
        if not original:
            raise ValueError('Every source clip requires an original_video_id')
        path = resolve_file(root, record['path'], index)
        if path.suffix.lower() not in extensions:
            raise ValueError(f'Wrong source modality: {path}')
        split = record.get('split', '').strip().lower()
        split = {'val': 'validation', 'valid': 'validation'}.get(split, split)
        if split not in {'', 'train', 'validation', 'test'}:
            raise ValueError(f'Unknown source split {split}')
        rows.append({'dataset': args.source_kind, 'modality': 'rgb' if args.source_kind == 'ekman_rgb' else 'events',
                     'clip_id': path.relative_to(root).as_posix(), 'path': str(path), 'label': name,
                     'label_id': CLASSES.index(name), 'participant_id': record.get('participant_id', '').strip(),
                     'original_video_id': original, 'split': split, 'sha256': sha256(path), 'expected_label': ''})
    return rows, exclusions


def split_source(rows, seed):
    # Union connected originals, participants, and byte-identical recordings.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for field in ('original_video_id', 'participant_id', 'sha256'):
            key = (field, row[field])
            if not row[field]:
                continue
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    assigned = [bool(row['split']) for row in rows]
    if any(assigned) and not all(assigned):
        raise ValueError('Source official split assignments must be provided for every clip or none')
    if all(assigned):
        for group in groups.values():
            if len({r['split'] for r in group}) != 1:
                raise ValueError('Official source splits leak an original, participant, or identical recording')
        return 'provided source splits (audited)'
    ordered = sorted(groups.values(), key=lambda g: min(r['clip_id'] for r in g))
    if len(ordered) < 3:
        raise ValueError('Need at least three independent source groups')
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    nval, ntest = max(1, round(n * .1)), max(1, round(n * .2))
    ntrain = n - nval - ntest
    for i, group in enumerate(ordered):
        split = 'train' if i < ntrain else 'validation' if i < ntrain + nval else 'test'
        for row in group:
            row['split'] = split
    return 'seeded grouped 70/10/20 by group count; clip proportions may differ'


def annotation_clips(path):
    clips = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        for number, row in enumerate(csv.reader(f), 1):
            if len(row) != 4:
                raise ValueError(f'{path}:{number}: expected official headerless four-column CSV')
            user, clip, frame, emotion = (x.strip() for x in row)
            match = re.fullmatch(r'user_(\d{2})', user)
            if not match:
                raise ValueError(f'Unrecognized NEFER participant {user}')
            key = (match[1], clip)
            emotion = label(emotion)
            if key in clips and clips[key] != emotion:
                raise ValueError(f'Conflicting frame labels within NEFER clip {key}')
            clips[key] = emotion
    return clips


def nefer_rows(args):
    root = args.nefer_root.expanduser().resolve()
    index = file_index(root, EVENT)
    rows, exclusions, seen = [], [], set()
    for partition, users, split in [('train', TRAIN_USERS, 'target_reserve'),
                                     ('validation', TARGET_USERS, 'target_test')]:
        annotations = args.annotations_root / 'reported_emotions' / f'{partition}_event_subjective.csv'
        clips = annotation_clips(annotations)
        expected_path = args.annotations_root / 'expected_emotions' / f'{partition}_event.csv'
        expected = annotation_clips(expected_path) if expected_path.exists() else {}
        observed = {user for user, clip in clips}
        if observed != users:
            raise ValueError(f'{partition} annotation participant set differs from official split: {sorted(observed)}')
        for (user, clip), emotion in sorted(clips.items()):
            if (user, clip) in seen:
                raise ValueError('NEFER clip appears in multiple official partitions')
            seen.add((user, clip))
            if emotion in {'none', 'contempt'}:
                exclusions.append({'dataset': 'nefer', 'clip_id': clip, 'reason': f'reported label: {emotion}'})
                continue
            if emotion not in CLASSES:
                raise ValueError(f'Unknown NEFER label {emotion!r}')
            # Annotation IDs can end in "_cd" while raw filenames omit it.
            # Preserve the annotation clip ID and resolve only its file path.
            candidate_stems = {clip}
            if clip.endswith('_cd'):
                candidate_stems.add(clip[:-3])
            matches = sorted({
                path
                for stem in candidate_stems
                for path in index.get(stem, [])
                if path.stem.startswith(f'user{user}_')
            })
            # Verified against the user's listing of the published archive:
            # this recording has a .bias file but no corresponding .raw file.
            if (
                not matches
                and clip == 'user22_2022-06-17_12-30-32_cd'
                and user == '22'
                and split == 'target_reserve'
            ):
                exclusions.append({
                    'dataset': 'nefer',
                    'clip_id': clip,
                    'reason': (
                        'Raw recording absent from published event_raw.zip; '
                        'only matching .bias file exists. '
                        'Excluded from optional real-data reserve.'
                    ),
                })
                continue
            if len(matches) != 1:
                raise ValueError(f'NEFER raw clip {clip}: found {len(matches)} matches. Extract raw events or inspect inventory; TBR frames cannot replace polarity counts.')
            path = matches[0]
            rows.append({'dataset': 'nefer', 'modality': 'events', 'clip_id': clip,
                         'path': str(path), 'label': emotion, 'label_id': CLASSES.index(emotion),
                         'participant_id': user, 'original_video_id': clip, 'split': split,
                         'sha256': sha256(path), 'expected_label': expected.get((user, clip), '')})
    return rows, exclusions


def build(args):
    if args.output.exists():
        raise ValueError('Output directory already exists. Use a new version directory; frozen manifests are never overwritten.')
    for root in (args.source_root, args.nefer_root, args.annotations_root):
        if not root.is_dir():
            raise ValueError(f'Directory not found: {root}')
    if args.ekman_pickle and args.source_kind != 'ekman_rgb':
        raise ValueError('Ekman feature labels describe RGB videos, not released Emo-DVS events')
    source, excluded = source_rows(args)
    if not source:
        raise ValueError('No source clips')
    protocol = split_source(source, args.seed)
    target, removed = nefer_rows(args)
    excluded += removed
    rows = source + target
    seen_paths, by_hash = set(), defaultdict(list)
    for row in rows:
        if row['path'] in seen_paths:
            raise ValueError(f'Duplicate manifest path: {row["path"]}')
        seen_paths.add(row['path'])
        if Path(row['path']).stat().st_size == 0:
            raise ValueError(f'Empty recording: {row["path"]}')
        by_hash[row['sha256']].append(row)
    duplicates = []
    for digest, group in by_hash.items():
        if len({r['split'] for r in group}) > 1:
            raise ValueError(f'Identical bytes across splits: {[r["clip_id"] for r in group]}')
        if len(group) > 1:
            duplicates.append([r['clip_id'] for r in group])
    warnings = ['Native event timestamps, coordinate bounds, polarity conventions, and crops still require format-specific validation.',
                'SHA256 checks byte-identical files; re-encoded videos and unknown cross-dataset identities may remain undetected.',
                'Dataset permissions and release versions must be recorded by the team.']
    if any(not r['participant_id'] for r in source):
        warnings.append('Source participant IDs are incomplete: original-video grouping does not establish identity independence.')
    if args.source_kind == 'ekman_rgb':
        warnings.append('Source is Ekman-6 RGB, not Emo-DVS. Simulation and a documented protocol amendment are required before event-model training.')
    counts = Counter((r['dataset'], r['split'], r['label']) for r in rows)
    for split in ('train', 'validation', 'test'):
        missing = [c for c in CLASSES if not counts[(args.source_kind, split, c)]]
        if missing:
            raise ValueError(f'Source {split} missing classes {missing}; review grouping/protocol before freezing')
    for c in CLASSES:
        if not counts[('nefer', 'target_test', c)]:
            warnings.append(f'NEFER target_test has no {c} clips; report fixed-six-class metric handling.')
    args.output.mkdir(parents=True)
    rows.sort(key=lambda r: (r['dataset'], r['split'], r['clip_id']))
    write_csv(args.output / 'manifest.csv', rows, FIELDS)
    write_csv(args.output / 'exclusions.csv', excluded, ['dataset', 'clip_id', 'reason'])
    write_csv(args.output / 'class_counts.csv',
              [{'dataset': d, 'split': s, 'label': c, 'clips': n} for (d, s, c), n in sorted(counts.items())],
              ['dataset', 'split', 'label', 'clips'])
    participants = Counter((r['dataset'], r['split'], r['participant_id'], r['label']) for r in rows)
    write_csv(args.output / 'participant_counts.csv',
              [{'dataset': d, 'split': s, 'participant_id': u, 'label': c, 'clips': n}
               for (d, s, u, c), n in sorted(participants.items())],
              ['dataset', 'split', 'participant_id', 'label', 'clips'])
    report = {'status': 'metadata_splits_frozen_event_integrity_pending', 'seed': args.seed,
              'source_protocol': protocol, 'class_to_idx': {c: i for i, c in enumerate(CLASSES)},
              'target_test_participants': sorted(TARGET_USERS), 'same_split_identical_files': duplicates,
              'warnings': warnings, 'clip_counts': dict(Counter(r['split'] for r in rows)),
              'normalization_policy': 'Fit all statistics and class weights on source train only; no global feature stats.',
              'source_csv_sha256': sha256(args.source_csv) if args.source_csv else None,
              'source_pickle_sha256': sha256(args.ekman_pickle) if args.ekman_pickle else None,
              'annotation_sha256': {p.relative_to(args.annotations_root).as_posix(): sha256(p)
                                    for p in sorted(args.annotations_root.rglob('*.csv'))},
              'command': sys.argv, 'script_sha256': sha256(Path(__file__))}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lock = {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / 'checksums.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('inventory', help='Inspect files and archives without loading pickle data')
    inv.add_argument('--roots', nargs='+', type=Path, required=True)
    inv.add_argument('--output', type=Path, default=Path('artifacts/data_inventory.json'))
    cmd = commands.add_parser('build', help='Audit and freeze metadata splits (requires extracted recordings)')
    cmd.add_argument('--source-root', type=Path, required=True)
    cmd.add_argument('--source-kind', choices=['ekman_rgb', 'emo_events'], required=True)
    inputs = cmd.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source-csv', type=Path)
    inputs.add_argument('--ekman-pickle', type=Path)
    cmd.add_argument('--trust-pickle', action='store_true')
    cmd.add_argument('--nefer-root', type=Path, required=True)
    cmd.add_argument('--annotations-root', type=Path, required=True)
    cmd.add_argument('--seed', type=int, default=476)
    cmd.add_argument('--output', type=Path, default=Path('artifacts/splits_v1'))
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args.roots, args.output)
    else:
        build(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f'Audit stopped: {error}')
