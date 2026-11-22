"""Create a new frozen manifest after deterministic source duplicate cleanup.

Uses Python's standard library only. Never edits the input directory or deletes
recordings. Conflicting-label source hash groups are excluded entirely; for a
same-label source group, retain the shortest basename, then alphabetical clip ID.
All retained rows keep their original split assignments, and NEFER is untouched.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
REQUIRED = {'manifest.csv', 'audit.json', 'exclusions.csv', 'class_counts.csv', 'participant_counts.csv'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def verify_checksums(directory):
    directory = Path(directory).resolve()
    lock = json.loads((directory / 'checksums.json').read_text(encoding='utf-8'))
    if not REQUIRED.issubset(lock):
        raise ValueError(f'Checksum lock must cover: {sorted(REQUIRED)}')
    for name, expected in lock.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Frozen file is missing or changed: {name}')
    return lock


def validate_rows(rows, source_dataset):
    if not rows:
        raise ValueError('Empty manifest')
    identities, paths, hashes, groups = set(), set(), defaultdict(set), defaultdict(set)
    for row in rows:
        key = (row['dataset'], row['clip_id'])
        if key in identities or row['path'] in paths:
            raise ValueError(f'Duplicate clip ID or recording path: {key}')
        identities.add(key)
        paths.add(row['path'])
        if row['label'] not in CLASSES or int(row['label_id']) != CLASSES.index(row['label']):
            raise ValueError(f'Unexpected class mapping: {key}')
        digest = row['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Invalid recording SHA256: {key}')
        hashes[digest].add(row['split'])
        if row['dataset'] == source_dataset:
            if row['split'] not in {'train', 'validation', 'test'}:
                raise ValueError('Source clip has a non-source split')
            for field in ('original_video_id', 'participant_id'):
                if row.get(field):
                    groups[(field, row[field])].add(row['split'])
        elif row['dataset'] == 'nefer':
            if row['split'] not in {'target_test', 'target_reserve'}:
                raise ValueError('NEFER cannot enter source training/validation/test')
        else:
            raise ValueError(f'Unexpected dataset: {row["dataset"]}')
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError('Identical recordings appear across splits')
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError('Known source video/participant group appears across splits')
    for split in ('train', 'validation', 'test'):
        present = {r['label'] for r in rows if r['dataset'] == source_dataset and r['split'] == split}
        if present != set(CLASSES):
            raise ValueError(f'Source {split} has missing classes: {sorted(set(CLASSES) - present)}')


def count_tables(rows):
    counts = Counter((r['dataset'], r['split'], r['label']) for r in rows)
    classes = [{'dataset': d, 'split': s, 'label': c, 'clips': n}
               for (d, s, c), n in sorted(counts.items())]
    counts = Counter((r['dataset'], r['split'], r.get('participant_id', ''), r['label']) for r in rows)
    participants = [{'dataset': d, 'split': s, 'participant_id': u, 'label': c, 'clips': n}
                    for (d, s, u, c), n in sorted(counts.items())]
    return classes, participants


def duplicate_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sha256']].append(row['clip_id'])
    return [sorted(group) for group in groups.values() if len(group) > 1]


def verify(directory, source_dataset='ekman_rgb'):
    directory = Path(directory).resolve()
    verify_checksums(directory)
    rows, _ = read_csv(directory / 'manifest.csv')
    validate_rows(rows, source_dataset)
    audit = json.loads((directory / 'audit.json').read_text(encoding='utf-8'))
    expected_counts = dict(Counter(r['split'] for r in rows))
    if audit['clip_counts'] != expected_counts:
        raise ValueError('Audit clip counts disagree with manifest')
    expected_classes, expected_participants = count_tables(rows)
    for name, expected in [('class_counts.csv', expected_classes), ('participant_counts.csv', expected_participants)]:
        actual, _ = read_csv(directory / name)
        normalized = [{k: str(v) for k, v in row.items()} for row in expected]
        if actual != normalized:
            raise ValueError(f'Count table disagrees with manifest: {name}')
    return {'verified_directory': str(directory), 'clip_counts': expected_counts,
            'source_duplicates_remaining': duplicate_groups([r for r in rows if r['dataset'] == source_dataset]),
            'event_integrity': 'pending separate format-specific validation'}


def clean(args):
    original, output = args.input.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(original):
        raise ValueError('Choose a new output directory outside the original frozen directory')
    verify(original, args.source_dataset)
    rows, fields = read_csv(original / 'manifest.csv')
    old_exclusions, _ = read_csv(original / 'exclusions.csv')
    old_audit = json.loads((original / 'audit.json').read_text(encoding='utf-8'))
    if args.verify_recordings:
        for i, row in enumerate(rows, 1):
            if sha256(row['path']) != row['sha256']:
                raise ValueError(f'Recording changed since original audit: {row["path"]}')
            if i % 100 == 0:
                print(f'Verified recording hashes: {i}/{len(rows)}', flush=True)
    groups = defaultdict(list)
    for row in rows:
        if row['dataset'] == args.source_dataset:
            groups[row['sha256']].append(row)
    removed, decisions = set(), []
    for digest, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        labels = sorted({r['label'] for r in group})
        if len(labels) > 1:
            discard = group
            retained = ''
            reason = 'Identical source recording has conflicting labels: ' + ', '.join(labels)
        else:
            retained_row = min(group, key=lambda r: (len(PurePosixPath(r['clip_id']).name), r['clip_id']))
            discard = [r for r in group if r is not retained_row]
            retained = retained_row['clip_id']
            reason = 'Same-label identical source recording; retained shortest basename then alphabetical clip ID: ' + retained
        for row in discard:
            removed.add((row['dataset'], row['clip_id']))
            decisions.append({'dataset': row['dataset'], 'clip_id': row['clip_id'],
                              'label': row['label'], 'split': row['split'], 'sha256': digest,
                              'retained_clip_id': retained, 'reason': reason})
    kept = [r for r in rows if (r['dataset'], r['clip_id']) not in removed]
    validate_rows(kept, args.source_dataset)
    before_nefer = [r for r in rows if r['dataset'] == 'nefer']
    after_nefer = [r for r in kept if r['dataset'] == 'nefer']
    if before_nefer != after_nefer:
        raise ValueError('NEFER rows changed')
    if duplicate_groups([r for r in kept if r['dataset'] == args.source_dataset]):
        raise ValueError('Source duplicates remain after cleanup')
    counts = Counter(r['split'] for r in kept if r['dataset'] == args.source_dataset)
    if args.expect_source_counts and list(args.expect_source_counts) != [counts[s] for s in ('train', 'validation', 'test')]:
        raise ValueError(f'Unexpected source counts: {dict(counts)}; output was not written')
    exclusions = old_exclusions + [{k: row[k] for k in ('dataset', 'clip_id', 'reason')} for row in decisions]
    classes, participants = count_tables(kept)
    policy = {'source_dataset': args.source_dataset,
              'conflicting_labels': 'exclude every row in the source SHA256 group',
              'same_label_duplicates': 'retain shortest basename, then alphabetical clip ID',
              'split_assignments': 'preserved for every retained row; no resplitting',
              'nefer_rows': 'unchanged', 'recordings_deleted': False,
              'parent_directory': str(original), 'parent_manifest_sha256': sha256(original / 'manifest.csv'),
              'parent_checksums_sha256': sha256(original / 'checksums.json'),
              'parent_audit_sha256': sha256(original / 'audit.json'),
              'recording_hashes_rechecked': args.verify_recordings,
              'removed_rows': len(decisions), 'decisions': decisions}
    report = dict(old_audit)
    report.update(status='metadata_splits_frozen_event_integrity_pending',
                  clip_counts=dict(Counter(r['split'] for r in kept)),
                  same_split_identical_files=duplicate_groups(kept),
                  source_protocol=old_audit['source_protocol'] + '; duplicate cleanup without resplitting',
                  parent_script_sha256=old_audit.get('script_sha256'),
                  command=sys.argv, script_sha256=sha256(Path(__file__)),
                  duplicate_cleanup={'policy_file': 'cleanup.json', 'removed_rows': len(decisions),
                                     'parent_directory': str(original)})
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'manifest.csv', kept, fields)
    write_csv(output / 'exclusions.csv', exclusions, ['dataset', 'clip_id', 'reason'])
    write_csv(output / 'class_counts.csv', classes, ['dataset', 'split', 'label', 'clips'])
    write_csv(output / 'participant_counts.csv', participants, ['dataset', 'split', 'participant_id', 'label', 'clips'])
    write_json(output / 'cleanup.json', policy)
    write_json(output / 'audit.json', report)
    write_json(output / 'checksums.json', {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    result = verify(output, args.source_dataset)
    result['removed_rows'] = len(decisions)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cmd = sub.add_parser('clean')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    cmd.add_argument('--source-dataset', default='ekman_rgb')
    cmd.add_argument('--expect-source-counts', nargs=3, type=int, metavar=('TRAIN', 'VALIDATION', 'TEST'))
    cmd.add_argument('--verify-recordings', action='store_true', help='Rehash every original recording before cleanup')
    check = sub.add_parser('verify')
    check.add_argument('--directory', type=Path, required=True)
    check.add_argument('--source-dataset', default='ekman_rgb')
    args = parser.parse_args()
    if args.command == 'clean':
        clean(args)
    else:
        print(json.dumps(verify(args.directory, args.source_dataset), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f'Cleanup stopped: {error}')
