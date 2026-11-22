import csv
import hashlib
import json
from pathlib import Path

CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
SPLITS = {'train', 'validation', 'test', 'target_test', 'target_reserve'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty manifest/index: {path}')
    seen = set()
    for row in rows:
        key = (row['dataset'], row['clip_id'])
        if key in seen:
            raise ValueError(f'Duplicate clip identity: {key}')
        seen.add(key)
        if row['label'] not in CLASSES or int(row['label_id']) != CLASSES.index(row['label']):
            raise ValueError(f'Class mapping mismatch: {key}')
        if row['split'] not in SPLITS:
            raise ValueError(f'Unknown split: {row["split"]}')
        if row['dataset'].lower() == 'nefer' and row['split'] not in {'target_test', 'target_reserve'}:
            raise ValueError('NEFER cannot enter source training or source validation')
    return rows


def verify_lock(path):
    path = Path(path)
    lock = path.parent / 'checksums.json'
    if not lock.is_file():
        raise ValueError(f'Missing frozen checksums: {lock}')
    checksums = json.loads(lock.read_text(encoding='utf-8'))
    if path.name not in checksums:
        raise ValueError(f'{path.name} not covered by lock')
    for name, expected in checksums.items():
        candidate = (path.parent / name).resolve()
        if not candidate.is_relative_to(path.parent.resolve()) or digest(candidate) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')


def lock_directory(directory, names):
    json_write(Path(directory) / 'checksums.json', {name: digest(Path(directory) / name) for name in names})


def uid(row):
    return hashlib.sha256((row['dataset'] + '\0' + row['clip_id']).encode()).hexdigest()[:24]


def fresh_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path
