"""Download the user-supplied video emotion release; verify Zenodo MD5 hashes."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'ekman6_videoemotion8'
API = 'https://zenodo.org/api/records/17159328'


def digest(path):
    result = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def download_file(item, destination, attempts=10):
    path = destination / item['key']
    expected = item['checksum'].split(':', 1)[1]
    if path.exists():
        if path.stat().st_size == item['size'] and digest(path) == expected:
            print(f'Already verified: {path.name}', flush=True)
            return
        raise RuntimeError(f'Existing file failed verification: {path}. Move it aside before retrying.')
    partial = path.with_name(path.name + '.part')
    for attempt in range(1, attempts + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item['size']:
                raise ValueError(f'Partial file exceeds expected size: {partial}')
            if offset < item['size']:
                request = urllib.request.Request(item['links']['self'])
                if offset:
                    request.add_header('Range', f'bytes={offset}-')
                with urllib.request.urlopen(request, timeout=120) as response:
                    resumed = offset > 0 and response.status == 206
                    if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected Content-Range')
                    offset = offset if resumed else 0
                    print(f'Downloading {path.name}: {offset:,}/{item["size"]:,} bytes', flush=True)
                    last = time.monotonic()
                    with partial.open('ab' if resumed else 'wb') as output:
                        while chunk := response.read(8 * 1024 * 1024):
                            output.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last > 30:
                                print(f'{path.name}: {offset / item["size"]:.1%} ({offset:,} bytes)', flush=True)
                                last = time.monotonic()
            if partial.stat().st_size != item['size']:
                raise RuntimeError('Downloaded size does not match metadata')
            print(f'Checking MD5: {path.name}', flush=True)
            if digest(partial) != expected:
                raise ValueError(f'Checksum mismatch; inspect {partial} before retrying')
            partial.replace(path)
            print(f'Verified: {path.name}', flush=True)
            return
        except ValueError:
            raise
        except Exception as error:
            print(f'Attempt {attempt}: {error}', flush=True)
            if attempt == attempts:
                raise
            time.sleep(min(5 * attempt, 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST, help='Destination directory (default: repository data directory)')
    parser.add_argument('--files', nargs='+', help='Download only these filenames; default: all five release files')
    parser.add_argument('--list', action='store_true', help='Show files and sizes without downloading')
    parser.add_argument('--aria2', type=Path, help='Optional aria2c executable for parallel transfers')
    args = parser.parse_args()
    with urllib.request.urlopen(API, timeout=60) as response:
        record = json.load(response)
    files = record['files']
    if args.files:
        unknown = set(args.files) - {item['key'] for item in files}
        if unknown:
            parser.error(f'Unknown filenames: {sorted(unknown)}')
        files = [item for item in files if item['key'] in args.files]
    for item in files:
        if Path(item['key']).name != item['key'] or '/' in item['key'] or '\\' in item['key']:
            raise ValueError('Unexpected filename in release metadata')
        if not item['checksum'].startswith('md5:'):
            raise ValueError('Unexpected checksum algorithm')
        print(f'{item["key"]}: {item["size"]:,} bytes', flush=True)
    print(f'Total selected: {sum(item["size"] for item in files) / 1e9:.2f} GB', flush=True)
    if args.list:
        return
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Conservative check: allow room to restart a partial transfer if Range is ignored.
    needed = sum(item['size'] for item in files if not (destination / item['key']).exists())
    free = shutil.disk_usage(destination).free
    if free < needed + 1024 ** 3:
        raise RuntimeError(f'Insufficient free space: need {needed / 1e9:.2f} GB plus 1 GiB headroom; have {free / 1e9:.2f} GB')
    (destination / 'zenodo_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if args.aria2:
        entries = []
        for item in files:
            path = destination / item['key']
            expected = item['checksum'].split(':', 1)[1]
            if path.exists():
                if path.stat().st_size != item['size'] or digest(path) != expected:
                    raise RuntimeError(f'Existing file failed verification: {path}')
                continue
            entries.append('\n'.join([
                item['links']['self'], f'  dir={destination}',
                f'  out={item["key"]}.part', f'  checksum={item["checksum"].replace(":", "=", 1)}',
            ]))
        if entries:
            queue = destination / 'aria2_downloads.txt'
            queue.write_text('\n'.join(entries) + '\n', encoding='utf-8')
            subprocess.run([
                str(args.aria2.resolve()), f'--input-file={queue}',
                '--continue=true', '--max-concurrent-downloads=2',
                '--max-connection-per-server=8', '--split=8', '--min-split-size=8M',
                '--file-allocation=none', '--auto-file-renaming=false',
                '--allow-overwrite=true', '--check-integrity=true',
                '--max-tries=10', '--retry-wait=10', '--summary-interval=30',
                '--console-log-level=warn', '--enable-color=false',
            ], check=True)
    # Small files first, so metadata is available while the videos download.
    for item in sorted(files, key=lambda entry: entry['size']):
        download_file(item, destination)
    print('All files downloaded and verified.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to resume.', file=sys.stderr)
        sys.exit(130)
