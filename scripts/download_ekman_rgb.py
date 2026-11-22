"""Download Ekman RGB videos directly from the ZIP using HTTP ranges.

Saves extracted videos without storing a second 22.9 GB archive. Each completed
video is checked against its ZIP CRC32 and size. Reruns skip verified videos.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib

URL = 'https://zenodo.org/api/records/17159328/files/videos.zip/content'
SIZE = 22925062255
CLASSES = {'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'}


def byte_range(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    response = urllib.request.urlopen(request, timeout=120)
    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
        response.close()
        raise RuntimeError('Server did not return the requested archive range')
    return response


class RemoteArchive(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position = url, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else SIZE if whence == 2 else 0)
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = SIZE - self.position
        size = min(size, SIZE - self.position)
        if size <= 0:
            return b''
        with byte_range(self.url, self.position, self.position + size - 1) as response:
            data = response.read()
        if len(data) != size:
            raise RuntimeError('Truncated archive directory read')
        self.position += size
        return data


def file_crc(path):
    crc = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xffffffff


def target_path(output, info):
    parts = PurePosixPath(info.filename).parts
    if len(parts) < 3 or parts[0] != 'videos' or parts[1] not in CLASSES or '..' in parts:
        raise ValueError(f'Unexpected archive path: {info.filename}')
    path = output.joinpath(*parts[1:])
    if not path.resolve().is_relative_to(output.resolve()):
        raise ValueError('Archive member escapes output directory')
    return path


def download_member(url, output, info, end):
    path = target_path(output, info)
    if path.exists():
        if path.stat().st_size == info.file_size and file_crc(path) == info.CRC:
            return 'verified existing', info.filename
        raise ValueError(f'Existing file is not the expected video: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    for attempt in range(1, 6):
        try:
            with byte_range(url, info.header_offset, end) as response:
                header = response.read(30)
                if len(header) != 30:
                    raise RuntimeError('Truncated local ZIP header')
                signature, version, flags, method, _, _, _, _, _, namesize, extrasize = struct.unpack('<4s5H3I2H', header)
                if signature != b'PK\x03\x04' or flags & 1 or method not in {0, 8}:
                    raise ValueError('Unsupported ZIP member encoding')
                skip = namesize + extrasize
                if len(response.read(skip)) != skip:
                    raise RuntimeError('Truncated ZIP member name')
                decoder = zlib.decompressobj(-15) if method == 8 else None
                remaining, count, crc = info.compress_size, 0, 0
                with partial.open('wb') as stream:
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError('Interrupted compressed video transfer')
                        remaining -= len(chunk)
                        data = decoder.decompress(chunk) if decoder else chunk
                        stream.write(data)
                        count += len(data)
                        crc = zlib.crc32(data, crc)
                    if decoder:
                        data = decoder.flush()
                        stream.write(data)
                        count += len(data)
                        crc = zlib.crc32(data, crc)
                        if not decoder.eof:
                            raise RuntimeError('Incomplete deflate stream')
                if count != info.file_size or crc & 0xffffffff != info.CRC:
                    raise RuntimeError('Video size/CRC32 mismatch')
                partial.replace(path)
                return 'downloaded', info.filename
        except (OSError, RuntimeError):
            if attempt == 5:
                raise
            time.sleep(attempt * 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/Ekman6')
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    with zipfile.ZipFile(RemoteArchive(URL)) as archive:
        infos = archive.infolist()
        directory_start = archive.start_dir
    order = sorted(infos, key=lambda item: item.header_offset)
    ends = {item.filename: order[i + 1].header_offset - 1 if i + 1 < len(order) else directory_start - 1
            for i, item in enumerate(order)}
    videos = [info for info in infos if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() == '.mp4'
              and len(PurePosixPath(info.filename).parts) >= 3 and PurePosixPath(info.filename).parts[1] in CLASSES]
    if not videos:
        raise RuntimeError('No six-class videos found')
    for name in sorted(CLASSES):
        subset = [x for x in videos if PurePosixPath(x.filename).parts[1] == name]
        print(f'{name}: {len(subset)} videos, {sum(x.file_size for x in subset) / 1e9:.2f} GB', flush=True)
    if args.list:
        return
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    needed = sum(info.file_size for info in videos if not target_path(output, info).exists())
    if shutil.disk_usage(output).free < needed + 1024 ** 3:
        raise RuntimeError('Insufficient space for extracted videos plus 1 GiB headroom')
    manifest = [{'archive_member': x.filename, 'path': target_path(output, x).relative_to(output).as_posix(),
                 'bytes': x.file_size, 'crc32': f'{x.CRC:08x}'} for x in videos]
    (output / 'download_manifest.json').write_text(json.dumps({'source': URL, 'videos': manifest}, indent=2), encoding='utf-8')
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_member, URL, output, info, ends[info.filename]): info for info in videos}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                status, name = future.result()
                print(f'[{i}/{len(videos)}] {status}: {name}', flush=True)
            except Exception as error:
                failures.append({'video': futures[future].filename, 'error': str(error)})
                print(f'[{i}/{len(videos)}] FAILED: {error}', flush=True)
    (output / 'download_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if failures:
        raise SystemExit(f'{len(failures)} videos failed. Rerun to retry; verified videos are skipped.')
    print(f'All {len(videos)} RGB videos downloaded and CRC32-verified.', flush=True)


if __name__ == '__main__':
    main()
