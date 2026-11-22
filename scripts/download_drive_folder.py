"""Download the supplied Google Drive folder recursively using gdown."""
import argparse
import sys
import time
from pathlib import Path

FOLDER_URL = 'https://drive.google.com/drive/folders/1OAnM8B88XX_TqJj2XF61w1nA-nWNV4YC'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'data' / 'google_drive_dataset'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=FOLDER_URL, help='Google Drive folder URL')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Destination directory')
    parser.add_argument('--attempts', type=int, default=3, help='Total attempts (default: 3)')
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error('--attempts must be at least 1')
    try:
        import gdown
    except ImportError:
        parser.exit(1, 'Install the dependency first: python -m pip install "gdown>=5.2,<6"\n')

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f'Folder: {args.url}\nDestination: {output}', flush=True)
    for attempt in range(1, args.attempts + 1):
        try:
            files = gdown.download_folder(
                url=args.url,
                output=str(output),
                quiet=False,
                use_cookies=False,
                resume=True,
            )
            if files is None:
                raise RuntimeError('Google Drive did not return a successful folder download.')
            print(f'Download finished: {len(files)} files processed in {output}', flush=True)
            return 0
        except Exception as error:
            print(f'Attempt {attempt}/{args.attempts} failed: {error}', file=sys.stderr, flush=True)
            if attempt < args.attempts:
                time.sleep(10)
    print(
        'Download incomplete. Check folder sharing permissions, available storage, '
        'and Google Drive download quotas. Run the same command again to resume. '
        'Existing completed files are skipped, not checksum-verified.',
        file=sys.stderr,
    )
    return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nStopped. Run the same command again to resume.', file=sys.stderr)
        sys.exit(130)
