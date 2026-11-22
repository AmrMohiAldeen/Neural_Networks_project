#!/usr/bin/env python3
"""Run independent v2e conversions concurrently, with resumable success records.

Keep the pinned upstream converter unchanged. Existing H5 files without this
launcher's matching success record are regenerated (they may be partial).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tqdm import tqdm


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=positive_int, default=2)
    parser.add_argument("--threads-per-worker", type=positive_int, default=1)
    parser.add_argument("--width", type=positive_int, default=336)
    parser.add_argument("--height", type=positive_int, default=224)
    parser.add_argument("--limit", type=positive_int, help="Convert only the first N sorted videos for timing")
    parser.add_argument("--include-list", type=Path,
                        help="Text file of input-relative MP4 paths (one per line); convert only these")
    return parser.parse_known_args()


def main():
    args, extra = parse_args()
    input_root = args.input_root.absolute()
    output_root = args.output_root.absolute()
    v2e_py = args.v2e_root.absolute() / "v2e.py"
    if not input_root.is_dir() or not v2e_py.is_file():
        raise SystemExit("Input directory or v2e.py does not exist")
    # These are owned by the launcher: allowing overrides would break resume.
    reserved = {"-i", "--input", "-o", "--output_folder", "--output_width",
                "--output_height", "--dvs_h5", "--unique_output_folder",
                "--output_in_place"}
    if any(item.split("=", 1)[0] in reserved for item in extra):
        raise SystemExit("Do not override input, output, dimensions, or H5 paths in v2e arguments")
    videos = sorted(p for p in input_root.rglob("*") if p.suffix.lower() == ".mp4")
    if args.include_list:
        wanted = {line.strip() for line in args.include_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        videos = [p for p in videos if p.relative_to(input_root).as_posix() in wanted]
        absent = wanted - {p.relative_to(input_root).as_posix() for p in videos}
        if absent:
            raise SystemExit(f"{len(absent)} listed videos are not in the input root, e.g. {sorted(absent)[0]}")
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit("No .mp4 videos found")
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads_per_worker)
    env.setdefault("MPLBACKEND", "Agg")
    stopped = threading.Event()
    processes = set()
    process_lock = threading.Lock()

    def convert(video):
        if stopped.is_set():
            return "cancelled"
        output = output_root / video.relative_to(input_root).with_suffix("")
        output.mkdir(parents=True, exist_ok=True)
        h5 = output / "events.h5"
        marker = output / ".conversion-success.json"
        command = [args.python, str(v2e_py), "-i", str(video), "-o", str(output),
                   "--overwrite", "--no_preview", "--skip_video_output", "--disable_slomo",
                   "--output_width", str(args.width), "--output_height", str(args.height),
                   "--dvs_h5", "events.h5", *extra]
        stat = video.stat()
        signature = {"command": command, "input_size": stat.st_size,
                     "input_mtime_ns": stat.st_mtime_ns}
        if marker.exists() and h5.exists():
            try:
                saved = json.loads(marker.read_text(encoding="utf-8"))
                if saved["signature"] == signature and saved["h5_size"] == h5.stat().st_size:
                    return "skipped"
            except (ValueError, KeyError):
                pass
        marker.unlink(missing_ok=True)
        started = time.monotonic()
        # Stream output to disk rather than retaining logs from every worker in RAM.
        with (output / "conversion.stdout.log").open("w", encoding="utf-8") as stdout, \
                (output / "conversion.stderr.log").open("w", encoding="utf-8") as stderr:
            with process_lock:
                if stopped.is_set():
                    return "cancelled"
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
                processes.add(process)
            try:
                code = process.wait()
            finally:
                with process_lock:
                    processes.discard(process)
        if code != 0 or not h5.exists() or h5.stat().st_size == 0:
            tqdm.write(f"FAILED: {video}; see {output / 'conversion.stderr.log'}")
            return "failed"
        record = {"signature": signature, "h5_size": h5.stat().st_size,
                  "seconds": time.monotonic() - started, "returncode": code}
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(marker)
        return "success"

    print(f"Videos={len(videos)} Workers={args.workers} Threads/worker={args.threads_per_worker}", flush=True)
    print("Existing outputs without matching success records will be regenerated. Stop other converters first.", flush=True)
    counts = dict(success=0, skipped=0, failed=0, cancelled=0)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, video): video for video in videos}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="v2e", unit="video"):
                try:
                    counts[future.result()] += 1
                except Exception as error:
                    counts["failed"] += 1
                    tqdm.write(f"FAILED: {futures[future]}: {error}")
        except KeyboardInterrupt:
            stopped.set()
            for future in futures:
                future.cancel()
            with process_lock:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
            print("Stopped. Rerun the same command to resume completed videos.", flush=True)
            raise SystemExit(130)
    print(f"Counts={counts} Wall_seconds={time.monotonic() - started:.1f}", flush=True)
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
