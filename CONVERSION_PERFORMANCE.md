> **Superseded (2026-11-23):** steps 6 onward are replaced by [SUBSET300_GUIDE.md](SUBSET300_GUIDE.md), which uses a 300-clip source subset and runs to the end. Kept for reference only.

# Parallel Ekman conversion

The pinned upstream batch wrapper launches one v2e process at a time. v2e
automatically selects CUDA when available. With `--disable_slomo`, small GPU
operations are interleaved with video decoding, CPU work, transfers, and H5
writes. Low GPU utilization alone does not identify which stage limits speed.

`scripts/convert_events_parallel.py` runs independent videos concurrently without
changing the pinned upstream files or simulation settings. Start with two
workers; benchmark four only if CPU, RAM, storage, and GPU capacity permit it.
There is no measured speedup on the lab machine yet. Each worker limits common
CPU math libraries to one thread by default to reduce contention.

Upload from laptop PowerShell:

```powershell
scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\convert_events_parallel.py" nndl1@AI-Lab-12:/home/nndl1/scripts/
```

In the existing Screen session on Linux, stop the old command with Ctrl+C.
Confirm that its wrapper and v2e child have exited before starting the replacement:

```bash
pgrep -af 'convert_videos_to_events.py|convert_events_parallel.py|[v]2e.py'
```

Do not run two launchers into the same output tree. Start the replacement:

```bash
cd /home/nndl1
conda activate ekman-events
mkdir -p logs data/conversion_tmp
export MPLBACKEND=Agg
export TMPDIR=/home/nndl1/data/conversion_tmp
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u scripts/convert_events_parallel.py \
  --input-root data/conversion_input/Ekman6_v2 \
  --output-root data/events/Ekman6_v2 \
  --v2e-root third_party/v2e \
  --workers 2 --threads-per-worker 1 \
  --dvs_aedat2 None --dvs_emulator_seed 476 \
  2>&1 | tee -a logs/ekman_v2_parallel.log
```

The launcher writes stdout/stderr logs and an atomic `.conversion-success.json`
record in each clip directory. It skips clips only when the record matches the
input path/size/mtime, conversion command, and output H5 size. This is completion
evidence, not a full integrity check. Retain the event validation step in
`POST_AUDIT_GUIDE.md` and require all 1,632 outputs.

**The initial switch regenerates existing clips once**, including the first 11
completed by the old wrapper: those lack per-clip completion records, and an
interrupted H5 may look readable. Subsequent runs of this launcher resume clips
with matching success records. Do not use `--save-error-output`; this launcher
always keeps per-clip logs.

For a comparison, use `--limit 8` with separate output roots such as
`data/events_pilot/timing_w1`, `timing_w2`, and `timing_w4`, and workers 1, 2,
and 4 respectively. Keep all other arguments identical. Compare total wall time
for those same eight clips, not the first few progress-bar estimates. Do not
benchmark concurrently with the production converter. Watch `nvidia-smi -l 2`,
`free -h`, and available disk space. More workers can increase contention and
slow the job, so keep the best measured setting. Do not change resolution,
simulation thresholds, or SuperSloMo settings just to improve utilization.
