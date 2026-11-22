> **Superseded (2026-11-23):** steps 6 onward are replaced by [SUBSET300_GUIDE.md](SUBSET300_GUIDE.md), which uses a 300-clip source subset and runs to the end. Kept for reference only.

# What to do after the verified splits_v2 audit

This is the current guide for **nndl1 on AI-Lab-12**. All Linux commands run from
`/home/nndl1`, not `~/coe476-project`.

## Current verified state

Your `artifacts/splits_v2` verification succeeded:

| Partition | Clips |
|---|---:|
| Source training | 1,140 |
| Source validation | 159 |
| Source test | 333 |
| NEFER optional reserve | 347 |
| NEFER target test | 97 |

All six classes are represented in every partition. Source byte-identical
duplicates have been cleaned. The four conflicting-label copies and one
same-label duplicate were excluded without resplitting. The confirmed missing
NEFER user-22 recording is already recorded in exclusions.

**Do not redownload, re-extract, rerun the original split builder, or edit v2.**
Keep `splits_v1`, `splits_v2`, their checksums, and the cleanup evidence.
Native event integrity is still pending. Source identity independence remains
unverified because participant metadata is incomplete.

The data locations confirmed by your inventory are:

```text
/home/nndl1/data/raw/videos/             # Original Ekman RGB videos
/home/nndl1/data/nefer/raw/              # Original NEFER recordings
/home/nndl1/artifacts/splits_v2/         # Current frozen metadata
```

## 1. Upload the current code from Windows

Connect to the AUS VPN. In **Windows PowerShell**, run the following. If your
normal SSH connection uses an IP instead of `AI-Lab-12`, substitute that IP.

```powershell
ssh nndl1@AI-Lab-12 "mkdir -p /home/nndl1/scripts"

scp -r "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\eventfer" nndl1@AI-Lab-12:/home/nndl1/

scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\requirements-conversion.txt" nndl1@AI-Lab-12:/home/nndl1/

scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\requirements-baseline.txt" nndl1@AI-Lab-12:/home/nndl1/

scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\validate_event_streams.py" nndl1@AI-Lab-12:/home/nndl1/scripts/

scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\clean_frozen_splits.py" nndl1@AI-Lab-12:/home/nndl1/scripts/

ssh nndl1@AI-Lab-12
```

The remaining commands are **Linux commands** inside SSH.

## 2. Check the environment and frozen metadata

Reuse the existing environment; do not create another one:

```bash
cd /home/nndl1
conda activate ekman-events

python -m pip install -r requirements-conversion.txt -r requirements-baseline.txt
export MPLBACKEND=Agg

python scripts/clean_frozen_splits.py verify --directory artifacts/splits_v2
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"

df -h /home/nndl1
mkdir -p logs data/conversion_tmp
```

The metadata verifier should show the counts above and no source duplicates.
The Python GPU check must succeed before GPU conversion/extraction/training.
If it fails, stop and share its error; the working environment previously used
the CUDA 12.1 PyTorch 2.5.1 / torchvision 0.20.1 installation.

## 3. Install or reuse the pinned conversion repositories

These shell blocks clone only missing checkouts. The checkout commands select
the versions used by this pipeline; they do not force-discard local changes.

```bash
cd /home/nndl1
mkdir -p third_party

if [ ! -d third_party/Emo-DVS/.git ]; then
  git clone https://github.com/Typistchen/Emo-DVS.git third_party/Emo-DVS
fi
git -C third_party/Emo-DVS checkout 88cb52a70729a0731bca47f0cb7d2992ac9ceccd

if [ ! -d third_party/v2e/.git ]; then
  git clone --branch v2e-1.7.0 --depth 1 https://github.com/SensorsINI/v2e.git third_party/v2e
fi
git -C third_party/v2e fetch --depth 1 origin tag v2e-1.7.0
git -C third_party/v2e checkout v2e-1.7.0

python third_party/v2e/v2e.py --help
python third_party/Emo-DVS/preprocessing/convert_videos_to_events.py --help
```

If a command fails, resolve the error before moving to the next step. The
headless dependencies are already in the two requirements files; do not install
the old v2e requirements file containing Windows/GUI-specific packages.

## 4. Stage only the 1,632 retained source clips

The original RGB directory still contains 1,637 files. Running the official batch
converter directly on it would also convert the five excluded copies. Instead,
create a small symlink tree containing only clips from the frozen v2 manifest:

```bash
python -m eventfer.stage_source \
  --manifest artifacts/splits_v2/manifest.csv \
  --rgb-root data/raw/videos \
  --source-dataset ekman_rgb \
  --expected-count 1632 \
  --output data/conversion_input/Ekman6_v2
```

Expected final message: `Staged 1632 verified source videos`.
This rechecks source hashes and preserves class-relative paths. It does not copy,
rename, or delete the originals. It also records the manifest hash and clip/split
mapping in `data/conversion_input/Ekman6_v2/staging.json`.

The new staging directory must not already exist. If you already completed this
step, do not create another copy blindly; inspect its `staging.json` and reuse the
verified tree. If a staging run failed partway, choose a fresh staging directory
and use that new path in the conversion/validation commands below.

## 5. Run a two-second source-training pilot

Pick the first source **training** clip from the frozen manifest. The pilot is
written separately from the full outputs and will never serve as a full clip:

```bash
PILOT_VIDEO="$(python - <<'PY'
import csv
with open('artifacts/splits_v2/manifest.csv', newline='') as f:
    row = next(r for r in csv.DictReader(f)
               if r['dataset'] == 'ekman_rgb' and r['split'] == 'train')
print(row['path'])
PY
)"

export MPLBACKEND=Agg
export TMPDIR=/home/nndl1/data/conversion_tmp

CUDA_VISIBLE_DEVICES=0 python third_party/v2e/v2e.py \
  -i "$PILOT_VIDEO" \
  -o data/events_pilot/v2_source \
  --overwrite --no_preview --skip_video_output --disable_slomo \
  --output_width 336 --output_height 224 \
  --dvs_h5 events.h5 --dvs_aedat2 None \
  --dvs_emulator_seed 476 --stop_time 2

python scripts/validate_event_streams.py \
  --root data/events_pilot/v2_source \
  --output artifacts/v2_pilot_integrity.json

du -sh data/events_pilot/v2_source
df -h /home/nndl1
```

Require `Valid=1 Invalid=0 Missing=0`. If the pilot fails or yields no events,
inspect the error before the full run. A short pilot is not a reliable whole-data
storage/runtime estimate; keep enough storage available for generated streams
and v2e's temporary frames. The configured temporary directory is on the same
home filesystem, so it also uses that free space.

These settings follow the public converter's no-SuperSloMo route. They do not
recover motion timing absent from the original frame rate. Preserve any simulator
warnings and parameter logs when documenting simulation limitations.

## 6. Convert the retained source dataset inside Screen

Use the new output directory below, separate from any earlier pilot or trial
outputs. If it already exists from a previous run, inspect its logs and integrity
before deciding how to resume; the upstream wrapper skips existing `events.h5`
files without checking whether an earlier conversion finished.

```bash
screen -S ekman-v2-convert
```

Inside Screen:

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg
export TMPDIR=/home/nndl1/data/conversion_tmp
mkdir -p logs data/conversion_tmp
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python -u third_party/Emo-DVS/preprocessing/convert_videos_to_events.py \
  --input-root data/conversion_input/Ekman6_v2 \
  --output-root data/events/Ekman6_v2 \
  --v2e-root third_party/v2e \
  --save-error-output --dvs_aedat2 None --dvs_emulator_seed 476 \
  2>&1 | tee logs/ekman_v2_conversion.log
```

Press **Ctrl+A, then D** to detach. Reconnect with:

```bash
screen -r ekman-v2-convert
```

To inspect progress from another SSH terminal:

```bash
tail -n 20 /home/nndl1/logs/ekman_v2_conversion.log
```

Each retained source clip should produce:

```text
data/events/Ekman6_v2/<class>/<video_stem>/events.h5
```

For this first run in a fresh output directory, expect
`Total=1632 Success=1632 Skipped=0 Failed=0`.

## 7. Check full conversion completion and event integrity

After the conversion command finishes:

```bash
cd /home/nndl1
conda activate ekman-events

python - <<'PY'
from pathlib import Path
root = Path('data/events/Ekman6_v2/_logs')
def lines(name):
    return [line for line in (root/name).read_text().splitlines() if line.strip()]
success = lines('success.txt')
skipped = lines('skipped.txt')
failed = lines('failed.txt')
print('Success:', len(success), 'Skipped:', len(skipped), 'Failed:', len(failed))
assert len(success) == 1632 and not skipped and not failed, 'Inspect conversion logs before proceeding'
PY

python scripts/validate_event_streams.py \
  --root data/events/Ekman6_v2 \
  --expected-input-root data/conversion_input/Ekman6_v2 \
  --output artifacts/ekman_v2_event_integrity.json
```

Require `Valid=1632 Invalid=0 Missing=0`. If either check fails, stop and share
the log/report. Do not silently remove failed clips or change the frozen v2
manifest. An interrupted file can sometimes be a readable H5 prefix, so retain
successful converter exit/log evidence as well as the structural checks.

The 333 source-test clips may be converted and checked mechanically here, but
do not inspect their model scores or use them to choose parameters.

## 8. Link the frozen splits to full event files

```bash
python -m eventfer.prepare_manifest \
  --manifest artifacts/splits_v2/manifest.csv \
  --source-rgb-root data/raw/videos \
  --source-events-root data/events/Ekman6_v2 \
  --source-width 336 --source-height 224 \
  --output artifacts/event_manifest_v2
```

This preserves labels, clip IDs, group IDs, and splits. It changes the source
input paths to full converted streams and records their hashes. NEFER paths
remain the original raw recordings. The combined derivative manifest should
contain 2,076 rows: 1,632 source + 347 reserve + 97 target test.

Existing artifact/output directories are deliberately not overwritten by these
pipeline commands. If one already exists, verify and reuse it or choose a new
versioned output directory, updating later paths consistently.

## 9. Choose and record the crop policy

Use **one** of the following options for this experiment version. Do not run both
into the same count cache.

### A. Preferred: one fixed face box per clip

If usable face boxes are available, create `artifacts/fixed_crops_v2.csv` with
these columns and actual boxes for every selected clip:

```text
dataset,clip_id,x0,y0,x1,y1
```

Use exact dataset and clip IDs from the manifest. Coordinates are in the event
sensor's pixel grid; right/bottom edges are exclusive. A clip has one fixed box,
not a crop that moves each frame. Derive boxes without emotion labels or target
performance. Do not fill this file with invented or placeholder coordinates.

```bash
python -m eventfer.preprocess counts \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --crop-policy fixed-boxes --crops artifacts/fixed_crops_v2.csv \
  --splits train validation \
  --output artifacts/counts_source_v2
```

### B. Explicit fallback when usable face boxes are unavailable

```bash
python -m eventfer.preprocess counts \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --crop-policy full-frame \
  --splits train validation \
  --output artifacts/counts_source_v2
```

This logs full-frame processing. It does not establish that every Ekman video
contains a face or that the signal is face-localized; disclose that limitation.
Do not claim a face-cropped baseline if this fallback is used.

Both choices build **1,299 source training/validation clips**, each with eight
224x224 positive/negative count images. Neither includes source-test or NEFER
target-test counts. Counting is a CPU task and validates each selected stream.

## 10. Fit training-only statistics and inspect a preview

```bash
python -m eventfer.preprocess fit \
  --index artifacts/counts_source_v2/index.csv \
  --output artifacts/source_statistics_v2

python -m eventfer.preprocess preview \
  --index artifacts/counts_source_v2/index.csv \
  --statistics artifacts/source_statistics_v2/statistics.json \
  --output artifacts/preprocessing_preview_v2.png
```

Inspect the preview for crop alignment, visible activity, and plausible temporal
bins. The preview selects a source training clip. The threshold R uses only the
1,140 source training clips' nonempty bins. The same statistics will be reused
for every representation and all later test data.

To copy the preview back, run in **Windows PowerShell**:

```powershell
scp nndl1@AI-Lab-12:/home/nndl1/artifacts/preprocessing_preview_v2.png "$env:USERPROFILE\Downloads\preprocessing_preview_v2.png"
```

## 11. Install the pinned DINOv2 code

Back on Linux:

```bash
cd /home/nndl1
conda activate ekman-events

if [ ! -d third_party/dinov2/.git ]; then
  git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
fi
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

The pretrained ViT-S/14 weights download on first extraction. DINOv2 stays frozen;
only the later GRU/head is trained. Optional xFormers warnings are expected.

## 12. Cache baseline features on the GPU

Run this inside a new Screen session if it may outlast your SSH connection:

```bash
screen -S ekman-v2-features
```

Inside that session:

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg

CUDA_VISIBLE_DEVICES=0 python -m eventfer.features \
  --index artifacts/counts_source_v2/index.csv \
  --statistics artifacts/source_statistics_v2/statistics.json \
  --dinov2-repo third_party/dinov2 \
  --mode baseline --device cuda --batch-size 4 \
  --output artifacts/features_baseline_v2
```

This should cache 1,299 feature arrays of shape `(8,384)`. If GPU memory is
insufficient, use batch size 2 or 1 and a fresh output directory after the failed
attempt. The extractor does not resume partial caches. Detach with Ctrl+A then D;
reattach using `screen -r ekman-v2-features`.

## 13. Check training on a tiny source subset

After feature extraction finishes:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventfer.train \
  --index artifacts/features_baseline_v2/index.csv \
  --output runs/baseline_tiny_v2 --device cuda \
  --seeds 17 --tiny-per-class 2 --dropout 0 \
  --lr 0.003 --epochs 100 --patience 100
```

Inspect the training trajectory:

```bash
python - <<'PY'
import json
from pathlib import Path
history = [json.loads(line) for line in Path('runs/baseline_tiny_v2/seed_17/history.jsonl').read_text().splitlines()]
print('Best tiny-subset training macro-F1:', max(r['train_macro_f1'] for r in history))
print('Last epoch:', history[-1]['epoch'])
PY
```

The head should be able to fit a very small source subset. If training F1 stays
low, inspect the data/representations and training behavior before spending time
on full runs. This is a correctness check, not a reportable experimental result.
Checkpoint selection still uses source validation; test sets remain untouched.

## 14. Train the three-seed source baseline

Start another Screen session, or reuse the current one after its job finishes:

```bash
screen -S ekman-v2-baseline
```

Inside it:

```bash
cd /home/nndl1
conda activate ekman-events

CUDA_VISIBLE_DEVICES=0 python -m eventfer.train \
  --index artifacts/features_baseline_v2/index.csv \
  --output runs/baseline_v2 --device cuda \
  --seeds 17 29 43
```

The baseline trains the GRU and linear head only, with training-only class weights
and source-validation macro-F1 for checkpoint selection. Defaults are 50 epochs
maximum and patience 8. Retained seeds share the prescribed architecture/budget.

After training:

```bash
cat runs/baseline_v2/validation_summary.json

python -m pip freeze > logs/baseline_environment.txt
git -C third_party/Emo-DVS rev-parse HEAD > logs/emo_dvs_commit.txt
git -C third_party/v2e rev-parse HEAD > logs/v2e_commit.txt
git -C third_party/dinov2 rev-parse HEAD > logs/dinov2_commit.txt
```

Each seed directory contains `best.pt`, `best_validation.json`, `history.jsonl`,
`config.json`, and `summary.json`. These are source-development outputs, not
final target-test results.

## 15. Stop before final test scoring

The next implementation and exact commands are in
[CONTROLLED_EXPERIMENT_GUIDE.md](CONTROLLED_EXPERIMENT_GUIDE.md), covering steps 7
and 8: the nine paired source-only runs, checkpoint/evaluation freeze, target RAW
export, and locked scoring/analysis. Continue there after completing this guide.

Share the preprocessing preview, event-integrity report, tiny-run training
behavior, and baseline `validation_summary.json` for review.

The next controlled experiments will use the existing count cache and training
statistics, re-extracting DINOv2 features with `--mode per_bin` and
`--mode whole_clip`. Use the same seeds/settings with separate cache/run paths.
Do not scale already-cached embeddings as a substitute for changing counts.

Full target evaluation is a later step. NEFER RAW decoding needs the compatible
Metavision/OpenEB SDK or a documented canonical export. It is not required for
the source-only baseline commands above, which retain/hash NEFER paths but do
not decode its recordings or use its scores.

## What can run in parallel now?

| Task | Parallel work allowed |
|---|---|
| Source conversion | DINOv2 clone, literature/report work, storage monitoring |
| Source count-cache building | CPU work; other GPU work only if host RAM/I/O allow |
| DINOv2 extraction | Report work; avoid another heavy job on the same GPU |
| GRU/head training | CPU/report work; run seeds sequentially on this single GPU |

Do not start downstream commands before their inputs are complete, or two jobs
writing to the same directory. Source/test preprocessing choices must stay fixed
before inspecting test performance.

## How to describe this study

Use **locally simulated Ekman-6-to-NEFER transfer using the public Emo-DVS
conversion code**, rather than claiming the exact released Emo-DVS subset.
Disclose the duplicate cleanup, missing NEFER reserve clip, crop policy,
frame-rate-limited simulation, remaining identity-overlap uncertainty, and all
pretrained code/weights. A rigorous negative result is a valid outcome.
