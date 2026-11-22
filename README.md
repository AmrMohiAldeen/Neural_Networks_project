# COE 476 Track A project

**Current Linux workflow:** follow [SUBSET300_GUIDE.md](SUBSET300_GUIDE.md) from conversion to final
results (300-clip source subset: 210 train / 29 validation / 61 test, plus the 97 NEFER target-test
clips). It supersedes steps 6+ of [POST_AUDIT_GUIDE.md](POST_AUDIT_GUIDE.md) and
[CONTROLLED_EXPERIMENT_GUIDE.md](CONTROLLED_EXPERIMENT_GUIDE.md).

The actual project root is `/home/nndl1`; `artifacts/splits_v2` has been verified
with 1,140 source train / 159 validation / 333 test clips, 347 NEFER reserve clips,
and 97 target-test clips. The cleaned source total is 1,632. Use the new staging
command to exclude the five removed copies from conversion.

The earlier download/setup sections below retain historical example paths under
`~/coe476-project`. Do not rerun downloads, extraction, or the original split
builder for the current verified dataset. Native event validation and model
training are separate remaining stages.

Open `main.tex` as the root document. In Overleaf, upload `main.tex`, `references.bib`, and the `sections/` directory, then select pdfLaTeX. To build locally with a TeX distribution that includes IEEEtran, run:

```sh
latexmk -pdf main.tex
```

The literature review and research design are in separate files under `sections/`. Uncomment the remaining headings in `main.tex` when their content is ready. Replace the course line with the team names and complete the contribution and AI assistance statements before submission. The current source is a proposal draft; it has no experimental results. The final report target is 6–8 pages excluding references.

## Downloading the supplied video dataset on the AI Lab machine

This downloads [Zenodo record 17159328](https://zenodo.org/records/17159328):
Ekman-6/VideoEmotion-8 RGB videos and pretrained features by Serkan Sulun.
It is **not Emo-DVS** and does not include event-camera recordings. RGB-to-event
simulation would be a separate step. The script never loads the pickle files.

The workflow follows the lab's [VPN, SSH, Conda, and Screen guide](https://github.com/Mahmoud-Darweesh/AI-Lab-Machine-Guide).
Replace `YOUR_USERNAME` and `LAB_HOST` with the details assigned by the lab.

1. Connect to the AUS VPN. In your laptop's PowerShell terminal, create the remote
   project directory and upload the downloader:

   ```powershell
   ssh YOUR_USERNAME@LAB_HOST "mkdir -p ~/coe476-project/scripts"
   scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\download_video_emotion.py" YOUR_USERNAME@LAB_HOST:coe476-project/scripts/
   ssh YOUR_USERNAME@LAB_HOST
   ```

2. The remaining commands run on the Linux lab machine. Use an existing Python
   3.8+ environment, or create a dedicated one with the lab's Conda installation:

   ```bash
   conda create -n coe476-data python=3.11 -y
   conda activate coe476-data
   cd ~/coe476-project
   python --version
   df -h .
   python -u scripts/download_video_emotion.py --list
   ```

   The full release is approximately 24.9 GB. Allow at least 30 GB free for the
   download; extraction needs additional space. Check your account's storage
   quota too, if applicable. Nothing is extracted automatically. If home storage
   is limited, use your lab-assigned persistent data directory with `--output`.

3. Start a persistent session. If `screen` is unavailable, install it in the Conda
   environment with `conda install -c conda-forge screen` as in the lab guide.

   ```bash
   screen -S coe476-download
   ```

   Inside Screen, run:

   ```bash
   conda activate coe476-data
   cd ~/coe476-project
   mkdir -p logs
   set -o pipefail
   python -u scripts/download_video_emotion.py 2>&1 | tee -a logs/dataset-download.log
   ```

   Press **Ctrl+A**, then **D** to detach. You may then disconnect SSH. Reconnect
   and run `screen -r coe476-download` to return. Do not launch a second downloader
   into the same destination while the first is running.

4. A successful run ends with `All files downloaded and verified.` The output is:

   ```text
   ~/coe476-project/data/ekman6_videoemotion8/
     readme.txt
     ekman6_blacklist.txt
     ekman6_features.pkl
     videoemotion8_features.pkl
     videos.zip
     zenodo_record.json
   ```

   The script retries interrupted transfers, resumes `.part` files when the
   server supports byte ranges, and skips completed files only after checking
   their sizes and published MD5 hashes. Run the same command again after an
   interruption. A checksum mismatch stops the run for inspection.

Optional commands (run from the remote project directory):

```bash
# Use another persistent storage directory assigned to you by the lab.
python -u scripts/download_video_emotion.py --output /YOUR/ASSIGNED/STORAGE/video-emotion

# Download only the Ekman-6 features and documentation, without the large videos.
python -u scripts/download_video_emotion.py --files ekman6_features.pkl readme.txt ekman6_blacklist.txt

# Inspect the ZIP file listing without extracting it.
python -m zipfile -l data/ekman6_videoemotion8/videos.zip
```

Keep downloaded datasets out of Git. Preserve the supplied README and citations.
For later experiments, compute normalization statistics on your training split;
the supplied feature statistics were computed across the entire dataset.

## Downloading the supplied Google Drive folder

`scripts/download_drive_folder.py` downloads this folder recursively:
https://drive.google.com/drive/folders/1OAnM8B88XX_TqJj2XF61w1nA-nWNV4YC

Upload this script to `~/coe476-project/scripts/` using the same SCP procedure
above. On the lab machine, activate your Python environment and run:

```bash
cd ~/coe476-project
python -m pip install 'gdown>=5.2,<6'
screen -S drive-download
```

Inside Screen:

```bash
cd ~/coe476-project
python -u scripts/download_drive_folder.py
```

Press Ctrl+A, then D to detach; reconnect with `screen -r drive-download`.
The default destination is `data/google_drive_dataset/` under the project.
To choose lab-assigned storage instead:

```bash
python -u scripts/download_drive_folder.py --output /YOUR/ASSIGNED/STORAGE/dataset
```

Run the same command to resume interrupted downloads. Completed files are skipped
by gdown; they are not validated against published checksums. This downloader
uses public access: the folder and files must be accessible without signing in.
If Drive reports a quota or access error, resolve that and rerun the command.
Folder discovery failures are reported rather than silently downloading a subset.
No files are extracted automatically, and no GPU is needed.

## Step 4: data audit and frozen metadata splits

Run `scripts/audit_datasets.py` on the machine containing the datasets, using
Python 3.11. This step uses CPU and storage, not CUDA. It does not fit any
normalization statistics or inspect target performance.

First upload the script using SCP as above, then inventory the two dataset roots.
Replace the paths below with the actual directories:

```bash
cd ~/coe476-project
python scripts/audit_datasets.py inventory \
  --roots /PATH/TO/SOURCE_DATA /PATH/TO/NEFER \
  --output artifacts/data_inventory.json
```

This lists file types, sample paths, CSV previews, and ZIP contents without
extracting archives or opening pickle files. Share this JSON with Codex to
confirm the actual release layout before building manifests. Downloaded archives
must be extracted before `build` can resolve recordings. Do not treat TBR
occupancy frames as raw polarity-separated event counts.

Get the official NEFER annotations if they are not included in your download:

```bash
git clone https://github.com/miccunifi/NEFER.git third_party/NEFER
```

If that directory already exists, use its `annotations` directory. This script
expects the publisher's headerless, four-column frame annotation CSVs, groups
them into clips, and uses reported emotions as primary labels. Official validation
participants `03,07,17,19,27,28` become `target_test`; the 20 official training
participants become `target_reserve` and never enter the main source training.
`None` and `Contempt` clips are recorded in exclusions; unknown labels stop the
audit. Expected labels are included when the matching official CSV exists.

For the supplied **Ekman-6 RGB videos** and trusted publisher feature pickle:

```bash
python -m pip install numpy
python -u scripts/audit_datasets.py build \
  --source-kind ekman_rgb \
  --source-root /PATH/TO/EXTRACTED/VIDEOS \
  --ekman-pickle /PATH/TO/ekman6_features.pkl --trust-pickle \
  --nefer-root /PATH/TO/EXTRACTED/NEFER \
  --annotations-root third_party/NEFER/annotations \
  --output artifacts/splits_v1
```

Only use `--trust-pickle` for the file from the trusted publisher: unpickling
executes code. The importer uses paths and one-hot labels, not pretrained features
or global feature statistics. It matches exact paths, or a unique filename stem,
and stops on missing or ambiguous recordings. RGB input is explicitly marked
`ekman_rgb`: this is not an Emo-DVS experiment. Event simulation and an amendment
to the proposed source-data protocol are still needed. Ekman-6 video-level emotion
labels also do not establish that each video contains a face or a visible facial
expression; assess suitability before preprocessing. The supplied blacklist is
not applied automatically, and no clips are filtered based on test performance.

Alternatively, use a documented source metadata CSV (required for actual Emo-DVS
events or when participant/original-video IDs are available):

```text
path,label,original_video_id,participant_id,split
relative/path/clip.raw,anger,original_001,person_01,train
```

Required columns: `path,label,original_video_id`. Optional: `participant_id,split`.
Labels must be names; `joy` maps to `happiness`. Obtain original-video IDs and
participant IDs from actual metadata rather than inventing them. Provide
`train/validation/test` for every source clip to audit existing partitions, or
leave every split empty to generate seeded grouped 70/10/20 partitions. The
ratio is by group count, so clip/class proportions can differ. All clips connected
by original-video ID, known participant, or identical SHA256 are kept together.
Missing class coverage stops the build for protocol review; do not search seeds
based on target results.

```bash
python -u scripts/audit_datasets.py build \
  --source-kind emo_events \
  --source-root /PATH/TO/EMO_DVS \
  --source-csv /PATH/TO/source_metadata.csv \
  --nefer-root /PATH/TO/NEFER \
  --annotations-root third_party/NEFER/annotations \
  --output artifacts/splits_v1
```

Use `--source-kind ekman_rgb` with `--source-csv` for RGB metadata. Include only
the intended compatible emotion subset in the source CSV; exclude real FAU data.
The script does not infer provenance from a file extension.

Outputs are `manifest.csv`, `class_counts.csv`, `participant_counts.csv`,
`exclusions.csv`, `audit.json`, and `checksums.json`. SHA256 hashing reads every
included recording and may take time. Empty files, identical files across splits,
known source group leaks, and wrong/missing NEFER participant sets stop the build.
An existing output directory is never overwritten; use a new version directory
for a documented correction before any training or test scoring.

These are frozen **metadata splits**, not proof of event-stream integrity.
Native event timestamps, coordinates, polarity conventions, and face crops need
format-specific validation once the actual file layout is known. Unknown source
identities and re-encoded duplicate videos remain limitations. Record versions,
permissions, and preprocessing rules with the manifests. Fit the scaling threshold,
normalization, and class weights on source training clips only in step 5.

Verify the scripts offline with:

```bash
python -m unittest discover -s tests -v
```

## RGB-to-event conversion on the Linux lab machine

This uses the official [Emo-DVS batch conversion script](https://github.com/Typistchen/Emo-DVS/blob/88cb52a70729a0731bca47f0cb7d2992ac9ceccd/preprocessing/convert_videos_to_events.py).
That script disables SuperSloMo, saves `events.h5`, and uses width 336 / height
224. The generated data are locally simulated Ekman-6 events using the authors'
public conversion code; do not claim the exact released Emo-DVS subset or settings.
The repository's multimodal training split is not the grouped split in our study.

These commands assume the earlier downloader stored `videos.zip` in
`~/coe476-project/data/ekman6_videoemotion8/`. Run on Linux after connecting to
the VPN and SSH. Clone each repository only once:

```bash
cd ~/coe476-project
conda create -n ekman-events -c conda-forge python=3.11 tk ffmpeg screen -y
conda activate ekman-events
mkdir -p third_party
git clone https://github.com/Typistchen/Emo-DVS.git third_party/Emo-DVS
git -C third_party/Emo-DVS checkout 88cb52a70729a0731bca47f0cb7d2992ac9ceccd
git clone --branch v2e-1.7.0 --depth 1 https://github.com/SensorsINI/v2e.git third_party/v2e

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 \
  numba==0.61.2 'h5py>=3.8,<4' 'matplotlib>=3.7,<3.10' 'pandas>=2,<3' \
  pillow tqdm argcomplete engineering-notation easygui screeninfo plyer
export MPLBACKEND=Agg
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(1,device='cuda'))"
python third_party/v2e/v2e.py --help
```

Stop if the CUDA check fails. These PyTorch wheels need a compatible NVIDIA
driver; the GPU appearing in `lspci` alone does not establish CUDA availability.
The pinned v2e release avoids the current main branch's Python 3.14 requirement
and the legacy requirements file's Windows/GUI-only packages. This is a minimal
headless conversion environment; the unrelated audio/text/FAU training stack is
not required for this step.

Extract the downloaded ZIP. It has a `videos/` prefix containing exactly six
class directories, with 1,637 MP4 files. It expands to approximately 23.13 GB.
Keep extra storage available for generated events; measure pilot size before
launching the full dataset. The following extraction checks the ZIP MD5 first:

```bash
cd ~/coe476-project
df -h .
python - <<'PY'
import hashlib
from pathlib import Path
import zipfile
archive = Path('data/ekman6_videoemotion8/videos.zip')
h = hashlib.md5()
with archive.open('rb') as f:
    for chunk in iter(lambda: f.read(8*1024*1024), b''):
        h.update(chunk)
assert h.hexdigest() == 'a0ec04425c3ac71bcfd050270c10a533', 'Archive checksum mismatch'
root = Path('data/raw').resolve()
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        assert (root/member.filename).resolve().is_relative_to(root), 'Unsafe archive path'
    z.extractall(root)
print('Extracted to', root/'videos')
PY
```

If you used `scripts/download_ekman_rgb.py` instead of downloading `videos.zip`,
the extracted videos are already at `data/raw/Ekman6`; skip extraction and change
`RGB_ROOT` below accordingly. That alternative requires no extra Python packages:

```bash
python -u scripts/download_ekman_rgb.py --workers 4
```

Run a **two-second pilot only** into a separate directory before the full run:

```bash
RGB_ROOT="$HOME/coe476-project/data/raw/videos"
PILOT_VIDEO="$(python -c "from pathlib import Path; print(next(iter(sorted(Path('$RGB_ROOT').rglob('*.mp4')))))")"
python third_party/v2e/v2e.py \
  -i "$PILOT_VIDEO" -o data/events_pilot/first_clip \
  --overwrite --no_preview --skip_video_output --disable_slomo \
  --output_width 336 --output_height 224 \
  --dvs_h5 events.h5 --dvs_aedat2 None --dvs_emulator_seed 476 --stop_time 2

python - <<'PY'
import h5py
with h5py.File('data/events_pilot/first_clip/events.h5', 'r') as f:
    e = f['events']
    print('Shape:', e.shape, 'dtype:', e.dtype)
    print('First events [t_us,x,y,p]:', e[:5])
    assert e.ndim == 2 and e.shape[1] == 4 and len(e) > 0
PY
du -sh data/events_pilot
```

The pilot can be unrepresentative of storage/runtime across all videos. The
timestamp resolution is limited by source frame rate because this follows the
authors' `--disable_slomo` path; it does not recover real sensor timing.

Once the pilot succeeds and storage is sufficient, run the entire dataset in
Screen. Skip Screen creation if you are already in a persistent session:

```bash
screen -S ekman-convert
conda activate ekman-events
cd ~/coe476-project
export MPLBACKEND=Agg
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u third_party/Emo-DVS/preprocessing/convert_videos_to_events.py \
  --input-root data/raw/videos \
  --output-root data/events/Ekman6 \
  --v2e-root third_party/v2e \
  --save-error-output --dvs_aedat2 None --dvs_emulator_seed 476 \
  2>&1 | tee logs/ekman-conversion.log
```

Change `--input-root` to `data/raw/Ekman6` for direct video downloads. Detach with
Ctrl+A then D, and reconnect with `screen -r ekman-convert`. Each video produces
`data/events/Ekman6/<class>/<video_stem>/events.h5`. The official wrapper writes
success, skipped, failed, and executed-command lists under
`data/events/Ekman6/_logs/`. It skips any existing `events.h5` without validating
it, so an interrupted output must be inspected and regenerated before relying on
the skipped count. Do not run two conversions into the same output directory.

Upload `scripts/validate_event_streams.py` from this project, then validate every
generated stream and require one output per input video:

```bash
python scripts/validate_event_streams.py \
  --root data/events/Ekman6 --expected-input-root data/raw/videos \
  --output artifacts/ekman_event_integrity.json
python -m pip freeze > logs/conversion-environment.txt
git -C third_party/Emo-DVS rev-parse HEAD > logs/emo-dvs-commit.txt
git -C third_party/v2e rev-parse HEAD > logs/v2e-commit.txt
```

Again change `--expected-input-root` if using direct downloads. Validation checks
the pinned v2e H5 format, nonempty events, timestamp order, coordinate bounds,
binary polarity codes, and missing output clips. It does not detect whether an
otherwise valid file contains only the beginning of a clip; retain successful
converter exit status and logs as completion evidence too. For an interrupted
individual conversion, regenerate that clip into a fresh output directory with
the direct `v2e.py` command, omitting `--stop_time`.

## Steps 5 and 6: preprocessing and the frozen-backbone baseline

The implementation is in `eventfer/`. Run the commands below from the project
root after copying that directory and `requirements-baseline.txt` to the Linux
machine. The cleaned step-4 metadata audit is complete; use the verified splits_v2 manifest. The
one-clip fixture in `artifacts/baseline_smoke/` is only a software smoke test;
it is not an experimental split or a reported baseline result.

To copy the new code from your laptop, run in PowerShell (substitute your login):

```powershell
scp -r "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\eventfer" nndl1@AI-Lab-12:/home/nndl1/
scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\requirements-baseline.txt" nndl1@AI-Lab-12:/home/nndl1/
```

The pipeline implements:

```text
frozen metadata manifest
  -> link simulated events, preserving every clip's split
  -> fixed crop and 8 equal-duration bins
  -> 224x224 positive/negative counts
  -> optional bounded scaling before log compression
  -> [positive, negative, sum] channels
  -> frozen pretrained DINOv2 ViT-S/14: 8x384 embeddings
  -> trainable GRU(128) + dropout + six-class linear head
```

### Install the baseline dependencies

Reuse the `ekman-events` environment and its CUDA PyTorch installation:

```bash
cd /home/nndl1
conda activate ekman-events
python -m pip install -r requirements-baseline.txt

git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

Skip cloning if that directory exists. The extractor records the DINOv2 commit
and SHA256 of the loaded model weights, and rejects tracked code modifications.
Pretrained weights (approximately 84 MB) download on first extraction. Optional
xFormers warnings are expected; it is not needed for this inference baseline.
See the [official pretrained backbone instructions](https://github.com/facebookresearch/dinov2).

### Link the frozen RGB splits to generated events

This expects `artifacts/splits_v2/manifest.csv` and its step-4 `checksums.json`.
It verifies the frozen files and the RGB hashes, and creates a new derivative
manifest rather than modifying the frozen original:

```bash
python -m eventfer.prepare_manifest \
  --manifest artifacts/splits_v2/manifest.csv \
  --source-rgb-root data/raw/videos \
  --source-events-root data/events/Ekman6_v2 \
  --source-width 336 --source-height 224 \
  --output artifacts/event_manifest_v2
```

The matching rule is exact: `anger/example.mp4` maps to
`anger/example/events.h5`. All conversions must finish successfully before this
step. The event manifest retains dataset names, labels, clip IDs, group IDs, and
splits, and records the generated event-file hashes. Existing event rows are
verified and retained. For source H5/NPZ events that were already in the original
manifest, set `--source-width` and `--source-height` to their actual sensor size.
For canonical target H5/NPZ, also supply actual `--target-width`/`--target-height`;
RAW/DAT dimensions can be read from the sensor metadata.

### Build source training and validation count frames

Choose a crop policy before testing. Preferred: supply a fixed box per clip
using a CSV with these columns:

```text
dataset,clip_id,x0,y0,x1,y1
ekman_rgb,anger/example.mp4,40,20,280,210
```

Coordinates are integer pixels in the event sensor coordinate system, with
exclusive right/bottom edges. `dataset` and `clip_id` must exactly match the
manifest. Every selected clip needs one box; moving frame-by-frame crops are
not used. Boxes must be obtained without using emotion labels or target scores.
This code does not invent face boxes or assume that all Ekman videos show faces.

```bash
python -m eventfer.preprocess counts \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --crop-policy fixed-boxes --crops artifacts/fixed_crops.csv \
  --splits train validation \
  --output artifacts/counts_source_v2
```

If no usable face annotations exist and you choose the documented, label-blind
full-frame fallback, replace `--crop-policy fixed-boxes --crops ...` with
`--crop-policy full-frame`. Full-frame input is explicitly logged and does not
establish face-localized expression recognition. Use the same predeclared policy
for all representation variants.

Each event stream is validated before counting: integer timestamps in
microseconds, monotonic ordering including chunk boundaries, coordinate bounds,
and explicit binary or signed polarity. H5 is read in chunks, with two passes
for temporal extent and counts. Time bins cover the full event stream's first
to last timestamps, even when cropping removes events; the final endpoint is
included in bin eight. Empty temporal bins stay zero. Empty entire streams stop
the run for audit instead of being silently excluded.

Supported inputs:

- v2e H5: `events[N,4]` with `[t_us,x,y,p]` columns.
- Canonical NPZ: equal-length 1D arrays `t,x,y,p`, with integer microsecond
  timestamps. NPZ loads into memory; prefer H5 for large clips.
- Prophesee RAW/DAT through `metavision_core.event_io.EventsIterator`. This needs
  the separately installed Metavision/OpenEB Python SDK. Do not try to load RAW
  as a NumPy array or replace it with rendered TBR images. The source-only
  baseline can be developed before installing this target-data reader.

The SDK adapter follows [Metavision's EventsIterator API](https://docs.prophesee.ai/stable/api/python/core/event_io.html).
H5 layouts other than the stated canonical format require an explicit adapter.

### Fit the source-training scaling threshold and inspect the frames

```bash
python -m eventfer.preprocess fit \
  --index artifacts/counts_source_v2/index.csv \
  --output artifacts/source_statistics_v2

python -m eventfer.preprocess preview \
  --index artifacts/counts_source_v2/index.csv \
  --statistics artifacts/source_statistics_v2/statistics.json \
  --output artifacts/preprocessing_preview.png
```

The threshold R is the linearly interpolated 75th percentile of **nonempty bins
from source training only**, after cropping. Neither source validation nor NEFER
contributes. The preview uses a source training clip; add `--clip-id` to choose
another. Verify face/crop alignment and positive/negative activity before GPU
feature extraction.

All variants use the same fixed compression `log1p(count)/log(256)`, followed by
fixed ImageNet mean/std normalization at the backbone. Inputs are not rescaled
per frame and are not clipped. Values above 1 are possible for very dense pixels.
The preview clips values to [0,1] for display only. Per-bin scaling is bounded to
[0.25,1] and applied to both polarities before recomputing the sum channel and
compressing. Whole-clip ablation uses the mean bin activity for one shared scale.

### Extract baseline features on the GPU

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventfer.features \
  --index artifacts/counts_source_v2/index.csv \
  --statistics artifacts/source_statistics_v2/statistics.json \
  --dinov2-repo third_party/dinov2 \
  --mode baseline --device cuda --batch-size 4 \
  --output artifacts/features_baseline_v2
```

This caches a float32 `(8,384)` array per clip. Only four bin images are sent
through the backbone at a time, so an 8 GB GPU can start with a modest batch.
Reduce `--batch-size` to 2 or 1 if necessary. DINOv2 runs in evaluation/inference
mode with gradients disabled. Its weights are never trained.

Feature extraction defaults to source train/validation. Test representations can
later be built with explicitly requested `--splits`, using the same frozen
source statistics and crop settings. The trainer still never loads test or
target feature arrays. No final test evaluation is implemented in these steps.
Test extraction now also requires `--evaluation-lock` from the controlled
experiment freeze. The separate controlled-experiment guide covers that command,
NEFER decoding/export, scoring, and the generated analysis artifacts.

### Check the head on a tiny source subset

```bash
python -m eventfer.train \
  --index artifacts/features_baseline_v2/index.csv \
  --output runs/baseline_tiny_v2 --device cuda \
  --seeds 17 --tiny-per-class 2 --dropout 0 \
  --lr 0.003 --epochs 100 --patience 100
```

Inspect training macro-F1 in `history.jsonl` to see whether the head can memorize
the tiny subset. This is a correctness check, not a reportable baseline result.
Validation is still the source validation split, and its performance does not
determine which tiny training samples are used. Poor memorization can indicate
bad labels, identical/empty representations, or a training problem.

### Train the actual baseline with three seeds

```bash
python -m eventfer.train \
  --index artifacts/features_baseline_v2/index.csv \
  --output runs/baseline_v2 --device cuda \
  --seeds 17 29 43
```

Defaults: GRU hidden size 128, dropout 0.2, AdamW learning rate 0.001, weight decay
0.0001, batch size 32, gradient norm clipping at 1, maximum 50 epochs, and patience
8. Inverse-frequency class weights use source training labels only. The first
best source-validation macro-F1 checkpoint is saved; ties keep the earlier one.
Macro-F1 always averages all six classes, assigning zero to undefined class F1.

Each seed directory contains:

```text
config.json             # Settings, feature provenance, class weights, hardware
history.jsonl           # Per-epoch train loss/F1 and source-validation metrics
best.pt                 # GRU/head weights and model/config metadata
best_validation.json    # Source-validation metrics and confusion matrix
summary.json            # Best epoch, validation F1, checkpoint checksum
```

`validation_summary.json` collects the seed summaries. Output directories are
versioned and never overwritten; choose a new output path after a correction.
Count and feature caches have checksummed indexes/configs and per-clip hashes.
Failed cache builds are incomplete; use a fresh directory when rerunning. These
commands do not promise cache-build resume.

### Proposed representation and ablation, when ready

Use the same counts and statistics. Repeat feature extraction with
`--mode per_bin` / `--output artifacts/features_per_bin_v2`, then train with the
same seeds and settings under `runs/per_bin_v2`. For the ablation use
`--mode whole_clip` and separate output directories. Changing representations
requires extracting the frozen backbone features again; scaling cached DINOv2
embeddings would not implement the intervention.

Run long feature extraction or training inside Screen as described above.
Keep the source test and NEFER target test results locked until model development
and evaluation rules are complete. The full experimental audit and split freeze
are separate from these software checks.
