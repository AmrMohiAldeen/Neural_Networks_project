# Remaining work: 300-clip source subset, from conversion to final results

This is now the **only guide to follow**. It replaces steps 6–15 of
`POST_AUDIT_GUIDE.md` and all of `CONTROLLED_EXPERIMENT_GUIDE.md`. Steps 1–5 of
the post-audit guide (upload, environment, repositories, staging the 1,632 clips,
two-second pilot) are already done and are not repeated.

Everything runs as **nndl1 on AI-Lab-12**, from `/home/nndl1`. If `AI-Lab-12`
does not resolve from Windows, use `10.25.34.79` instead.

## What changes with the subset

To save time, only **300 of the 1,632 cleaned source clips** are converted and
used. They are picked by a fixed rule (no hand-picking, no model scores) and
keep their original split:

| Partition | Full v2 | This run |
|---|---:|---:|
| Source training | 1,140 | **210** |
| Source validation | 159 | **29** |
| Source test | 333 | **61** |
| NEFER target test (real events, no conversion) | 97 | **97** |
| NEFER reserve (unused) | 347 | 347 |

Each split gets its share of the 300 in proportion to its size, and each split
still contains all six classes. Within each split/class group, the clips are
ordered by a seeded hash of the clip ID (seed 476). Clips that have already
been converted go first, so finished work is reused. All new artifacts get a
`_s300` suffix, so they never mix with full-dataset files. Do not edit
`splits_v2`. The full run is still possible later with the same code.

Numbers you will see along the way:

| Check | Expected |
|---|---:|
| Clips to convert / validate | 300 |
| Event manifest rows (300 + 347 + 97) | 744 |
| Source train+validation count/feature cache | 239 |
| Evaluation count/feature cache (61 + 97) | 158 |
| Predictions (9 models × 158) | 1,422 |

---

## 1. Upload the updated code (Windows PowerShell)

Connect to the AUS VPN, then:

```powershell
scp -r "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\eventfer" nndl1@AI-Lab-12:/home/nndl1/

scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\convert_events_parallel.py" "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\validate_event_streams.py" nndl1@AI-Lab-12:/home/nndl1/scripts/

ssh nndl1@AI-Lab-12
```

This adds `eventfer/subset.py` and adds an `--include-list` option to the
converter and the validator. All commands below are **Linux**, unless marked as
PowerShell.

## 2. Stop the running full conversion

Reattach to whichever Screen session is running the converter. It is
`ekman-v2-convert` if you followed the guide.

```bash
screen -ls
screen -r ekman-v2-convert
```

Press **Ctrl+C** and wait for the prompt to return. Then check that no converter
or v2e process is still running:

```bash
pgrep -af 'convert_videos_to_events.py|convert_events_parallel.py|[v]2e.py'
```

This must print nothing. If it prints something, wait a few seconds and run it
again. Keep the Screen session open; you will reuse it in step 4.

Already-converted clips in `data/events/Ekman6_v2` are kept. If you were running
`convert_events_parallel.py`, its finished clips have completion records and are
reused. If you were running the upstream Emo-DVS wrapper, its outputs have no
completion records. Any of those clips that land in the subset are converted
once more, because an interrupted H5 can look valid.

## 3. Freeze the 300-clip subset

```bash
cd /home/nndl1
conda activate ekman-events

python -m eventfer.subset create \
  --manifest artifacts/splits_v2/manifest.csv \
  --rgb-root data/raw/videos \
  --size 300 --seed 476 \
  --prefer-converted-root data/events/Ekman6_v2 \
  --output artifacts/splits_v2_s300

python -m eventfer.subset verify --directory artifacts/splits_v2_s300
```

Expect `Source subset: {'train': 210, 'validation': 29, 'test': 61}` and a line
showing how many chosen clips are already converted. `verify` prints `PASS` and
the per-class table. The directory contains:

```text
manifest.csv                 # 300 source rows + all 444 NEFER rows, unchanged
source_relative_paths.txt    # the 300 videos to convert
class_counts.csv, subset.json, checksums.json
```

The command refuses to overwrite an existing output. Make the subset once and
keep it, because every later step depends on its hash.

## 4. Convert only the 300 clips (inside Screen)

Inside the `ekman-v2-convert` Screen session (or a new one: `screen -S ekman-s300-convert`):

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg
export TMPDIR=/home/nndl1/data/conversion_tmp
mkdir -p logs data/conversion_tmp
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python -u scripts/convert_events_parallel.py \
  --input-root data/conversion_input/Ekman6_v2 \
  --output-root data/events/Ekman6_v2 \
  --v2e-root third_party/v2e \
  --include-list artifacts/splits_v2_s300/source_relative_paths.txt \
  --workers 2 --threads-per-worker 1 \
  --dvs_aedat2 None --dvs_emulator_seed 476 \
  2>&1 | tee -a logs/ekman_s300_conversion.log
```

The first line must say `Videos=300`. Keep the input root and output root
exactly as written: the existing 1,632-clip staging tree and output tree. That
way, the completion records of clips you already converted still match and
those clips are skipped. Detach with **Ctrl+A, then D**; reattach with
`screen -r ekman-v2-convert`. To check progress from another terminal:

```bash
tail -n 5 /home/nndl1/logs/ekman_s300_conversion.log
```

If the job stops or fails, run the same command again. It resumes and only
retries unfinished clips. A failure leaves `conversion.stderr.log` in that
clip's output folder; read it before retrying more than once.

## 5. Confirm completion and event integrity

Run the conversion command from step 4 **again**. This time it acts as the
completion check: the last line must show
`Counts={'success': 0, 'skipped': 300, 'failed': 0, 'cancelled': 0}`.
Then validate the 300 event files:

```bash
python scripts/validate_event_streams.py \
  --root data/events/Ekman6_v2 \
  --expected-input-root data/conversion_input/Ekman6_v2 \
  --include-list artifacts/splits_v2_s300/source_relative_paths.txt \
  --output artifacts/ekman_s300_event_integrity.json
```

The required output is `Valid=300 Invalid=0 Missing=0`. Partial files from
clips outside the subset are ignored. If any check fails, stop and share the
report. Do not drop clips or edit the subset by hand.

## 6. Link the subset to its event files

```bash
python -m eventfer.prepare_manifest \
  --manifest artifacts/splits_v2_s300/manifest.csv \
  --source-rgb-root data/raw/videos \
  --source-events-root data/events/Ekman6_v2 \
  --source-width 336 --source-height 224 \
  --output artifacts/event_manifest_v2_s300
```

Expect `Linked 744 clips`. This step hashes every recording, including the NEFER
files, so it takes a while.

## 7. Build source count frames (choose one crop policy)

Use **one** policy for the whole experiment.

**Full-frame (use this unless you actually have face boxes):**

```bash
python -m eventfer.preprocess counts \
  --manifest artifacts/event_manifest_v2_s300/manifest.csv \
  --crop-policy full-frame --splits train validation \
  --output artifacts/counts_source_v2_s300
```

**Fixed face boxes:** only if you have real, label-blind boxes for all 239
train/validation clips. Put them in `artifacts/fixed_crops_v2_s300.csv` with the
columns `dataset,clip_id,x0,y0,x1,y1`, in event-sensor pixels with exclusive
right/bottom edges. Then run the same command with
`--crop-policy fixed-boxes --crops artifacts/fixed_crops_v2_s300.csv`. Never
invent or fill in placeholder coordinates.

The cache must contain **239 clips**. With full-frame, report the baseline as
*not* face-cropped.

## 8. Fit training-only statistics and check the preview

```bash
python -m eventfer.preprocess fit \
  --index artifacts/counts_source_v2_s300/index.csv \
  --output artifacts/source_statistics_v2_s300

python -m eventfer.preprocess preview \
  --index artifacts/counts_source_v2_s300/index.csv \
  --statistics artifacts/source_statistics_v2_s300/statistics.json \
  --output artifacts/preprocessing_preview_v2_s300.png
```

The threshold is fitted on the 210 training clips only. To view the preview,
run in **PowerShell**:

```powershell
scp nndl1@AI-Lab-12:/home/nndl1/artifacts/preprocessing_preview_v2_s300.png "$env:USERPROFILE\Downloads\"
```

Check the preview for visible facial activity in plausible time bins before you
continue.

## 9. Install the pinned DINOv2 checkout

```bash
cd /home/nndl1
if [ ! -d third_party/dinov2/.git ]; then
  git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
fi
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

The weights download on first use. Warnings about the optional xFormers package
are expected.

## 10. Cache all three source representations (GPU, inside Screen)

```bash
screen -S s300-features
```

Inside it:

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg

for MODE in baseline per_bin whole_clip; do
  CUDA_VISIBLE_DEVICES=0 python -m eventfer.features \
    --index artifacts/counts_source_v2_s300/index.csv \
    --statistics artifacts/source_statistics_v2_s300/statistics.json \
    --dinov2-repo third_party/dinov2 \
    --mode "$MODE" --device cuda --batch-size 4 \
    --output "artifacts/features_${MODE}_v2_s300" || break
done
```

Each cache must end at `[239/239] ... -> (8, 384)`. If GPU memory runs out,
use `--batch-size 2` and delete or rename the failed cache folder first,
because caches do not resume.

## 11. Tiny overfitting check (correctness only)

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventfer.train \
  --index artifacts/features_baseline_v2_s300/index.csv \
  --output runs/baseline_tiny_v2_s300 --device cuda \
  --seeds 17 --tiny-per-class 2 --dropout 0 \
  --lr 0.003 --epochs 100 --patience 100

python - <<'PY'
import json
from pathlib import Path
h = [json.loads(l) for l in Path('runs/baseline_tiny_v2_s300/seed_17/history.jsonl').read_text().splitlines()]
print('Best tiny-subset training macro-F1:', max(r['train_macro_f1'] for r in h))
PY
```

Training macro-F1 should get close to 1. If it stays low, stop and investigate.
This run is not a result. A separate three-seed baseline run is **not** needed:
the controlled experiment below already trains the baseline with all three seeds.

## 12. Create the controlled experiment plan

```bash
python -m eventfer.experiment create \
  --baseline-index artifacts/features_baseline_v2_s300/index.csv \
  --per-bin-index artifacts/features_per_bin_v2_s300/index.csv \
  --whole-clip-index artifacts/features_whole_clip_v2_s300/index.csv \
  --expected-train-counts 210 29 \
  --seeds 17 29 43 \
  --epochs 50 --patience 8 --batch-size 32 \
  --hidden-dim 128 --dropout 0.2 --lr 0.001 --weight-decay 0.0001 \
  --output runs/controlled_v2_s300
```

This checks that the three conditions use identical clips, labels, counts,
backbone, and statistics. It then writes a checksummed `plan.json`. Do not
re-upload `eventfer/` from here to the end: the code hashes are locked into the
plan.

## 13. Train the nine heads (inside Screen)

```bash
screen -S s300-train
```

Inside it:

```bash
cd /home/nndl1
conda activate ekman-events
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m eventfer.experiment train \
  --experiment runs/controlled_v2_s300 --device cuda \
  2>&1 | tee logs/controlled_v2_s300_training.log
```

To check progress from another terminal:

```bash
python -m eventfer.experiment status --experiment runs/controlled_v2_s300
```

You need nine `verified_complete` entries. Rerunning `train` skips verified
runs. It refuses to overwrite an incomplete seed folder: archive that folder
after reading its log, then rerun.

## 14. Freeze before any test scoring

For **full-frame**:

```bash
python -m eventfer.experiment freeze \
  --experiment runs/controlled_v2_s300 \
  --manifest artifacts/event_manifest_v2_s300/manifest.csv \
  --expected-test-counts 61 97
```

For **fixed boxes**, prepare `artifacts/evaluation_crops_v2_s300.csv` with
label-blind boxes for all 61 source-test and 97 target-test clips, in the same
format as before. Then add
`--evaluation-crops artifacts/evaluation_crops_v2_s300.csv` to the command.

This writes `runs/controlled_v2_s300/frozen/experiment.json`. After that,
training is closed and the evaluation rules are locked.

## 15. Get a NEFER RAW reader

The 97 target recordings are not affected by the subset. Check the Conda
interpreter first:

```bash
python -c "from metavision_core.event_io import EventsIterator; print('Conda RAW reader OK')" && SDK_PYTHON="$(command -v python)"
```

If that fails, try the system interpreter:

```bash
/usr/bin/python3 -c "from metavision_core.event_io import EventsIterator; print('System RAW reader OK')"
```

If the system interpreter works, create a small export-only environment:

```bash
/usr/bin/python3 -m venv --system-site-packages /home/nndl1/venvs/nefer-sdk
SYSTEM_NUMPY_VERSION="$(/usr/bin/python3 -c 'import numpy; print(numpy.__version__)')"
/home/nndl1/venvs/nefer-sdk/bin/python -m pip install "numpy==$SYSTEM_NUMPY_VERSION" 'h5py>=3.8,<4'
SDK_PYTHON=/home/nndl1/venvs/nefer-sdk/bin/python
```

If neither interpreter works, the lab administrator must install OpenEB. See
[Appendix A](#appendix-a-installing-openeb-administrator-only). Steps 1–14 can
all be finished before then.

## 16. Export the 97 target recordings

```bash
cd /home/nndl1
"$SDK_PYTHON" -m eventfer.export_raw \
  --manifest artifacts/event_manifest_v2_s300/manifest.csv \
  --output data/nefer_decoded_v2_s300
```

It must end with `All target RAW exports completed and validated.` If a recording
cannot be opened, share the error. Do not substitute rendered frames.

## 17. Build evaluation count frames

```bash
conda activate ekman-events

python -m eventfer.preprocess counts \
  --manifest artifacts/event_manifest_v2_s300/manifest.csv \
  --splits test target_test --crop-policy full-frame \
  --decoded-raw-root data/nefer_decoded_v2_s300 \
  --output artifacts/counts_eval_v2_s300
```

For fixed boxes, use `--crop-policy fixed-boxes --crops artifacts/evaluation_crops_v2_s300.csv`
instead. The cache must contain **158 clips**. Do not run `preprocess fit` on it.

## 18. Cache the three evaluation representations (inside Screen)

```bash
screen -S s300-eval-features
```

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg

for MODE in baseline per_bin whole_clip; do
  CUDA_VISIBLE_DEVICES=0 python -m eventfer.features \
    --index artifacts/counts_eval_v2_s300/index.csv \
    --statistics artifacts/source_statistics_v2_s300/statistics.json \
    --dinov2-repo third_party/dinov2 \
    --mode "$MODE" --splits test target_test \
    --evaluation-lock runs/controlled_v2_s300/frozen/experiment.json \
    --device cuda --batch-size 4 \
    --output "artifacts/features_eval_${MODE}_v2_s300" || break
done
```

Each cache must end at `[158/158] ... -> (8, 384)`.

## 19. Evaluate the nine locked models

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventfer.evaluate \
  --evaluation-lock runs/controlled_v2_s300/frozen/experiment.json \
  --baseline-index artifacts/features_eval_baseline_v2_s300/index.csv \
  --per-bin-index artifacts/features_eval_per_bin_v2_s300/index.csv \
  --whole-clip-index artifacts/features_eval_whole_clip_v2_s300/index.csv \
  --device cuda --batch-size 32 \
  --output results/controlled_v2_s300

cat results/controlled_v2_s300/REPORT.md

python - <<'PY'
import csv, json
from pathlib import Path
root = Path('results/controlled_v2_s300')
with (root/'predictions.csv').open(newline='') as f:
    n = sum(1 for _ in csv.DictReader(f))
assert n == 9 * 158, n
s = json.loads((root/'summary.json').read_text())
print('Predictions:', n)
print('Paired target gain (pp):', s['primary_target_gain_pp'])
print('Paired source loss (pp):', s['primary_source_loss_pp'])
print('Criterion met:', s['hypothesis']['point_estimate_criterion_met'])
PY
```

The success rule was fixed before scoring: mean NEFER macro-F1 gain of at
least 3 pp for `per_bin` over `baseline`, with a mean source-test loss of at
most 2 pp, over the paired seeds. `whole_clip` is a separate ablation.

Main outputs: `REPORT.md`/`summary.json` (means, SDs, paired effects),
`predictions.csv`, `seed_metrics.*`, `paired_differences.csv`, `per_class.csv`,
`participant_omission.csv`, `event_density.csv`, `transfer_gaps.csv`,
`error_examples.csv`, `figures/`, `checksums.json`.

## 20. Copy the results to your laptop

```bash
tar -czf results/controlled_v2_s300_report.tar.gz -C results controlled_v2_s300
```

In **PowerShell**:

```powershell
scp nndl1@AI-Lab-12:/home/nndl1/results/controlled_v2_s300_report.tar.gz "$env:USERPROFILE\Downloads\"
```

Done. Write the report from these files.

---

## Interpreting and reporting the results

- Primary metric: clip-level macro-F1 over all six classes; an undefined class
  F1 counts as 0. Accuracy is secondary. Gains and losses are **percentage
  points**.
- Error bars are the sample SD across three seeds, not confidence intervals.
  Do not claim significance.
- Confusion figures pool the three seeds for display. They are not ensembles.
- Report the outcome even if the hypothesis fails. Do not retune and rescore
  the same test sets.
- **Subset limitations to state explicitly:**
  - Only 300 of 1,632 source clips were used, chosen by the stratified hashed
    rule in `artifacts/splits_v2_s300/subset.json`, with already-converted
    clips preferred.
  - Validation has only 29 clips (about 5 per class), so checkpoint selection
    is noisy.
  - Source test has only 61 clips, so the 2 pp source-loss bound is a coarse
    check.
  - The NEFER target test is the complete 97-clip set.
- Other disclosures, unchanged from before:
  - The source is locally simulated Ekman-6 data made with the public Emo-DVS
    code, not the exact released Emo-DVS subset.
  - Duplicate cleanup.
  - The missing NEFER user-22 recording.
  - The crop policy used.
  - Simulation limited by the original video frame rate.
  - Possible identity overlap between source splits.
  - All pretrained code and weights used.

## Rules that still apply

- Run one GPU job at a time. Never point two jobs at the same output folder.
- Output folders are never overwritten. If a step fails, keep the evidence and
  use a fresh path, keeping later paths consistent.
- Source-test and NEFER scores never guide preprocessing, crops, or model
  choices.

## Appendix A: installing OpenEB (administrator only)

Check `lsb_release -ds` and `uname -m`. On Ubuntu 22.04/24.04 x86_64, an
administrator can run:

```bash
sudo apt install curl software-properties-common python3-venv
curl --fail -L https://propheseeai.jfrog.io/artifactory/api/security/keypair/prophesee-gpg/public \
  -o /tmp/prophesee-openeb-public.asc
sudo install -m 644 /tmp/prophesee-openeb-public.asc /etc/apt/trusted.gpg.d/prophesee-openeb-public.asc
sudo add-apt-repository 'https://propheseeai.jfrog.io/artifactory/openeb-debian/'
sudo apt update
sudo apt install metavision-openeb
/usr/bin/python3 -c "from metavision_core.event_io import EventsIterator; print('System reader OK')"
```

Then build the export-only environment in step 15 and continue with step 16.
See the [OpenEB package instructions](https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html).
