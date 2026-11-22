> **Superseded (2026-11-23):** steps 6 onward are replaced by [SUBSET300_GUIDE.md](SUBSET300_GUIDE.md), which uses a 300-clip source subset and runs to the end. Kept for reference only.

# Steps 7 and 8: full controlled experiment and evaluation

This guide is for `/home/nndl1` on the Linux lab machine. It follows
`POST_AUDIT_GUIDE.md`; complete conversion, preprocessing, previews, and the tiny
training check there first. No full experimental results have been produced by
the local software tests.

## What this will run

| Condition | Representation | Seeds |
|---|---|---|
| `baseline` | Original counts, fixed log compression | 17, 29, 43 |
| `per_bin` | Bounded per-bin scaling before compression | 17, 29, 43 |
| `whole_clip` | One scale from mean bin activity; ablation | 17, 29, 43 |

There are **nine head-training runs**. The backbone, raw count cache, source
statistics, clip order, class weights, architecture, budget, and paired initial
head weights are controlled. Validation selects checkpoints; source test and
NEFER target-test scores are not used for training or tuning.

The primary predeclared criterion is a mean NEFER macro-F1 improvement of at
least **3 percentage points**, with a mean source-test macro-F1 loss of at most
**2 points**, for `per_bin` versus `baseline` over paired seeds. A negative result
is valid. The ablation is a separate comparison, not a replacement selected after
seeing test results.

## 1. Confirm the prerequisites

These artifacts must exist and be complete:

```text
artifacts/splits_v2/                       # Verified cleaned split metadata
artifacts/event_manifest_v2/               # Full converted-source + original NEFER paths
artifacts/counts_source_v2/                # 1,140 train + 159 validation clips
artifacts/source_statistics_v2/            # Source-training-only threshold
artifacts/features_baseline_v2/            # Full 1,299-clip baseline embedding cache
```

Source conversion/integrity must already have passed. Frozen counts remain
1,140 train / 159 validation / 333 source test, with 97 NEFER target-test clips
and 347 unused reserve clips. The reserve never enters this main experiment.

Existing step-6 baseline checkpoints remain development evidence. This guide
trains a fresh, consistently recorded set of nine heads, reusing cached baseline
embeddings so DINOv2 does not need to run again for that source condition.

## 2. Upload the updated code from Windows

Connect to the VPN. In **Windows PowerShell**, substitute your normal SSH IP if
`AI-Lab-12` does not resolve:

```powershell
 nndl1@10.25.34.79:/home/nndl1
scp -r "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\eventfer" nndl1@10.25.34.79:/home/nndl1/

scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\requirements-baseline.txt" nndl1@10.25.34.79:/home/nndl1/

ssh nndl1@AI-Lab-12
```

All subsequent commands are **Linux commands**, except the final download step.
Upload the complete updated package before creating the experiment. The code
hashes are locked with the plan; do not update the package partway through a run.

```bash
cd /home/nndl1
conda activate ekman-events
python -m pip install -r requirements-baseline.txt
export MPLBACKEND=Agg
mkdir -p logs

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__); print(torch.cuda.get_device_name(0))"
python -m eventfer.experiment --help
```

## 3. Cache the two changed source representations

Use the exact count cache, training-only statistics, and pinned DINOv2 checkout
from the baseline. The encoding math is unchanged by this update; an existing
baseline cache can be reused if its numerical recipe and PyTorch version match.
The experiment creator checks these properties rather than assuming compatibility.

Start a persistent session:

```bash
screen -S controlled-source-features
```

Inside Screen:

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg

CUDA_VISIBLE_DEVICES=0 python -m eventfer.features \
  --index artifacts/counts_source_v2/index.csv \
  --statistics artifacts/source_statistics_v2/statistics.json \
  --dinov2-repo third_party/dinov2 \
  --mode per_bin --device cuda --batch-size 4 \
  --output artifacts/features_per_bin_v2
```

After that command succeeds:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventfer.features \
  --index artifacts/counts_source_v2/index.csv \
  --statistics artifacts/source_statistics_v2/statistics.json \
  --dinov2-repo third_party/dinov2 \
  --mode whole_clip --device cuda --batch-size 4 \
  --output artifacts/features_whole_clip_v2
```

Each cache must finish with 1,299 arrays of shape `(8,384)`. Detach with Ctrl+A,
then D; reconnect with `screen -r controlled-source-features`. If extraction
fails, diagnose it and use a fresh output directory; incomplete caches do not
resume. If your baseline was made with different settings/library versions, stop
at the compatibility error and reconcile the recipes before training.

## 4. Create the controlled source experiment plan

After all three source caches are complete:

```bash
cd /home/nndl1
conda activate ekman-events

python -m eventfer.experiment create \
  --baseline-index artifacts/features_baseline_v2/index.csv \
  --per-bin-index artifacts/features_per_bin_v2/index.csv \
  --whole-clip-index artifacts/features_whole_clip_v2/index.csv \
  --expected-train-counts 1140 159 \
  --seeds 17 29 43 \
  --epochs 50 --patience 8 --batch-size 32 \
  --hidden-dim 128 --dropout 0.2 --lr 0.001 --weight-decay 0.0001 \
  --output runs/controlled_v2
```

This creates checksummed `plan.json`. It validates identical source identities,
labels, split assignments, row order, unscaled count hashes, backbone weights,
normalization, and scaling statistics across conditions. It also fixes the
event-rate diagnostic bands using source training only, before target scoring.

Use a new experiment directory if the specified one already exists. Do not
manually edit a plan or its checksum file.

## 5. Run the nine source-only training jobs sequentially

```bash
screen -S controlled-train
```

Inside it:

```bash
cd /home/nndl1
conda activate ekman-events
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python -u -m eventfer.experiment train \
  --experiment runs/controlled_v2 --device cuda \
  2>&1 | tee logs/controlled_v2_training.log
```

One command runs all three conditions and seeds. It never loads test feature
arrays. Each seed selects the first best source-validation macro-F1 checkpoint.

Inspect progress from another SSH terminal:

```bash
cd /home/nndl1
conda activate ekman-events
python -m eventfer.experiment status --experiment runs/controlled_v2
tail -n 20 logs/controlled_v2_training.log
```

Require nine `verified_complete` entries. The same `train` command can be rerun
before the evaluation freeze: it verifies and skips completed runs. It refuses
to overwrite an incomplete seed directory. If a job failed, retain its logs and
diagnose it before archiving that incomplete directory and retrying. Do not run
two training coordinators at once.

## 6. Freeze checkpoints and evaluation rules before scoring

All nine runs must be complete. Use the crop policy already chosen for source
training; do not change normalization/binning based on target behavior.

### If the source used full-frame fallback

```bash
python -m eventfer.experiment freeze \
  --experiment runs/controlled_v2 \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --expected-test-counts 333 97
```

### If the source used fixed face boxes

Prepare `artifacts/evaluation_crops_v2.csv` with one label-blind fixed box for
each of the 333 source-test and 97 NEFER target-test clips. Columns are:

```text
dataset,clip_id,x0,y0,x1,y1
```

Use the actual event-sensor coordinates, exclusive right/bottom edges, and exact
manifest clip IDs. Target boxes can be supplied dataset annotations; do not
choose boxes using emotion labels or test predictions. Unlike the training crop
file, this file covers evaluation clips. Its distinct SHA256 is explicitly locked.

```bash
python -m eventfer.experiment freeze \
  --experiment runs/controlled_v2 \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --evaluation-crops artifacts/evaluation_crops_v2.csv \
  --expected-test-counts 333 97
```

Run only the branch matching your source crop policy. The output lock is:

```text
runs/controlled_v2/frozen/experiment.json
```

Freeze verifies matching initial head weights for paired seeds, identical class
weights/architecture/source selection, source-validation checkpoint selection,
and checkpoint/config/history hashes. It records the test clip identities,
crop policy, metrics, hypothesis margins, and code hashes. Training is then closed
for this experiment. Keep the lock unchanged after seeing test results.

## 7. Check the NEFER RAW decoder

The original RAW recordings need the compatible Metavision/OpenEB reader.
Check the current Conda interpreter:

```bash
python -c "from metavision_core.event_io import EventsIterator; print('Conda RAW reader OK')"
```

**If this works**, select it for the one-time export:

```bash
SDK_PYTHON="$(command -v python)"
```

**If it fails**, check the system interpreter:

```bash
lsb_release -ds
uname -m
/usr/bin/python3 -c "from metavision_core.event_io import EventsIterator; print('System RAW reader OK')"
```

If system Python works but Conda does not, follow the isolated SDK-venv steps in
Appendix A. If both fail, follow Appendix A's OpenEB installation branch with
your lab administrator. RAW decoding is an external dependency; this repository
does not bundle the SDK or camera plugin. You can finish all source training
before this dependency is available, but cannot complete target evaluation.

## 8. Export the 97 target recordings to verified canonical H5

Use the interpreter selected above or by Appendix A:

```bash
cd /home/nndl1

"$SDK_PYTHON" -m eventfer.export_raw \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --output data/nefer_decoded_v2
```

The exporter processes `target_test` only, not the 347 reserve clips. It obtains
sensor size from the decoder, preserves integer microsecond timestamps and
binary polarity, and checks ordering/bounds. H5 uses int64 to avoid uint32
timestamp overflow. Each output has a sidecar binding it to the original RAW
SHA256. Verified complete exports are skipped on rerun.

Success ends with `All target RAW exports completed and validated.` If the
reader cannot open a recording, share the error rather than substituting rendered
TBR frames or treating RAW as a NumPy array.

## 9. Build locked source-test and target-test count frames

Return to the normal PyTorch environment:

```bash
cd /home/nndl1
conda activate ekman-events
```

For **full-frame** source training:

```bash
python -m eventfer.preprocess counts \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --splits test target_test --crop-policy full-frame \
  --decoded-raw-root data/nefer_decoded_v2 \
  --output artifacts/counts_eval_v2
```

For **fixed-boxes** source training, use this command instead:

```bash
python -m eventfer.preprocess counts \
  --manifest artifacts/event_manifest_v2/manifest.csv \
  --splits test target_test --crop-policy fixed-boxes \
  --crops artifacts/evaluation_crops_v2.csv \
  --decoded-raw-root data/nefer_decoded_v2 \
  --output artifacts/counts_eval_v2
```

Require **430 clips**: 333 simulated source-test + 97 real target-test.
This uses the verified canonical H5 for decoding while preserving the original
RAW paths/hashes in the manifest/index. It does not change the training manifest
or refit the threshold. Do not run `preprocess fit` on this cache.

## 10. Cache all three locked evaluation representations

```bash
screen -S controlled-eval-features
```

Inside it:

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg

for MODE in baseline per_bin whole_clip; do
  CUDA_VISIBLE_DEVICES=0 python -m eventfer.features \
    --index artifacts/counts_eval_v2/index.csv \
    --statistics artifacts/source_statistics_v2/statistics.json \
    --dinov2-repo third_party/dinov2 \
    --mode "$MODE" --splits test target_test \
    --evaluation-lock runs/controlled_v2/frozen/experiment.json \
    --device cuda --batch-size 4 \
    --output "artifacts/features_eval_${MODE}_v2" || break
done
```

Every representation must finish successfully with 430 `(8,384)` arrays. If any
command fails, stop before evaluation. Feature caches do not resume partial
builds; diagnose the failure and use fresh paths. The extractor verifies the
locked source statistics, crop file, count recipe, backbone, and implementation
before accepting evaluation artifacts.

## 11. Evaluate the nine locked models

```bash
cd /home/nndl1
conda activate ekman-events
export MPLBACKEND=Agg

CUDA_VISIBLE_DEVICES=0 python -m eventfer.evaluate \
  --evaluation-lock runs/controlled_v2/frozen/experiment.json \
  --baseline-index artifacts/features_eval_baseline_v2/index.csv \
  --per-bin-index artifacts/features_eval_per_bin_v2/index.csv \
  --whole-clip-index artifacts/features_eval_whole_clip_v2/index.csv \
  --device cuda --batch-size 32 \
  --output results/controlled_v2
```

Evaluation refuses changed checkpoints/configs/history, changed frozen code,
wrong representations/statistics/crops, unequal count inputs across conditions,
and omitted/added/relabeled test clips. It performs inference only and does not
change weights or select new epochs. Output directories are never overwritten.

## 12. Check the generated results

```bash
cat results/controlled_v2/REPORT.md

python - <<'PY'
import csv
import json
from pathlib import Path
root = Path('results/controlled_v2')
with (root/'predictions.csv').open(newline='') as f:
    predictions = list(csv.DictReader(f))
assert len(predictions) == 3870, len(predictions)
summary = json.loads((root/'summary.json').read_text())
print('Predictions:', len(predictions), '= 9 models x 430 clips')
print('Paired primary target gain (pp):', summary['primary_target_gain_pp'])
print('Paired primary source loss (pp):', summary['primary_source_loss_pp'])
print('Predeclared point-estimate criterion met:', summary['hypothesis']['point_estimate_criterion_met'])
PY
```

Main outputs:

| File | Meaning |
|---|---|
| `REPORT.md`, `summary.json` | Mean/sample SD, paired effects, hypothesis outcome |
| `predictions.csv` | Clip predictions and softmax outputs for all nine models |
| `seed_metrics.csv`, `seed_metrics.json` | Each seed's primary metrics; JSON includes full per-class/confusion data |
| `paired_differences.csv` | Same-seed differences for main comparison and ablation |
| `per_class.csv` | Per-class precision, recall, F1, support, and seed variability |
| `participant_omission.csv` | Leave-one-participant-out paired effects; missing-class flags |
| `event_density.csv` | Scores by unscaled event-rate bands fitted on source training only |
| `transfer_gaps.csv` | Descriptive source-minus-target macro-F1 gaps |
| `expected_label_diagnostic.csv` | Alternate expected-label scores on eligible primary-test clips only |
| `error_examples.csv` | Up to five highest-softmax errors per mode/seed/split, with deterministic tie-breaking |
| `figures/` | Overall F1, confusion matrices, learning curves, participant omission, event-rate analysis |
| `checksums.json` | Hashes of completed result tables, reports, and figures |

If no eligible expected labels exist, that diagnostic CSV has a header but no
rows; the primary reported-label evaluation is unaffected.

## 13. Interpret results using the frozen rules

- The primary metric is clip-level macro-F1, always averaging six classes.
  Undefined class F1 is zero. Accuracy is secondary.
- Gains and losses are **percentage points**, not relative percentages.
- The criterion checks the paired mean target gain and paired mean source loss.
  It does not select a seed or use the ablation as the main method.
- Error bars are sample SD across three seeds, not confidence intervals or proof
  of statistical significance. The small six-participant target limits inference.
- Participant-omission rows flag absent classes and participants with zero eligible
  clips. The six-class denominator stays fixed, so interpret those rows carefully.
- Density subgroup rows also flag absent classes. Empty bands remain empty rather
  than becoming artificial zero-performance groups.
- Confusion figures pool seed predictions for display. They are **not ensemble
  predictions**; each clip appears once per seed. The JSON retains each seed's matrix.
- Expected-label diagnostics and qualitative errors explain behavior; they do not
  change the primary labels, models, thresholds, crop policy, or success rule.
- A source/target score difference includes recording, annotation, and dataset
  shifts; do not attribute it entirely to sensor simulation.

After scoring, report the outcome even if the hypothesis fails. Do not adjust the
model or threshold and re-present the same target set as an untouched final test.
Preserve all experiment versions and diagnose genuine scoring bugs transparently.

## 14. Copy report artifacts to your laptop

On Linux:

```bash
tar -czf results/controlled_v2_report.tar.gz -C results controlled_v2
```

In **Windows PowerShell**:

```powershell
scp nndl1@AI-Lab-12:/home/nndl1/results/controlled_v2_report.tar.gz "$env:USERPROFILE\Downloads\controlled_v2_report.tar.gz"
```

This copies the result package, not raw recordings or embedding caches. Keep
prediction metadata and any later qualitative examples within dataset-use terms.

scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\convert_events_parallel.py" nndl1@10.25.34.79:/home/nndl1/scripts/

## Appendix A: if the RAW-reader import does not work

The SDK Python bindings depend on OS and Python version. Current official
OpenEB packages target Ubuntu 22.04/24.04 amd64, with different supported Python
versions. A system reader working under Ubuntu's default Python does not imply
that the Python 3.11 Conda environment can import it. See the
[official OpenEB package instructions](https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html)
and [SDK version/platform compatibility](https://docs.prophesee.ai/stable/installation/index.html).

### A1. System Python already imports the reader

Create an isolated export environment that can see system SDK packages. These
commands do not install PyTorch or change the main Conda environment:

```bash
cd /home/nndl1
/usr/bin/python3 -m venv --system-site-packages /home/nndl1/venvs/nefer-sdk
SYSTEM_NUMPY_VERSION="$(/usr/bin/python3 -c 'import numpy; print(numpy.__version__)')"
/home/nndl1/venvs/nefer-sdk/bin/python -m pip install "numpy==$SYSTEM_NUMPY_VERSION" 'h5py>=3.8,<4'
/home/nndl1/venvs/nefer-sdk/bin/python -c "from metavision_core.event_io import EventsIterator; import h5py; print('Export environment OK')"
SDK_PYTHON=/home/nndl1/venvs/nefer-sdk/bin/python
```

Return to step 8. Use this interpreter for RAW export only; use `ekman-events`
for DINOv2 and GRU commands. The NumPy pin retains the version used by the working
system reader, rather than upgrading its binary-extension dependency implicitly.

### A2. Neither interpreter imports the reader

Check `lsb_release -ds` and `uname -m`. On a supported Ubuntu 22.04/24.04 x86_64
machine, ask the lab administrator to install the compatible OpenEB package and
recording/camera plugin using the official instructions. The essential package
setup, for an administrator authorized to modify this shared machine, is:

```bash
sudo apt install curl software-properties-common python3-venv
curl --fail -L \
  https://propheseeai.jfrog.io/artifactory/api/security/keypair/prophesee-gpg/public \
  -o /tmp/prophesee-openeb-public.asc
sudo install -m 644 /tmp/prophesee-openeb-public.asc /etc/apt/trusted.gpg.d/prophesee-openeb-public.asc
sudo add-apt-repository 'https://propheseeai.jfrog.io/artifactory/openeb-debian/'
sudo apt update
sudo apt install metavision-openeb

/usr/bin/python3 -c "from metavision_core.event_io import EventsIterator; print('System reader OK')"
```

These are conditional installation commands, not commands to run again over a
working lab SDK. Older camera/recording support may require a compatible SDK
version/plugin rather than the newest package; retain any decoder error for the
administrator. For other OS versions, use the official source-build route rather
than substituting an unsupported package repository.

After the system import succeeds, do A1 and return to step 8. If import succeeds
but the real recording cannot be decoded, stop and share the actual error. The
local tests validated the export schema/bridge with synthetic reader data; they
cannot establish compatibility with your lab's SDK and physical file encoding.

## Parallel execution

Use the single Quadro GPU sequentially for feature extraction and head training.
Report writing or SDK setup can run alongside source training if host CPU/RAM/I/O
permit. Finish all nine runs and freeze before scoring. Do not run two jobs into
the same cache, run, export, or results directory.
