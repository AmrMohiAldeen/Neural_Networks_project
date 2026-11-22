# Dataset preparation: historical setup notes

**Current guide: [POST_AUDIT_GUIDE.md](POST_AUDIT_GUIDE.md).** The datasets have
been downloaded and extracted, and the cleaned `splits_v2` metadata has been
verified. Your actual Linux project root is `/home/nndl1`.

The sections below retain the earlier setup/download instructions for reference.
They use historical example paths under `~/coe476-project`; do not rerun these
completed steps as your current workflow. The source split counts are now
1,140 train / 159 validation / 333 test, with 347 NEFER reserve and 97 target-test
clips. Event integrity and model development remain pending on the lab data.

## 1. Finish the Ekman RGB download

If the downloader is already running, leave it running. Do not start a second
copy writing to the same directory.

Otherwise, run:

```bash
cd ~/coe476-project
python -u scripts/download_ekman_rgb.py --workers 4
```

The default destination is:

```text
data/raw/Ekman6/
├── anger/
├── disgust/
├── fear/
├── joy/
├── sadness/
└── surprise/
```

This script downloads the videos directly from the archive and saves them as
individual files. **Do not download or extract `videos.zip` again.**

Rerunning the command checks existing completed videos and skips verified files.
The successful final message is:

```text
All 1637 RGB videos downloaded and CRC32-verified.
```

For a new long-running download, you can first start a persistent session:

```bash
screen -S ekman-download
```

Run the downloader inside that session. Press **Ctrl+A, then D** to detach.
Reconnect with:

```bash
screen -r ekman-download
```

Starting Screen does not move an already-running download into it.

## 2. Download NEFER in parallel, only if needed

You previously said NEFER was installed. **Skip this step if its download has
already completed.** Otherwise, open another SSH terminal and run:

```bash
cd ~/coe476-project
python -m pip install 'gdown>=5.2,<6'
screen -S nefer-download
```

Inside Screen:

```bash
cd ~/coe476-project
python -u scripts/download_drive_folder.py
```

The default destination is:

```text
data/google_drive_dataset/
```

Press **Ctrl+A, then D** to detach. Reconnect with:

```bash
screen -r nefer-download
```

The two downloads can run simultaneously, but they share network bandwidth.
The Drive script does not extract downloaded archives.

You do not need the large pretrained `.pkl` feature files for our event-only
model. Source emotion labels can be taken from the RGB class directories.

## 3. Prepare the conversion environment while downloading

This can run in another terminal while either dataset downloads. If the Conda
environment or repositories already exist from the previous instructions, reuse
them and skip their creation or clone commands.

```bash
cd ~/coe476-project

conda create -n ekman-events -c conda-forge python=3.11 tk ffmpeg screen -y
conda activate ekman-events

mkdir -p third_party
git clone https://github.com/Typistchen/Emo-DVS.git third_party/Emo-DVS
git -C third_party/Emo-DVS checkout 88cb52a70729a0731bca47f0cb7d2992ac9ceccd

git clone --branch v2e-1.7.0 --depth 1 \
  https://github.com/SensorsINI/v2e.git third_party/v2e

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  numpy==1.26.4 opencv-python-headless==4.10.0.84 numba==0.61.2 \
  'h5py>=3.8,<4' 'matplotlib>=3.7,<3.10' 'pandas>=2,<3' \
  pillow tqdm argcomplete engineering-notation easygui screeninfo plyer

export MPLBACKEND=Agg
```

Check that PyTorch can actually use the GPU:

```bash
nvidia-smi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"

python third_party/v2e/v2e.py --help
```

The Python check should show the NVIDIA GPU and a CUDA tensor. If it fails,
resolve that before conversion. The GPU appearing in `lspci` alone does not
confirm CUDA availability.

This uses a pinned older v2e release with Python 3.11. It installs the dependencies
needed for headless event conversion rather than the full multimodal training
stack. See the [Emo-DVS repository](https://github.com/Typistchen/Emo-DVS) and
[PyTorch installation commands](https://pytorch.org/get-started/previous-versions/).

## 4. Inventory both completed datasets

Wait until the downloads finish. Make sure `audit_datasets.py` has been copied
into the lab machine's `scripts/` directory.

If needed, upload it from your laptop's PowerShell terminal. Replace the username
and host with your lab login details:

```powershell
scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\audit_datasets.py" YOUR_USERNAME@LAB_HOST:coe476-project/scripts/
```

On the Linux machine:

```bash
cd ~/coe476-project

python scripts/audit_datasets.py inventory \
  --roots data/raw/Ekman6 data/google_drive_dataset \
  --output artifacts/data_inventory.json

cat artifacts/data_inventory.json
```

If you specified another download destination, replace the corresponding path.
Inventory lists file types, example paths, annotation previews, and ZIP contents.
It does not extract archives or modify recordings.

## 5. Share the inventory and complete the audit

Send Codex the contents of `artifacts/data_inventory.json`, or attach the file.
The actual file layout is needed to finish these steps accurately:

1. Identify and extract the required NEFER archives.
2. Create source metadata using the six RGB class directories.
3. Check labels, duplicate recordings, original-video groups, and known
   participant groups.
4. Create class and participant count tables.
5. Freeze source train/validation/test manifests and the NEFER target test set.

The shared classes are anger, disgust, fear, happiness, sadness, and surprise;
`joy` maps to `happiness`. NEFER's official held-out participants
`03, 07, 17, 19, 27, 28` are reserved for target testing. Target test performance
must not guide preprocessing choices or model selection.

**Do not consider the splits frozen just because the audit script exists.**
The generated manifests and audit results must first be checked against your
actual data. Missing identity metadata and undetected re-encoded duplicates
must be documented as limitations.

## 6. Convert after the source metadata splits are frozen

After the audit, run a short conversion pilot and inspect its event output.
Then launch full conversion in Screen. The conversion instructions are in
`README.md`, under **RGB-to-event conversion on the Linux lab machine**.

For a direct-download setup, the original default input directory was:

```text
data/raw/Ekman6
```

Use it instead of `data/raw/videos` wherever the archive-extraction example in
the README uses that path. Each converted clip must retain its original split
assignment.

Your subsequent inventory confirmed `/home/nndl1/data/raw/videos` instead.
The current guide stages only the 1,632 retained source clips from that directory
before conversion; follow `POST_AUDIT_GUIDE.md` rather than this historical example.

Generated events are simulated Ekman-6 data produced using the public Emo-DVS
conversion code. They are not automatically the exact released Emo-DVS benchmark.

## What can run in parallel?

| Task | Alongside dataset downloads? |
|---|---|
| NEFER download and Ekman RGB download | Yes; bandwidth is shared |
| Cloning the conversion repositories | Yes |
| Creating the Conda environment and installing dependencies | Yes |
| Checking GPU availability | Yes |
| Final dataset audit and freezing splits | Wait for complete datasets |
| Full RGB-to-event conversion | Wait for the source audit and frozen splits |

Do not run two downloaders writing the same dataset directory, or two conversions
writing the same output directory.

## Code for preprocessing and the baseline

Steps 5 and 6 of the original project plan are implemented in `eventfer/`.
After the data audit, split freeze, and event conversion, follow the commands in
`README.md` under **Steps 5 and 6: preprocessing and the frozen-backbone baseline**.
That section includes the PowerShell upload commands and Linux execution order:

1. Install baseline dependencies and the pinned DINOv2 repository.
2. Link the frozen RGB manifest to converted event streams.
3. Build source train/validation count caches using an explicit crop policy.
4. Fit the event-count threshold from source training clips only.
5. Inspect a preprocessing preview.
6. Cache frozen DINOv2 embeddings.
7. Check GRU training on a tiny source subset.
8. Train the full source baseline with seeds 17, 29, and 43.

These commands do not evaluate the source test or NEFER target test. Software
smoke tests are not experimental results and do not complete the dataset audit.


