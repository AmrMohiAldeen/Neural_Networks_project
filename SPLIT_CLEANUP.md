# Create and verify cleaned splits_v2

The original audit succeeded, but found two source hash groups with conflicting
labels and one same-label source duplicate group. Before conversion/training,
create a cleaned derivative of the frozen metadata. Do not rerun the original
`audit_datasets.py build` command for this cleanup: use the script below.

## 1. Upload the cleanup script

Connect to the AUS VPN. Run this in your Windows PowerShell terminal. If your SSH
connection uses an IP address rather than `AI-Lab-12`, substitute that address.

```powershell
ssh nndl1@AI-Lab-12 "mkdir -p /home/nndl1/scripts"
scp "C:\Users\Pc\Desktop\AUS\AUS - Drive\senior1\COE 476\project\scripts\clean_frozen_splits.py" nndl1@AI-Lab-12:/home/nndl1/scripts/
ssh nndl1@AI-Lab-12
```

## 2. Create the cleaned version on Linux

The script uses Python's standard library only, and needs no GPU. The optional
recording verification in this command rehashes all original recordings; it may
take time because it reads approximately 26 GB. It does not decode event streams.

```bash
cd /home/nndl1
conda activate ekman-events

python scripts/clean_frozen_splits.py clean \
  --input artifacts/splits_v1 \
  --output artifacts/splits_v2 \
  --expect-source-counts 1140 159 333 \
  --verify-recordings
```

The cleanup checks the original frozen metadata checksums before using it.
It applies this rule to source clips only, without inspecting model scores:

- Conflicting labels for identical bytes: exclude every copy in the hash group.
- Same-label identical bytes: retain the shortest basename, then alphabetical
  clip ID to break ties.
- Preserve each retained row's split assignment; do not resplit or change seeds.
- Preserve every NEFER row, including its split, label, and path.
- Preserve all prior exclusions, including the known missing user-22 recording.

The original directory and all RGB/event recordings are left in place. The
script removes rows only from the new metadata version. If the destination
already exists, stop and inspect it; the script will not overwrite it.

## 3. Verify the new frozen metadata

```bash
python scripts/clean_frozen_splits.py verify \
  --directory artifacts/splits_v2

cat artifacts/splits_v2/class_counts.csv

python - <<'PY'
import json
from pathlib import Path

root = Path('artifacts/splits_v2')
audit = json.loads((root/'audit.json').read_text())
cleanup = json.loads((root/'cleanup.json').read_text())
expected = {
    'train': 1140,
    'validation': 159,
    'test': 333,
    'target_reserve': 347,
    'target_test': 97,
}
assert audit['clip_counts'] == expected, audit['clip_counts']
assert cleanup['removed_rows'] == 5, cleanup['removed_rows']
assert audit['same_split_identical_files'] == [], audit['same_split_identical_files']
print('PASS: expected counts; five exclusions; no identical-file duplicates remain.')
for row in cleanup['decisions']:
    print(row['split'], row['clip_id'], '->', row['reason'])
PY
```

Expected final counts:

| Partition | Clips |
|---|---:|
| Source training | 1,140 |
| Source validation | 159 |
| Source test | 333 |
| NEFER optional reserve | 347 |
| NEFER target test | 97 |

The source total is 1,632. Four conflicting-label rows and one same-label
duplicate row are excluded. The retained same-label copy is
`fear/8160830424_6f789562fb_700.mp4`.

## 4. Use v2 from this point onward

Use `artifacts/splits_v2/manifest.csv` as the frozen metadata input to the next
pipeline step. For example, after RGB-to-event conversion completes, use this
path with `eventfer.prepare_manifest` instead of `artifacts/splits_v1/manifest.csv`.
Do not hand-edit the manifests or checksums, and retain both versions.

Generated files in `artifacts/splits_v2/`:

```text
manifest.csv
exclusions.csv
class_counts.csv
participant_counts.csv
audit.json
cleanup.json
checksums.json
```

`cleanup.json` records the policy, each removed clip, its label/split/hash, the
retained duplicate where applicable, and the parent version's checksums.

Successful verification freezes the cleaned metadata only. Native event
timestamps, coordinate bounds, polarity, crop policy, and conversion completeness
still need the separate event preprocessing/integrity checks. This is locally
simulated Ekman-6 source data, rather than an assertion that the exact released
Emo-DVS benchmark has been reproduced.
