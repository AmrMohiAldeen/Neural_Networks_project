"""Cache frozen pretrained DINOv2 ViT-S/14 embeddings for each temporal bin."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .common import csv_write, digest, fresh_directory, json_write, lock_directory, read_rows, uid, verify_lock
from .events import represent
from .preprocess import load_counts


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def encode(model, frames, device, batch_size):
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    result = []
    for i in range(0, len(frames), batch_size):
        batch = torch.as_tensor(frames[i:i + batch_size], dtype=torch.float32, device=device)
        result.append(model((batch - mean) / std).float().cpu().numpy())
    embeddings = np.concatenate(result)
    if embeddings.shape != (len(frames), 384) or not np.isfinite(embeddings).all():
        raise ValueError(f'Invalid DINOv2 ViT-S/14 embeddings: {embeddings.shape}')
    return embeddings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--dinov2-repo', type=Path, required=True)
    p.add_argument('--mode', choices=['baseline', 'per_bin', 'whole_clip'], default='baseline')
    p.add_argument('--splits', nargs='+', default=['train', 'validation'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4, help='Number of bin images per backbone forward pass')
    p.add_argument('--evaluation-lock', type=Path, help='Required for locked test feature extraction')
    args = p.parse_args()
    verify_lock(args.index)
    verify_lock(args.statistics)
    stats = json.loads(args.statistics.read_text())
    config = json.loads((args.index.parent / 'config.json').read_text())
    training_config = stats['counts_config']
    evaluation = bool(set(args.splits) & {'test', 'target_test'})
    frozen = None
    if evaluation:
        if not args.evaluation_lock:
            raise ValueError('Test feature extraction requires --evaluation-lock from experiment freeze')
        from .protocol import load_frozen, check_evaluation_counts
        frozen = load_frozen(args.evaluation_lock)
        check_evaluation_counts(config, frozen)
    for key in ('manifest_sha256', 'bins', 'size', 'crop_policy', 'implementation_sha256'):
        if config[key] != training_config[key]:
            raise ValueError(f'Count preprocessing differs from source statistics: {key}')
    if not evaluation and config['crops_sha256'] != training_config['crops_sha256']:
        raise ValueError('Source crop annotations differ from source statistics')
    if config['size'] != 224 or args.batch_size < 1:
        raise ValueError('Use 224x224 grids for this baseline and a positive batch size')
    rows = [r for r in read_rows(args.index) if r['split'] in args.splits]
    if not rows or set(args.splits) - {r['split'] for r in rows}:
        raise ValueError('Requested splits absent from cache')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; fix GPU setup or explicitly use --device cpu')
    repo = args.dinov2_repo.resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty.strip():
        raise ValueError('DINOv2 checkout has tracked modifications; use a clean, recorded revision')
    model = torch.hub.load(str(repo), 'dinov2_vits14', source='local', pretrained=True)
    model.eval().requires_grad_(False).to(device)
    weights_sha256 = state_digest(model)
    output = fresh_directory(args.output)
    (output / 'features').mkdir()
    for i, row in enumerate(rows, 1):
        counts = load_counts(row)
        frames, scales = represent(counts, args.mode, stats['reference_count'])
        embeddings = encode(model, frames, device, args.batch_size)
        path = output / 'features' / f'{uid(row)}.npy'
        np.save(path, embeddings, allow_pickle=False)
        row.update(features_path=str(path), features_sha256=digest(path))
        print(f'[{i}/{len(rows)}] {row["clip_id"]} -> {embeddings.shape}', flush=True)
    csv_write(output / 'index.csv', rows)
    feature_config = {
        'mode': args.mode, 'index_sha256': digest(args.index), 'statistics_sha256': digest(args.statistics),
        'statistics': stats, 'count_config': config, 'splits': args.splits,
        'backbone': 'dinov2_vits14', 'backbone_commit': commit, 'weights_sha256': weights_sha256,
        'embedding_dim': 384, 'bins': config['bins'], 'torch_version': torch.__version__,
        'representation_sha256': digest(Path(__file__).with_name('events.py')),
        'extractor_sha256': digest(Path(__file__)),
    }
    if frozen:
        from .protocol import check_evaluation_features
        check_evaluation_features(feature_config, frozen, args.mode)
        feature_config['evaluation_lock_sha256'] = digest(args.evaluation_lock)
    json_write(output / 'config.json', feature_config)
    lock_directory(output, ['index.csv', 'config.json'])


if __name__ == '__main__':
    main()
