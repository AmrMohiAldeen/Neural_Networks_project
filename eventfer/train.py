"""Train only a GRU and linear head; select checkpoints on source validation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import CLASSES, digest, fresh_directory, json_write, read_rows, verify_lock


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class FeatureDataset(Dataset):
    def __init__(self, rows, bins, dim):
        self.rows = rows
        self.values = []
        for row in rows:
            if digest(row['features_path']) != row['features_sha256']:
                raise ValueError(f'Feature cache changed: {row["clip_id"]}')
            features = np.load(row['features_path'], allow_pickle=False)
            if features.shape != (bins, dim) or not np.isfinite(features).all():
                raise ValueError(f'Invalid features for {row["clip_id"]}')
            self.values.append(torch.tensor(features, dtype=torch.float32))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.values[i], int(self.rows[i]['label_id'])


class TemporalHead(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, dropout=.2, classes=6):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, features):
        sequence, _ = self.gru(features)
        return self.classifier(self.dropout(sequence[:, -1]))


def initial_state_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def metrics(labels, predictions):
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)), 1)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(6), where=(support + predicted) > 0)
    return {'macro_f1': float(f1.mean()), 'accuracy': float(tp.sum() / max(1, matrix.sum())),
            'confusion_matrix': matrix.tolist(), 'precision': precision.tolist(), 'recall': recall.tolist(),
            'per_class_f1': f1.tolist(), 'support': support.tolist(), 'class_order': CLASSES}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for features, target in loader:
        predictions.extend(model(features.to(device)).argmax(1).cpu().tolist())
        labels.extend(target.tolist())
    return metrics(labels, predictions)


def train_one(args, rows, feature_config, seed):
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu explicitly for a small smoke run')
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'validation']
    if not train_rows or not val_rows:
        raise ValueError('Source train and validation caches are both required')
    counts = Counter(int(r['label_id']) for r in train_rows)
    if set(counts) != set(range(6)):
        raise ValueError('All six classes required in source train')
    # Optional correctness check: overfit a stratified tiny SOURCE TRAIN subset.
    if args.tiny_per_class:
        selected = []
        for cls in range(6):
            selected.extend([r for r in train_rows if int(r['label_id']) == cls][:args.tiny_per_class])
        train_rows = selected
        counts = Counter(int(r['label_id']) for r in train_rows)
    cfg = {'input_dim': feature_config['embedding_dim'], 'hidden_dim': args.hidden_dim,
           'dropout': args.dropout, 'classes': 6}
    model = TemporalHead(**cfg).to(device)
    initial_sha256 = initial_state_hash(model)
    train_data = FeatureDataset(train_rows, feature_config['bins'], cfg['input_dim'])
    val_data = FeatureDataset(val_rows, feature_config['bins'], cfg['input_dim'])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_eval_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    weights = torch.tensor([len(train_rows) / (6 * counts[i]) for i in range(6)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = fresh_directory(args.output / f'seed_{seed}')
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(seed=seed, class_weights=weights.cpu().tolist(), model=cfg,
                      initial_state_sha256=initial_sha256,
                      feature_config=feature_config, feature_index_sha256=digest(args.index),
                      torch_version=str(torch.__version__), device=str(device),
                      trainer_sha256=digest(Path(__file__)),
                      gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                      selected_training_clips=[r['clip_id'] for r in train_rows])
    json_write(output / 'config.json', run_config)
    best, stale, best_epoch = -1., 0, 0
    with (output / 'history.jsonl').open('w', encoding='utf-8') as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, n = 0., 0
            for features, target in loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features), target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += float(loss.detach()) * len(target)
                n += len(target)
            result = evaluate(model, val_loader, device)
            train_result = evaluate(model, train_eval_loader, device)
            record = {'epoch': epoch, 'train_loss': total_loss / n,
                      'train_macro_f1': train_result['macro_f1'], 'validation': result}
            history.write(json.dumps(record) + '\n')
            history.flush()
            print(f'seed={seed} epoch={epoch} loss={total_loss/n:.4f} train_F1={train_result["macro_f1"]:.4f} val_F1={result["macro_f1"]:.4f}', flush=True)
            if result['macro_f1'] > best:
                best, stale, best_epoch = result['macro_f1'], 0, epoch
                checkpoint = output / 'best.pt.tmp'
                torch.save({'state_dict': model.state_dict(), 'model_config': cfg,
                            'epoch': epoch, 'seed': seed, 'validation': result,
                            'feature_index_sha256': digest(args.index), 'feature_config': feature_config}, checkpoint)
                checkpoint.replace(output / 'best.pt')
                json_write(output / 'best_validation.json', result)
            else:
                stale += 1
            if stale >= args.patience:
                break
    summary = {'seed': seed, 'best_epoch': best_epoch, 'source_validation_macro_f1': best,
               'tiny_correctness_run': bool(args.tiny_per_class), 'test_evaluated': False,
               'checkpoint_sha256': digest(output / 'best.pt')}
    summary['config_sha256'] = digest(output / 'config.json')
    json_write(output / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.0001)
    p.add_argument('--tiny-per-class', type=int, default=0)
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1 or args.tiny_per_class < 0:
        p.error('Counts must be positive (tiny-per-class may be zero)')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        p.error('Use distinct seeds in [0, 2**32)')
    verify_lock(args.index)
    rows = read_rows(args.index)
    # Do not load target/test arrays even if present in the index.
    rows = [r for r in rows if r['split'] in {'train', 'validation'}]
    config = json.loads((args.index.parent / 'config.json').read_text())
    args.output = fresh_directory(args.output)
    summaries = [train_one(args, rows, config, seed) for seed in args.seeds]
    json_write(args.output / 'validation_summary.json', summaries)


if __name__ == '__main__':
    main()
