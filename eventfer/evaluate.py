"""Score the nine frozen models and produce paired, clip-level analysis artifacts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import CLASSES, digest, fresh_directory, json_write, lock_directory, read_rows, verify_lock
from .protocol import MODES, check_evaluation_features, event_rate, identity, key, load_frozen
from .train import FeatureDataset, TemporalHead, metrics, seed_everything


def save_csv(path, rows, fields=None):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) > 1 else 0.}


def score_rows(rows, truth='label_id'):
    return metrics([int(r[truth]) for r in rows], [int(r['prediction_id']) for r in rows])


@torch.inference_mode()
def predict(model, dataset, rows, mode, seed, device, batch_size, rates):
    model.eval()
    output, cursor = [], 0
    for features, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        probabilities = torch.softmax(model(features.to(device)), dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite model probabilities')
        for probability in probabilities:
            row = rows[cursor]
            predicted = int(probability.argmax())
            output.append({
                'mode': mode, 'seed': seed, 'dataset': row['dataset'], 'clip_id': row['clip_id'],
                'split': row['split'], 'participant_id': row.get('participant_id', ''),
                'label': row['label'], 'label_id': int(row['label_id']),
                'expected_label': row.get('expected_label', ''),
                'prediction': CLASSES[predicted], 'prediction_id': predicted,
                'correct': int(predicted == int(row['label_id'])),
                'max_softmax': float(probability[predicted]), 'event_rate': rates[key(row)],
                **{'prob_' + label: float(probability[i]) for i, label in enumerate(CLASSES)},
            })
            cursor += 1
    return output


def analyze(predictions, frozen):
    seeds = frozen['seeds']
    scored, seed_rows, per_class = {}, [], []
    for mode in MODES:
        for seed in seeds:
            for split in ('test', 'target_test'):
                rows = [r for r in predictions if r['mode'] == mode and r['seed'] == seed and r['split'] == split]
                result = score_rows(rows)
                scored[(mode, seed, split)] = result
                seed_rows.append({'mode': mode, 'seed': seed, 'split': split, 'clips': len(rows),
                                  'macro_f1': result['macro_f1'], 'accuracy': result['accuracy']})
    aggregate = []
    for mode in MODES:
        for split in ('test', 'target_test'):
            results = [scored[(mode, seed, split)] for seed in seeds]
            f1 = mean_sd([r['macro_f1'] for r in results])
            accuracy = mean_sd([r['accuracy'] for r in results])
            aggregate.append({'mode': mode, 'split': split, 'clips_per_seed': sum(results[0]['support']),
                              'macro_f1_mean': f1['mean'], 'macro_f1_sd': f1['sd'],
                              'accuracy_mean': accuracy['mean'], 'accuracy_sd': accuracy['sd']})
            for i, label in enumerate(CLASSES):
                entry = {'mode': mode, 'split': split, 'label': label, 'support_per_seed': results[0]['support'][i]}
                for field, metric_name in [('precision', 'precision'), ('recall', 'recall'), ('per_class_f1', 'f1')]:
                    stats = mean_sd([r[field][i] for r in results])
                    entry.update({metric_name + '_mean': stats['mean'], metric_name + '_sd': stats['sd']})
                per_class.append(entry)
    comparisons = [('per_bin', 'baseline'), ('whole_clip', 'baseline'), ('per_bin', 'whole_clip')]
    paired, paired_summary = [], []
    for intervention, comparator in comparisons:
        rows = []
        for seed in seeds:
            source = 100 * (scored[(intervention, seed, 'test')]['macro_f1'] - scored[(comparator, seed, 'test')]['macro_f1'])
            target = 100 * (scored[(intervention, seed, 'target_test')]['macro_f1'] - scored[(comparator, seed, 'target_test')]['macro_f1'])
            row = {'intervention': intervention, 'comparator': comparator, 'seed': seed,
                   'source_macro_f1_delta_pp': source, 'target_macro_f1_delta_pp': target}
            paired.append(row)
            rows.append(row)
        source = mean_sd([r['source_macro_f1_delta_pp'] for r in rows])
        target = mean_sd([r['target_macro_f1_delta_pp'] for r in rows])
        paired_summary.append({'intervention': intervention, 'comparator': comparator,
                               'source_delta_pp_mean': source['mean'], 'source_delta_pp_sd': source['sd'],
                               'target_delta_pp_mean': target['mean'], 'target_delta_pp_sd': target['sd']})
    primary = paired_summary[0]
    hypothesis = frozen['hypothesis']
    success = (primary['target_delta_pp_mean'] >= hypothesis['target_gain_percentage_points_at_least'] - 1e-10
               and primary['source_delta_pp_mean'] >= -hypothesis['source_loss_percentage_points_at_most'] - 1e-10)
    lopo = []
    for user in frozen['expected_target_participants']:
        for intervention, comparator in comparisons:
            deltas, retained, omitted = [], [], []
            for seed in seeds:
                base = [r for r in predictions if r['mode'] == comparator and r['seed'] == seed and r['split'] == 'target_test']
                retained = [r for r in base if r['participant_id'] != user]
                omitted = [r for r in base if r['participant_id'] == user]
                proposed = [r for r in predictions if r['mode'] == intervention and r['seed'] == seed
                            and r['split'] == 'target_test' and r['participant_id'] != user]
                if retained:
                    deltas.append(100 * (score_rows(proposed)['macro_f1'] - score_rows(retained)['macro_f1']))
            missing = [c for c in CLASSES if c not in {r['label'] for r in retained}]
            stats = mean_sd(deltas) if deltas else {'mean': None, 'sd': None}
            lopo.append({'omitted_participant': user, 'intervention': intervention, 'comparator': comparator,
                         'retained_clips': len(retained), 'omitted_clips': len(omitted),
                         'target_delta_pp_mean': stats['mean'], 'target_delta_pp_sd': stats['sd'],
                         'missing_classes': ';'.join(missing), 'fixed_six_class_denominator': True})
    density, diagnostics, gaps = [], [], []
    cutoffs = frozen['density']['cutoffs']
    for mode in MODES:
        for seed in seeds:
            gaps.append({'mode': mode, 'seed': seed, 'source_minus_target_macro_f1_pp':
                         100 * (scored[(mode, seed, 'test')]['macro_f1'] - scored[(mode, seed, 'target_test')]['macro_f1'])})
            for split in ('test', 'target_test'):
                rows = [r for r in predictions if r['mode'] == mode and r['seed'] == seed and r['split'] == split]
                for band, name in enumerate(('low', 'medium', 'high')):
                    subset = [r for r in rows if int(np.searchsorted(cutoffs, r['event_rate'], side='right')) == band]
                    result = score_rows(subset) if subset else None
                    density.append({'mode': mode, 'seed': seed, 'split': split, 'density_band': name,
                                    'clips': len(subset), 'macro_f1': result['macro_f1'] if result else None,
                                    'accuracy': result['accuracy'] if result else None,
                                    'missing_classes': ';'.join(c for c in CLASSES if c not in {r['label'] for r in subset})})
            eligible = [dict(r, expected_id=CLASSES.index(r['expected_label'])) for r in predictions
                        if r['mode'] == mode and r['seed'] == seed and r['split'] == 'target_test' and r['expected_label'] in CLASSES]
            if eligible:
                result = score_rows(eligible, truth='expected_id')
                diagnostics.append({'mode': mode, 'seed': seed, 'clips': len(eligible),
                                    'expected_label_macro_f1': result['macro_f1'], 'expected_label_accuracy': result['accuracy'],
                                    'reported_expected_agreement': float(np.mean([r['label'] == r['expected_label'] for r in eligible]))})
    summary = {'aggregate': aggregate, 'paired': paired_summary,
               'hypothesis': dict(hypothesis, point_estimate_criterion_met=bool(success)),
               'primary_target_gain_pp': primary['target_delta_pp_mean'],
               'primary_source_loss_pp': -primary['source_delta_pp_mean'],
               'density_definition': frozen['density'], 'seeds': seeds,
               'notes': ['SD is sample variability across three seeds, not a confidence interval.',
                         'Macro-F1 always uses six classes; absent-class F1 is zero.',
                         'Source-minus-target gap includes dataset/annotation differences, not only sensor simulation.',
                         'Expected-label scores and participant omissions are diagnostics; they do not select models.']}
    return summary, scored, {'seed_metrics.csv': seed_rows, 'per_class.csv': per_class,
                             'paired_differences.csv': paired, 'participant_omission.csv': lopo,
                             'event_density.csv': density, 'transfer_gaps.csv': gaps,
                             'expected_label_diagnostic.csv': diagnostics}


def plots(output, summary, scored, frozen, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for offset, split, label in [(-.18, 'test', 'Simulated source test'), (.18, 'target_test', 'Real NEFER test')]:
        rows = [next(r for r in summary['aggregate'] if r['mode'] == mode and r['split'] == split) for mode in MODES]
        ax.bar(x + offset, [100 * r['macro_f1_mean'] for r in rows], .36,
               yerr=[100 * r['macro_f1_sd'] for r in rows], capsize=4, label=label)
    ax.set_xticks(x, MODES)
    ax.set_ylabel('Macro-F1 (%) — mean ± sample SD across seeds')
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'macro_f1.png', dpi=180)
    plt.close(fig)
    for mode in MODES:
        for split in ('test', 'target_test'):
            matrix = sum(np.asarray(scored[(mode, seed, split)]['confusion_matrix']) for seed in frozen['seeds'])
            sums = matrix.sum(1, keepdims=True)
            normalized = np.divide(matrix, sums, out=np.zeros_like(matrix, dtype=float), where=sums > 0) * 100
            save_csv(output / f'confusion_{mode}_{split}.csv',
                     [{'true_label': c, **{label: int(matrix[i, j]) for j, label in enumerate(CLASSES)}} for i, c in enumerate(CLASSES)])
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(normalized, vmin=0, vmax=100, cmap='Blues')
            ax.set_xticks(range(6), CLASSES, rotation=45, ha='right')
            ax.set_yticks(range(6), CLASSES)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title(f'{mode} / {split}\nPooled seed predictions, row-normalized; not an ensemble')
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f'{normalized[i,j]:.1f}', ha='center', va='center',
                            color='white' if normalized[i, j] > 50 else 'black', fontsize=8)
            fig.colorbar(im, ax=ax, label='% of true-class predictions')
            fig.tight_layout()
            fig.savefig(figures / f'confusion_{mode}_{split}.png', dpi=160)
            plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), squeeze=False)
    for i, mode in enumerate(MODES):
        for j, seed in enumerate(frozen['seeds']):
            history = [json.loads(line) for line in Path(frozen['models'][mode][str(seed)]['history']).read_text().splitlines()]
            ax = axes[i, j]
            ax.plot([r['epoch'] for r in history], [r['train_macro_f1'] for r in history], label='source train')
            ax.plot([r['epoch'] for r in history], [r['validation']['macro_f1'] for r in history], label='source validation')
            ax.set_title(f'{mode}, seed {seed}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Macro-F1')
            ax.set_ylim(0, 1)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figures / 'learning_curves.png', dpi=160)
    plt.close(fig)
    primary = [r for r in tables['participant_omission.csv']
               if r['intervention'] == 'per_bin' and r['comparator'] == 'baseline']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r['omitted_participant'] for r in primary],
           [r['target_delta_pp_mean'] if r['target_delta_pp_mean'] is not None else np.nan for r in primary],
           yerr=[r['target_delta_pp_sd'] or 0 for r in primary], capsize=4)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Omitted NEFER participant; see CSV for absent classes/clips')
    ax.set_ylabel('Per-bin minus baseline macro-F1 (pp)')
    fig.tight_layout()
    fig.savefig(figures / 'participant_omission.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for ax, split in zip(axes[0], ('test', 'target_test')):
        for mode in MODES:
            means, deviations = [], []
            for band in ('low', 'medium', 'high'):
                values = [r['macro_f1'] for r in tables['event_density.csv']
                          if r['mode'] == mode and r['split'] == split and r['density_band'] == band and r['clips']]
                stats = mean_sd(values) if values else {'mean': np.nan, 'sd': 0}
                means.append(100 * stats['mean'])
                deviations.append(100 * stats['sd'])
            ax.errorbar(range(3), means, yerr=deviations, marker='o', capsize=3, label=mode)
        ax.set_xticks(range(3), ['low', 'medium', 'high'])
        ax.set_title(split)
        ax.set_ylabel('Fixed-six-class macro-F1 (%)')
        ax.set_xlabel('Event-rate bands fitted on source training only')
        ax.set_ylim(0, 100)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figures / 'event_density.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-lock', type=Path, required=True)
    for mode in MODES:
        p.add_argument('--' + mode.replace('_', '-') + '-index', dest=mode + '_index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('Batch size must be positive')
    frozen = load_frozen(args.evaluation_lock)
    datasets, rows_by_mode, cache_hashes = {}, {}, {}
    reference_count_hashes = None
    for mode in MODES:
        index = getattr(args, mode + '_index')
        verify_lock(index)
        config = json.loads((index.parent / 'config.json').read_text())
        check_evaluation_features(config, frozen, mode)
        rows = read_rows(index)
        if {key(r): identity(r) for r in rows} != frozen['evaluation_rows']:
            raise ValueError('Evaluation cache omitted, added, or relabeled a frozen test clip')
        count_hashes = {key(r): r['counts_sha256'] for r in rows}
        if reference_count_hashes is None:
            reference_count_hashes = count_hashes
        elif count_hashes != reference_count_hashes:
            raise ValueError('Conditions did not use the same unscaled event count frames')
        datasets[mode] = FeatureDataset(rows, frozen['recipe']['bins'], frozen['recipe']['embedding_dim'])
        rows_by_mode[mode] = rows
        cache_hashes[mode] = digest(index)
    for mode in MODES:
        for entry in frozen['models'][mode].values():
            for filename, checksum in [('checkpoint', 'checkpoint_sha256'), ('run_config', 'run_config_sha256'), ('history', 'history_sha256')]:
                if digest(entry[filename]) != entry[checksum]:
                    raise ValueError('Frozen model/config/history changed; evaluation refused')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    seed_everything(0)
    rates = {key(r): event_rate(r) for r in rows_by_mode['baseline']}
    output = fresh_directory(args.output)
    predictions = []
    for mode in MODES:
        for seed in frozen['seeds']:
            entry = frozen['models'][mode][str(seed)]
            checkpoint = torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
            if checkpoint['seed'] != seed or checkpoint['feature_config']['mode'] != mode:
                raise ValueError('Frozen checkpoint seed/mode is inconsistent')
            model = TemporalHead(**checkpoint['model_config']).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            predictions.extend(predict(model, datasets[mode], rows_by_mode[mode], mode, seed, device, args.batch_size, rates))
            print(f'Scored locked {mode}, seed {seed}', flush=True)
    summary, scored, tables = analyze(predictions, frozen)
    save_csv(output / 'predictions.csv', predictions)
    for filename, rows in tables.items():
        fields = ['mode', 'seed', 'clips', 'expected_label_macro_f1', 'expected_label_accuracy', 'reported_expected_agreement'] if filename == 'expected_label_diagnostic.csv' else None
        save_csv(output / filename, rows, fields)
    errors = []
    for mode in MODES:
        for seed in frozen['seeds']:
            for split in ('test', 'target_test'):
                group = [r for r in predictions if r['mode'] == mode and r['seed'] == seed and r['split'] == split and not r['correct']]
                errors.extend(sorted(group, key=lambda r: (-r['max_softmax'], r['clip_id']))[:5])
    save_csv(output / 'error_examples.csv', errors, list(predictions[0]))
    json_write(output / 'seed_metrics.json', {f'{mode}/seed_{seed}/{split}': value
                                            for (mode, seed, split), value in scored.items()})
    plots(output, summary, scored, frozen, tables)
    summary['provenance'] = {'evaluation_lock_sha256': digest(args.evaluation_lock), 'feature_indexes_sha256': cache_hashes}
    json_write(output / 'summary.json', summary)
    text = ['# Controlled experiment results', '', 'Mean ± sample SD across three seeds; macro-F1 uses six classes.', '',
            '| Condition | Split | Clips/seed | Macro-F1 (%) | Accuracy (%) |', '|---|---|---:|---:|---:|']
    for row in summary['aggregate']:
        text.append(f'| {row["mode"]} | {row["split"]} | {row["clips_per_seed"]} | {100*row["macro_f1_mean"]:.2f} ± {100*row["macro_f1_sd"]:.2f} | {100*row["accuracy_mean"]:.2f} ± {100*row["accuracy_sd"]:.2f} |')
    text.extend(['', f'Primary paired NEFER gain: {summary["primary_target_gain_pp"]:.2f} percentage points.',
                 f'Primary paired source loss: {summary["primary_source_loss_pp"]:.2f} percentage points.',
                 f'Predeclared point-estimate criterion met: {summary["hypothesis"]["point_estimate_criterion_met"]}.',
                 '', *summary['notes']])
    (output / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    # Covers every completed table/figure/report; source checkpoints remain unchanged.
    lock_directory(output, [p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()])
    print(f'Evaluation complete: {output / "REPORT.md"}')


if __name__ == '__main__':
    main()
