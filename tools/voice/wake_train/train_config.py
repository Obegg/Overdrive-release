"""Writes the microWakeWord training YAML for make_features.py's feature sets.

python3 -I train_config.py <features-dir> <train-dir> [--steps 20000,10000] [--lr 0.001,0.0002] > training.yaml
"""
import argparse
import os

import yaml

# (set, truth, sampling weight, penalty weight, truncation, right cutoffs)
SETS = [
    ('pos_iso', True, 1.2, 1.0, 'fixed_right_cutoff', list(range(0, 26))),
    ('pos_runon', True, 1.0, 1.0, 'fixed_right_cutoff', list(range(0, 21))),
    ('neg_confusable', False, 1.5, 1.0, 'fixed_right_cutoff', list(range(0, 26, 2))),
    ('neg_speech_direct', False, 3.0, 1.0, 'random', None),
    ('neg_speech_car', False, 3.0, 1.0, 'random', None),
    ('neg_talk_radio', False, 2.5, 1.0, 'random', None),
    ('neg_music', False, 3.5, 1.0, 'random', None),
    ('neg_noise', False, 0.7, 1.0, 'random', None),
    ('neg_prefix', False, 1.5, 1.0, 'truncate_start', None),
    ('neg_hard', False, 1.0, 1.5, 'random', None),
    ('neg_mined', False, 2.0, 1.5, 'random', None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features')
    ap.add_argument('train_dir')
    ap.add_argument('--steps', default='20000,10000')
    ap.add_argument('--lr', default='0.001,0.0002')
    ap.add_argument('--neg-weight', default='20')
    ap.add_argument('--clip-ms', type=int, default=520)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--time-mask', type=int, default=5)
    ap.add_argument('--freq-mask', type=int, default=3)
    a = ap.parse_args()
    feats = []
    for name, truth, sw, pw, trunc, cut in SETS:
        d = os.path.join(a.features, name)
        if not os.path.isdir(d):
            continue
        e = {'features_dir': d, 'sampling_weight': sw, 'penalty_weight': pw, 'truth': truth,
             'truncation_strategy': trunc, 'type': 'mmap'}
        if cut is not None:
            e['fixed_right_cutoffs'] = cut
        feats.append(e)
    steps = [int(s) for s in a.steps.split(',')]
    cfg = {
        'window_step_ms': 10,
        'train_dir': a.train_dir,
        'features': feats,
        'training_steps': steps,
        'positive_class_weight': [1.0] * len(steps),
        'negative_class_weight': [float(w) for w in a.neg_weight.split(',')],
        'learning_rates': [float(x) for x in a.lr.split(',')],
        'batch_size': a.batch,
        'time_mask_max_size': [a.time_mask], 'time_mask_count': [2],
        'freq_mask_max_size': [a.freq_mask], 'freq_mask_count': [2 if a.freq_mask > 3 else 1],
        'eval_step_interval': a.eval_every,
        'clip_duration_ms': a.clip_ms,
        'target_minimization': 0.9,
        'minimization_metric': None,
        'maximization_metric': 'average_viable_recall',
    }
    print(yaml.dump(cfg, default_flow_style=None, sort_keys=False))


if __name__ == '__main__':
    main()
