"""Rebuild the requested 1-layer/3-layer slide comparison from saved Breakout histories.

Writes PNG/SVG and companion numeric/provenance files under the L3 ten-seed figure folder.
Preserves original panels' seed sets, budgets, smoothing and SD conventions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from utils.analysis.rl.atari.atari_learning_curves import (
    DEFAULT_MODELS, MODEL_COLORS, _aggregate_seeds,
)
from utils.analysis.rl.atari.atari_breakout_depth_curves import load_curve, aggregate


def main() -> None:
    """Validate source identities and save the compact two-panel slide figure."""
    root = Path(__file__).resolve().parents[4]
    output = root / 'results/figs/rl/atari/breakout_4action/fs4_stack4_l3_10seed_lrdecay'
    output.mkdir(parents=True, exist_ok=True)
    stem = 'layer1_vs_layer3_slides'
    for ext in ['png', 'svg', 'npz', 'json']:
        if (output / f'{stem}.{ext}').exists():
            raise FileExistsError(output / f'{stem}.{ext}')
    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 17, 'axes.labelsize': 15,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
    arrays, sources, groups = {}, [], []
    for ax, layers, seeds, budget in zip(axes, [1, 3], [[1, 2, 3, 4, 42], list(range(1, 11))],
                                       [1000000, 3000000]):
        models = DEFAULT_MODELS if layers == 1 else DEFAULT_MODELS[:5]
        for model in models:
            curves = []
            for seed in seeds:
                if layers == 1:
                    run = root / f'results/data/rl/atari/breakout_4action/fs4_stack4_l1_plain/{model}/seed{seed}'
                else:
                    run = root / ('results/data/rl/atari/breakout_l3_10seed_lrdecay/'
                                  f'atari_dqn_breakout_fs4_stack4_l3_10seed_lrdecay_{model}_seed{seed}')
                meta = json.loads((run / 'metrics.json').read_text())
                for key, value in dict(global_step=budget, num_layers=layers, frame_skip=4,
                                       frame_stack=4, model_type=model, num_actions=4,
                                       action_space_mode='minimal').items():
                    if meta.get(key) != value:
                        raise ValueError(f'{run}: {key}={meta.get(key)}, expected {value}')
                x, y = load_curve(run / 'metrics_history.jsonl')
                curves.append((x, y))
                key = f'l{layers}_{model}_seed{seed}'
                arrays[key + '_steps'], arrays[key + '_return100'] = x, y.astype(np.float32)
                for name in ['metrics.json', 'metrics_history.jsonl']:
                    path = run / name
                    sources.append(dict(path=str(path.relative_to(root)),
                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if layers == 1:
                x, mean, sd, count = _aggregate_seeds(curves, smooth=10)
                assert count == len(seeds)
            else:
                x, mean, sd = aggregate(curves, smooth=10)
            key = f'l{layers}_{model}'
            arrays.update({key+'_grid': x, key+'_mean': mean, key+'_sd': sd})
            groups.append(dict(layer=layers, model=model, seeds=seeds, budget=budget,
                               smoothing_log_points=10, sd_ddof=0 if layers == 1 else 1))
            ax.plot(x/1e6, mean, color=MODEL_COLORS[model], linewidth=2.1)
            ax.fill_between(x/1e6, mean-sd, mean+sd, color=MODEL_COLORS[model],
                            alpha=.16, linewidth=0)
        ax.set_title(f'{layers}-layer', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, budget/1e6)
        ax.set_ylim(0, 240)
        ax.set_yticks(np.arange(0, 241, 50))
        ax.set_xlabel('Environment steps (M)')
        ax.grid(axis='y', color='#dddddd', linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Episode return (last 100 episodes)')
    labels = ['ANN', 'RNN', 'GRU', 'LSTM', 'GaWF', 'S5', 'Mamba']
    handles = [Line2D([], [], color=MODEL_COLORS[m], linewidth=2.5, label=label)
               for m, label in zip(DEFAULT_MODELS, labels)]
    fig.legend(handles=handles, loc='upper center', ncol=7, frameon=False,
               bbox_to_anchor=(.52, 1.0), handlelength=2.0, columnspacing=1.5)
    fig.subplots_adjust(left=.075, right=.985, bottom=.16, top=.79, wspace=.12)
    for ext in ['png', 'svg']:
        fig.savefig(output/f'{stem}.{ext}', dpi=240, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    np.savez_compressed(output/f'{stem}.npz', **arrays)
    (output/f'{stem}.json').write_text(json.dumps(dict(
        script=str(Path(__file__).relative_to(root)), groups=groups, sources=sources,
        notes='Original 1-layer uses population SD; original 3-layer uses sample SD. '
              'Different seed counts and training budgets; not a controlled depth-only ablation.'
    ), indent=2)+'\n')
    print(output/f'{stem}.png')


if __name__ == '__main__':
    main()
