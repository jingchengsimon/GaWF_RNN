"""Analyze seed1 continuous pre-switch states with 22 events per previous digit.

Consumes full-stream NPZ and source TSV; never resets or reruns the networks. Selects temporally
spread events, fits one common PCA per model on all balanced pre windows, and saves numeric data
separately from one two-cell interactive notebook. Switch-frame markers are reference only and
are excluded from PCA fitting and pre-window variance statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils.analysis.clutter.continuous_switch_pca import _pca


def select_events(labels_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select 22 chronologically spread clean pre windows per old digit."""
    with labels_path.open() as file:
        rows = list(csv.DictReader(file, delimiter='\t'))
    digits = np.asarray([int(row['fg_char_id']) for row in rows])
    fg = np.asarray([int(row['fg_switch']) != 0 for row in rows])
    bg = np.asarray([int(row['bg_switch']) != 0 for row in rows])
    any_switch = fg | bg
    eligible = np.asarray([t for t in np.flatnonzero(fg & bg)
                           if t - 50 >= 2 and not any_switch[t - 50:t].any()], dtype=np.int64)
    selected, counts = [], []
    for digit in range(10):
        pool = eligible[digits[eligible - 1] == digit]
        counts.append(len(pool))
        if len(pool) < 22:
            raise ValueError(f'Digit {digit} has only {len(pool)} eligible events')
        indices = np.rint(np.linspace(0, len(pool) - 1, 22)).astype(int)
        chosen = pool[indices]
        assert len(np.unique(chosen)) == 22
        assert np.all(digits[chosen[:, None] + np.arange(-50, 0)] == digit)
        selected.append(chosen)
    return np.stack(selected), np.asarray(counts, dtype=np.int64), np.flatnonzero(any_switch)


def make_figure(scores: np.ndarray, switch_scores: np.ndarray, events: np.ndarray,
                labels: np.ndarray, model: str, digit: int, capture: float,
                limits: np.ndarray) -> dict:
    """Render pre trajectories with a shared model basis and switch reference crosses."""
    relative = np.arange(-50, 0)
    traces = []
    for index, trajectory in enumerate(scores):
        traces.append({'type': 'scatter3d', 'mode': 'lines+markers', 'showlegend': False,
                       'name': f'Event {index + 1}',
                       **dict(zip(('x', 'y', 'z'), trajectory.T.tolist())),
                       'line': {'color': '#999999', 'width': 1},
                       'marker': {'size': 2, 'color': relative.tolist(), 'colorscale': 'Viridis',
                                  'cmin': -50, 'cmax': 0, 'showscale': index == 0,
                                  'colorbar': {'title': 'Relative frame', 'tickvals': [-50, -1, 0],
                                               'ticktext': ['−50', '−1 (pre end)', '0 (switch)']}},
                       'customdata': np.column_stack((relative, events[index] + relative,
                                                       labels[index])).tolist(),
                       'hovertemplate': 'Relative: %{customdata[0]}<br>Frame: %{customdata[1]}'
                       '<br>Digit: %{customdata[2]}<br>Sector: %{customdata[3]}<extra></extra>'})
    traces.append({'type': 'scatter3d', 'mode': 'markers', 'showlegend': False,
                   **dict(zip(('x', 'y', 'z'), switch_scores.T.tolist())),
                   'marker': {'symbol': 'x', 'color': '#E31A1C', 'size': 4},
                   'customdata': events.tolist(),
                   'hovertemplate': 'Switch reference: 0<br>Frame: %{customdata}<extra></extra>'})
    return {'data': traces, 'layout': {
        'title': f'{model.upper()} seed1 · pre-digit {digit} · 22 events'
                 f'<br>Shared PC1–3 capture of within-digit variance: {capture:.1%}',
        'height': 700, 'showlegend': False,
        'scene': {**{f'{axis}axis': {'title': f'PC{i + 1}', 'range': limits[i].tolist()}
                     for i, axis in enumerate('xyz')}, 'aspectmode': 'cube'},
        'margin': {'l': 0, 'r': 0, 'b': 0, 't': 75}},
        'config': {'scrollZoom': True, 'displaylogo': False}}


def run(args: argparse.Namespace) -> None:
    """Extract both model banks, fit comparable within-model projections, and export."""
    if args.data_out.exists():
        raise FileExistsError(f'Refusing overwrite: {args.data_out}')
    events, counts, all_switches = select_events(args.labels)
    args.data_out.mkdir(parents=True)
    relative = np.arange(-50, 0, dtype=np.int64)
    raw_frames = events[:, :, None] + relative
    previous_switch = all_switches[np.searchsorted(all_switches, events, side='left') - 1]
    np.savez_compressed(args.data_out / 'event_selection.npz', events=events,
                        pre_digits=np.arange(10, dtype=np.int64), eligible_counts=counts,
                        previous_switch_frames=previous_switch, age_at_switch=events-previous_switch)
    bank, statistics = {}, {}
    for model in ('gawf', 'rnn'):
        source = args.stream_root / f'{model}-seed01' / 'stream_trajectory.npz'
        with np.load(source) as data:
            frames = data['frame_ids']
            indices = np.searchsorted(frames, raw_frames)
            switch_indices = np.searchsorted(frames, events)
            np.testing.assert_array_equal(frames[indices], raw_frames)
            np.testing.assert_array_equal(frames[switch_indices], events)
            hidden, labels = data['raw_hidden'][indices], data['labels'][indices]
            switch_hidden, switch_labels = data['raw_hidden'][switch_indices], data['labels'][switch_indices]
        assert hidden.shape[:3] == (10, 22, 50) and np.isfinite(hidden).all()
        for digit in range(10):
            assert np.all(labels[digit, :, :, 0] == digit)
        mean, basis, explained, _ = _pca(hidden, 3)
        scores = ((hidden - mean) @ basis.T).astype(np.float32)
        switch_scores = ((switch_hidden - mean) @ basis.T).astype(np.float32)
        np.savez_compressed(args.data_out / f'{model}_seed01_pca.npz', hidden=hidden,
                            labels=labels, switch_hidden=switch_hidden, switch_labels=switch_labels,
                            events=events, relative_frames=relative, mean=mean, components=basis,
                            explained_variance_ratio=explained, scores=scores,
                            switch_scores=switch_scores)
        combined = np.concatenate((scores.reshape(-1, 3), switch_scores.reshape(-1, 3)))
        bounds = np.stack((combined.min(0), combined.max(0)), axis=1)
        pad = (bounds[:, 1] - bounds[:, 0]) * .05
        bounds += np.stack((-pad, pad), axis=1)
        bank[model], captures = {}, []
        for digit in range(10):
            h = hidden[digit].reshape(-1, hidden.shape[-1]).astype(np.float64)
            centered = h - h.mean(0)
            capture = float(np.square(centered @ basis.T).sum() / np.square(centered).sum())
            captures.append(capture)
            bank[model][str(digit)] = make_figure(scores[digit], switch_scores[digit], events[digit],
                                                 labels[digit], model, digit, capture, bounds)
        statistics[model] = {'pc3_global_variance': float(explained[:3].sum()),
                             'within_digit_shared_pc3_capture': captures,
                             'source': str(source)}
    (args.data_out / 'interactive_figures.json').write_text(json.dumps(bank, allow_nan=False))
    metadata = {'events_per_digit': 22, 'total_events': 220, 'eligible_counts': counts.tolist(),
                'selection': '22 rounded evenly spaced chronological indices within each digit pool',
                'fit_frames': [-50, -1], 'switch_frame': '0 reference only, excluded from PCA fit',
                'post_window_constraint': False, 'sector_constraint': False,
                'state_reset': 'none; reuse original full-stream inference',
                'pca': 'one centered unscaled common basis per model fitted across all 220 pre windows',
                'limits': 'same axis ranges across10digits within model; axes not aligned across models',
                'statistics': statistics}
    (args.data_out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    rel = args.data_out.resolve().relative_to(Path.cwd().resolve()).as_posix()
    cells = []
    for index, model in enumerate(('gawf', 'rnn'), 1):
        code = ('import json\nfrom pathlib import Path\nfrom IPython.display import display\n\n'
                f'# {model.upper()} seed1, 10 pre-digits, 22 events each.\n'
                '# Fit: −50…−1; red X: switch0 reference only. Shared PCA basis across digits.\n'
                f'relative_data = Path({rel!r})\n'
                'data_dir = next(p / relative_data for p in [Path.cwd(), *Path.cwd().parents]\n'
                "                if (p / relative_data / 'interactive_figures.json').is_file())\n"
                "figures = json.loads((data_dir / 'interactive_figures.json').read_text())\n"
                'for digit in range(10):\n'
                f'    display({{"application/vnd.plotly.v1+json": figures[{model!r}][str(digit)]}}, raw=True)\n')
        cells.append({'cell_type': 'code', 'id': model, 'metadata': {}, 'source': code,
                      'execution_count': index, 'outputs': [
                          {'output_type': 'display_data', 'metadata': {},
                           'data': {'application/vnd.plotly.v1+json': bank[model][str(d)]}}
                          for d in range(10)]})
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
        'cells': cells}
    args.figure_out.mkdir(parents=True, exist_ok=True)
    (args.figure_out / 'pre_digit_seed01_pca.ipynb').write_text(json.dumps(notebook, allow_nan=False))
    print(json.dumps(metadata, indent=2))


def main() -> None:
    """Parse source and output leaves for the seed1 pre-digit analysis."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream_root', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--data_out', type=Path, required=True)
    p.add_argument('--figure_out', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
