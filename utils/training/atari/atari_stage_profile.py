"""Bounded CUDA/CPU stage profiling after replay warm-up; writes trace and JSON timings.

Instrumentation is opt-in and preserves the training arguments and update cadence.
Synchronized window timing includes profiler overhead, so it is not throughput evidence.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Iterator

import torch


class AtariStageProfile:
    """Capture 256 collection steps after 1,000 post-learning warm-up steps."""

    def __init__(self, output: Path, device: torch.device) -> None:
        self.output = output
        self.device = device
        self.started_at: int | None = None
        self.active = False
        self.finished = False
        self.profiler = None
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.tasks: dict[str, int] = {}
        self.window_start = 0.0
        self.baseline_start: tuple[int, float] | None = None

    def begin(self, step: int, ready: bool) -> None:
        """Activate a bounded trace only after optimization is warm."""
        if ready and self.started_at is None:
            self.started_at = step
        if self.started_at is None or self.finished:
            return
        offset = step - self.started_at
        if offset >= 1256 and self.active:
            self.finish_window(step)
        elif offset >= 1000 and not self.active:
            self.output.mkdir(parents=True, exist_ok=False)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities)
            self.profiler.start()
            self.active = True
            self.sync()
            self.window_start = time.perf_counter()

    def sync(self) -> None:
        """Synchronize only the instrumented window for host phase attribution."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure a phase; optimization includes its nested replay_sample phase."""
        if not self.active:
            yield
            return
        self.sync()
        start = time.perf_counter()
        with torch.profiler.record_function('aim3/' + name):
            yield
            self.sync()
        self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - start
        self.counts[name] = self.counts.get(name, 0) + 1

    def finish_window(self, step: int) -> None:
        """Persist measured stages and CUDA operator details, then disable tracing."""
        self.sync()
        wall = time.perf_counter() - self.window_start
        self.active = False
        self.profiler.stop()
        self.profiler.export_chrome_trace(str(self.output / 'trace.json'))
        events = self.profiler.key_averages()
        (self.output / 'operators.txt').write_text(events.table(
            sort_by='self_cuda_time_total', row_limit=60))
        result = dict(learning_started_step=self.started_at, window_end_step=step,
                      window_wall_seconds=wall, phase_seconds=self.totals,
                      phase_counts=self.counts, collection_task_counts=self.tasks,
                      notes=['CUDA synchronized window includes instrumentation overhead.',
                             'optimization includes replay_sample; do not sum both.'])
        (self.output / 'stages.json').write_text(json.dumps(result, indent=2) + '\n')
        self.finished = True
        self.baseline_start = (step, time.perf_counter())

    def close(self, step: int) -> None:
        """Record post-trace throughput separately from instrumented timings."""
        if self.active:
            self.finish_window(step)
        if self.baseline_start is not None:
            self.sync()
            initial, wall = self.baseline_start
            elapsed = time.perf_counter() - wall
            (self.output / 'throughput.json').write_text(json.dumps(dict(
                start_step=initial, end_step=step, steps=step-initial,
                wall_seconds=elapsed, steps_per_second=(step-initial)/max(elapsed, 1e-9),
                notes='Post-trace end-to-end throughput includes normal logging/checkpoints.'
            ), indent=2) + '\n')
