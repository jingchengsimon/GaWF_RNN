"""Verify bounded stage capture and separation of post-trace throughput."""
import json

import pytest
import torch

from utils.training.atari.atari_stage_profile import AtariStageProfile


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_bounded_profile(tmp_path, device_name):
    """Warm-up is excluded, the window closes once, and outputs contain real events."""
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA compute node required')
    p = AtariStageProfile(tmp_path / device_name, torch.device(device_name))
    p.begin(1, False)
    assert p.started_at is None
    p.begin(100, True)
    p.begin(1099, True)
    assert not p.active
    p.begin(1100, True)
    with p.phase('inference'):
        x = torch.ones(32, 32, device=device_name)
        x = x @ x
    p.begin(1356, True)
    assert p.finished and not p.active
    with p.phase('inference'):
        x = x + 1
    p.close(1400)
    stages = json.loads((p.output / 'stages.json').read_text())
    assert stages['phase_counts'] == {'inference': 1}
    assert stages['phase_seconds']['inference'] > 0
    assert (p.output / 'trace.json').stat().st_size > 0
    assert json.loads((p.output / 'throughput.json').read_text())['steps'] == 44
