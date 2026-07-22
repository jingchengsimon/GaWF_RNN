import numpy as np
import pandas as pd
import torch

from train_model import MC_RNN_Dataset
from utils.clutter_data_pipeline import BlockShuffleSampler, prepare_clutter_inputs


def _labels(n):
    return pd.DataFrame(
        {
            "fg_char_id": np.arange(n) % 10,
            "fg_char_x": np.zeros(n),
            "fg_char_y": np.zeros(n),
        }
    )


def test_all_cast_and_layout_modes_are_pixel_exact():
    data = np.arange(40 * 4 * 4, dtype=np.uint16).reshape(40, 4, 4).astype(np.uint8)
    outputs = {}
    for cast_mode in ("sample", "batch_cpu", "device"):
        for layout in ("stacked", "compact"):
            ds = MC_RNN_Dataset(
                data,
                _labels(len(data)),
                frame_num=4,
                chan_num=2,
                use_sector=True,
                input_cast_mode=cast_mode,
                frame_layout=layout,
            )
            frames = torch.from_numpy(ds[0][0]).unsqueeze(0)
            prepared = prepare_clutter_inputs(
                frames,
                device="cpu",
                cast_mode=cast_mode,
                frame_layout=layout,
                chan_num=2,
            )
            assert prepared.dtype == torch.float32
            assert prepared.shape == (1, 4, 2, 4, 4)
            outputs[(cast_mode, layout)] = prepared

    reference = outputs[("sample", "stacked")]
    for output in outputs.values():
        torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_block_shuffle_is_complete_and_deterministic():
    data = list(range(23))
    first = list(BlockShuffleSampler(data, block_size=5, seed=7))
    second_fresh = list(BlockShuffleSampler(data, block_size=5, seed=7))
    assert first == second_fresh
    assert sorted(first) == data
    block_runs = []
    for idx in first:
        block = idx // 5
        if not block_runs or block_runs[-1] != block:
            block_runs.append(block)
    assert len(block_runs) == 5
