from pathlib import Path

import numpy as np

from source.clutter.convert_float32_to_uint8 import convert_array, convert_splits
from source.clutter.generate_movies import MovingCharacter, StimulusConfig, paste_character


def test_convert_array_is_exact_and_idempotent(tmp_path: Path):
    source = tmp_path / "source.npy"
    target = tmp_path / "target.npy"
    values = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    np.save(source, values)

    result = convert_array(source, target, chunk_frames=2)
    assert result["status"] == "complete"
    converted = np.load(target)
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted.astype(np.float32), values)

    second = convert_array(source, target, chunk_frames=2)
    assert second["status"] == "already_complete"


def test_convert_array_rejects_fractional_values(tmp_path: Path):
    source = tmp_path / "source.npy"
    target = tmp_path / "target.npy"
    np.save(source, np.array([[[1.5]]], dtype=np.float32))

    try:
        convert_array(source, target, chunk_frames=1)
    except ValueError as exc:
        assert "non-integral" in str(exc)
    else:
        raise AssertionError("fractional float32 input must not be converted")


def test_convert_splits_copies_labels_exactly(tmp_path: Path):
    for split, stem in (
        ("train", "stimulus_reg-train"),
        ("validation", "stimulus_reg-validation"),
        ("test", "stimulus_reg-test"),
    ):
        np.save(tmp_path / f"{stem}-40h-float32.npy", np.zeros((2, 2, 2), np.float32))
        (tmp_path / f"{stem}-40h-float32.tsv").write_text(
            f"split\n{split}\n", encoding="utf-8"
        )

    convert_splits(
        tmp_path,
        "40h-float32",
        "40h-uint8",
        ("train", "validation", "test"),
        chunk_frames=1,
    )

    for stem in ("stimulus_reg-train", "stimulus_reg-validation", "stimulus_reg-test"):
        assert np.load(tmp_path / f"{stem}-40h-uint8.npy").dtype == np.uint8
        assert (tmp_path / f"{stem}-40h-uint8.tsv").read_bytes() == (
            tmp_path / f"{stem}-40h-float32.tsv"
        ).read_bytes()


def test_generator_defaults_to_uint8_and_saturates_overlap():
    assert StimulusConfig().storage_dtype == "uint8"
    frame = np.full((2, 2), 200, dtype=np.uint8)
    char = MovingCharacter(
        label=1,
        image=np.full((2, 2), 100, dtype=np.uint8),
        pos=np.array([1.0, 1.0]),
        vel=np.zeros(2),
    )
    paste_character(frame, char)
    np.testing.assert_array_equal(frame, np.full((2, 2), 255, dtype=np.uint8))
