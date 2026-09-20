"""Protocol guards for the SJC two-GPU IMDB launcher."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "experiments/remote/run_sjc_imdb_5model_best10seed_2gpu.sh"


def test_sjc_imdb_launcher_keeps_the_fixed_protocol() -> None:
    """The SJC migration must reuse the fixed task map and full-budget trainer."""

    text = LAUNCHER.read_text(encoding="utf-8")
    assert "imdb_5model_best10seed.py" in text
    assert 'for ((task_id=gpu; task_id<50; task_id+=2))' in text
    assert 'worker 0 &' in text
    assert 'worker 1 &' in text
    assert '--num_epochs "$NUM_EPOCHS"' in text
    assert '--patience "$PATIENCE"' in text
    assert '--num_layers "$NUM_LAYERS"' in text
    assert '--use_acceleration' in text


def test_sjc_imdb_launcher_validates_before_skip_and_completion() -> None:
    """Existing and newly completed units must pass the canonical validator."""

    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.count('"$GRID_UTIL" validate') >= 3
    assert "Refusing incomplete or mismatched result leaf" in text
    assert 'touch "$COMPLETE_FILE"' in text
