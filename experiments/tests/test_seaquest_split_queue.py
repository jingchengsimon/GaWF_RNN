"""Check queue partitioning, dependency gates, and duplicate-writer protection using stdlib."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

SCRIPT = Path(os.environ.get("SEAQUEST_SPLIT_SCRIPT", str(
    Path(__file__).resolve().parents[1] / "remote/run_sjc_seaquest_split_queue.py")))
SPEC = importlib.util.spec_from_file_location("split_queue", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class SplitQueueTests(unittest.TestCase):
    def test_exact_partition(self) -> None:
        self.assertEqual(queue.LANES, {"gpu1": ("a", "b", "c"), "gpu0": ("f995", "f999")})

    def test_writer_and_legacy_controller_rejected(self) -> None:
        result = Path("/tmp/seaquest/gawf_a_seed2")
        for argv in (["python", "--save_dir", str(result)], [
            "python", "-m", "experiments.remote.run_sjc_seaquest_diagnostics", "--model", "gawf"
        ]):
            with self.assertRaises(RuntimeError):
                queue.assert_no_writer(result, [argv])
        queue.assert_no_writer(result, [["python", "--save_dir", "/tmp/unrelated"]])

    def test_lstm_gate_waits_and_validates_all(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = SimpleNamespace(VARIANTS=["a", "b", "c", "f995", "f999"], validate=Mock())
            self.assertFalse(queue.lstm_ready(original, root / "results", root / "artifacts"))
            for variant in original.VARIANTS:
                for parent, name in [("results", "metrics.json"), ("artifacts", "done")]:
                    leaf = root / parent / f"lstm_{variant}_seed2"
                    leaf.mkdir(parents=True)
                    (leaf / name).touch()
            self.assertTrue(queue.lstm_ready(original, root / "results", root / "artifacts"))
            self.assertEqual(original.validate.call_count, 5)

    def test_failed_lstm_gate_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = root / "artifacts/lstm_a_seed2"
            status.mkdir(parents=True)
            (status / "fail").touch()
            with self.assertRaises(RuntimeError):
                queue.lstm_ready(SimpleNamespace(VARIANTS=["a"]), root / "results",
                                 root / "artifacts")


if __name__ == "__main__":
    unittest.main()
