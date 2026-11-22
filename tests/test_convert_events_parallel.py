"""Exercise worker concurrency and resume without requiring CUDA or v2e deps."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/convert_events_parallel.py"
FAKE_V2E = '''
import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('-i')
p.add_argument('-o')
a, _ = p.parse_known_args()
output = Path(a.o)
started = time.time()
time.sleep(0.4)
(output / 'events.h5').write_bytes(b'fixture')
(output / 'interval.json').write_text(json.dumps([started, time.time()]))
if Path(a.i).read_text() == 'fail':
    raise SystemExit(7)
'''


class ParallelConversionTests(unittest.TestCase):
    def fixture(self, root):
        (root / "input/anger").mkdir(parents=True)
        (root / "v2e").mkdir()
        (root / "v2e/v2e.py").write_text(FAKE_V2E)
        for name in ("a", "b"):
            (root / f"input/anger/{name}.mp4").write_text("ok")
        return [sys.executable, str(LAUNCHER), "--input-root", str(root / "input"),
                "--output-root", str(root / "output"), "--v2e-root", str(root / "v2e"),
                "--workers", "2", "--dvs_aedat2", "None", "--dvs_emulator_seed", "476"]

    def run_command(self, command, code=0):
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result.stdout

    def test_concurrent_workers_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.fixture(root)
            self.assertIn("'success': 2", self.run_command(command))
            intervals = [json.loads((root / f"output/anger/{name}/interval.json").read_text())
                         for name in ("a", "b")]
            self.assertLess(max(i[0] for i in intervals), min(i[1] for i in intervals))
            self.assertIn("'skipped': 2", self.run_command(command))
            (root / "input/anger/a.mp4").write_text("changed-input")
            output = self.run_command(command)
            self.assertIn("'success': 1", output)
            self.assertIn("'skipped': 1", output)

    def test_failed_or_unmarked_h5_is_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.fixture(root)
            output = root / "output/anger/a"
            output.mkdir(parents=True)
            (output / "events.h5").write_bytes(b"partial legacy output")
            (root / "input/anger/b.mp4").write_text("fail")
            self.assertIn("'failed': 1", self.run_command(command, code=2))
            self.assertEqual((output / "events.h5").read_bytes(), b"fixture")
            failed = root / "output/anger/b"
            self.assertTrue((failed / "events.h5").exists())
            self.assertFalse((failed / ".conversion-success.json").exists())
            (root / "input/anger/b.mp4").write_text("ok")
            result = self.run_command(command)
            self.assertIn("'success': 1", result)
            self.assertIn("'skipped': 1", result)


if __name__ == "__main__":
    unittest.main()
