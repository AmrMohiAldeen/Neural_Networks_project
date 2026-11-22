import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('event_check', Path(__file__).resolve().parents[1] / 'scripts/validate_event_streams.py')
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py required for HDF5 tests')
class EventChecks(unittest.TestCase):
    def validate(self, data):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.h5'
            with h5py.File(path, 'w') as stream:
                stream.create_dataset('events', data=np.asarray(data, dtype='uint32').reshape(-1, 4))
            return check.validate(path, 336, 224)

    def test_valid_polarity_and_microseconds(self):
        result = self.validate([[0, 0, 0, 0], [1000000, 335, 223, 1]])
        self.assertEqual(result['duration_seconds'], 1)
        self.assertEqual(result['polarities'], [0, 1])

    def test_unsigned_timestamp_decrease_detected(self):
        with self.assertRaisesRegex(ValueError, 'Timestamps decrease'):
            self.validate([[100, 0, 0, 0], [99, 0, 0, 1]])

    def test_out_of_bounds_detected(self):
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            self.validate([[0, 336, 0, 0]])

    def test_nonbinary_polarity_detected(self):
        with self.assertRaisesRegex(ValueError, 'polarity'):
            self.validate([[0, 0, 0, 2]])

    def test_empty_detected(self):
        with self.assertRaisesRegex(ValueError, 'No events'):
            self.validate([])


if __name__ == '__main__':
    unittest.main()
