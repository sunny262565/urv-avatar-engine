import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preparation


class PreparationStatusTests(unittest.TestCase):
    def test_runtime_failure_is_visible_in_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "sources"
            source = source_root / "professor_arya" / "source.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"not-empty")
            with patch.object(preparation, "RESULT_ROOT", root / "results"), \
                 patch.object(preparation, "SOURCE_ROOT", source_root), \
                 patch.object(preparation, "MUSETALK_HOME", root / "missing-runtime"):
                with self.assertRaisesRegex(RuntimeError, "runtime is not installed"):
                    preparation.prepare("professor_arya")
                current = preparation.status("professor_arya")
                self.assertEqual(current["state"], "failed")
                self.assertFalse(current["preprocessed"])
                self.assertIn("runtime is not installed", current["error"])


if __name__ == "__main__":
    unittest.main()
