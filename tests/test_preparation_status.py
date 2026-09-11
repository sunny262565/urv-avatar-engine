import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preparation


class PreparationStatusTests(unittest.TestCase):
    def test_staged_activation_replaces_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "professor_arya"
            staged = root / "staging-professor_arya-test"
            target.mkdir()
            staged.mkdir()
            (target / "old.txt").write_text("old", encoding="utf-8")
            (staged / "new.txt").write_text("new", encoding="utf-8")
            preparation._activate_staged(staged, target)
            self.assertFalse(staged.exists())
            self.assertFalse((target / "old.txt").exists())
            self.assertEqual((target / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(list(root.glob(".backup-*")))

    def test_motion_video_is_preferred_over_portrait(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.png").write_bytes(b"portrait")
            (root / "source.mp4").write_bytes(b"motion")
            source, source_kind = preparation._select_source(root)
            self.assertEqual(source.name, "source.mp4")
            self.assertEqual(source_kind, "motion-video")

    def test_portrait_remains_supported_as_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.png").write_bytes(b"portrait")
            source, source_kind = preparation._select_source(root)
            self.assertEqual(source.name, "source.png")
            self.assertEqual(source_kind, "portrait")

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
