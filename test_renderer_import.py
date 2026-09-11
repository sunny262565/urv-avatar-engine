import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from renderer import MuseTalkRenderer


class MuseTalkRendererImportTests(unittest.TestCase):
    def test_adapter_is_imported_from_musetalk_checkout(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / "urv_stream_adapter.py").write_text(
                "import os\n"
                "IMPORT_CWD = os.getcwd()\n"
                "def create_renderer(avatar_dir, device):\n"
                "    return {'cwd': IMPORT_CWD, 'avatar_dir': avatar_dir, 'device': device}\n",
                encoding="utf-8",
            )
            renderer = MuseTalkRenderer()
            with patch.dict(os.environ, {"MUSETALK_HOME": str(checkout), "URV_DEVICE": "cpu"}):
                renderer.load(checkout / "avatar")

            self.assertEqual(Path(renderer.engine["cwd"]), checkout)
            self.assertEqual(Path.cwd(), original_cwd)


if __name__ == "__main__":
    unittest.main()
