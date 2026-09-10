import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"teacher_meera", "dr_arjun_mehta", "professor_arya", "rohan_verma"}


class AvatarAssetTests(unittest.TestCase):
    def test_registry_contains_exactly_four_supported_avatars(self):
        registry = json.loads((ROOT / "avatars" / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(set(registry["avatars"]), EXPECTED)

    def test_each_avatar_has_source_and_matching_config(self):
        for avatar_id in EXPECTED:
            with self.subTest(avatar_id=avatar_id):
                directory = ROOT / "avatars" / avatar_id
                self.assertGreater((directory / "source.png").stat().st_size, 0)
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(config["id"], avatar_id)
                self.assertFalse(config["preprocessed"])


if __name__ == "__main__":
    unittest.main()
