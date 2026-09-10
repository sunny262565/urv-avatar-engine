import unittest

from security import token_matches


class TokenMatchesTests(unittest.TestCase):
    def test_rejects_when_server_token_is_not_configured(self):
        self.assertFalse(token_matches("", "Bearer anything", "anything"))

    def test_accepts_dedicated_avatar_header(self):
        self.assertTrue(token_matches("secret", "Bearer runpod-key", "secret"))

    def test_accepts_legacy_bearer_for_non_runpod_use(self):
        self.assertTrue(token_matches("secret", "Bearer secret", None))

    def test_rejects_incorrect_credentials(self):
        self.assertFalse(token_matches("secret", "Bearer wrong", "wrong"))


if __name__ == "__main__":
    unittest.main()
