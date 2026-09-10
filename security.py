"""Authentication helpers shared by HTTP and WebSocket routes."""

import hmac


def token_matches(expected: str, authorization: str | None, avatar_token: str | None) -> bool:
    """Accept the dedicated header, with legacy bearer auth as a fallback."""
    if not expected:
        return False
    return hmac.compare_digest(avatar_token or "", expected) or hmac.compare_digest(
        authorization or "", f"Bearer {expected}"
    )
