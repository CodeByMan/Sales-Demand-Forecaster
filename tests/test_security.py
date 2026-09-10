from src.security import MIN_ACCESS_KEY_LENGTH, is_authorized


def test_authorization_accepts_only_a_matching_strong_key():
    key = "correct-key-123456"

    assert len(key) >= MIN_ACCESS_KEY_LENGTH
    assert is_authorized(key, key)
    assert not is_authorized("wrong-key-123456", key)


def test_authorization_rejects_missing_or_weak_configuration():
    assert not is_authorized("short", "short")
    assert not is_authorized("", "correct-key-123456")
