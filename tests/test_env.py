import os

import pytest

from quant_platform.env import load_env_file, load_selected_env_keys


def test_load_env_file_reads_key_values_without_overriding_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "CRYPTO_WIZARDS_BASE_URL=https://api.example.test\n"
        "CRYPTO_WIZARDS_API_KEY='secret'\n"
        'DYDX_TESTNET_WALLET_ADDRESS="wallet"',
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "already-set")

    loaded = load_env_file(env_file)

    assert loaded == {
        "CRYPTO_WIZARDS_BASE_URL": "https://api.example.test",
        "DYDX_TESTNET_WALLET_ADDRESS": "wallet",
    }
    assert os.environ["CRYPTO_WIZARDS_BASE_URL"] == "https://api.example.test"
    assert os.environ["CRYPTO_WIZARDS_API_KEY"] == "already-set"
    assert os.environ["DYDX_TESTNET_WALLET_ADDRESS"] == "wallet"


def test_load_selected_env_keys_does_not_import_unrelated_credentials(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "CRYPTO_WIZARDS_API_KEY=wizard-key\nHYPERLIQUID_AGENT_PRIVATE_KEY=do-not-load\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    monkeypatch.delenv("HYPERLIQUID_AGENT_PRIVATE_KEY", raising=False)

    loaded = load_selected_env_keys(
        env_file,
        allowed_keys={"CRYPTO_WIZARDS_API_KEY"},
    )

    assert loaded == {"CRYPTO_WIZARDS_API_KEY": "wizard-key"}
    assert "HYPERLIQUID_AGENT_PRIVATE_KEY" not in os.environ


def _load_secret_file(path, *, selected):
    if selected:
        return load_selected_env_keys(
            path, allowed_keys={"CRYPTO_WIZARDS_API_KEY"}
        )
    return load_env_file(path)


@pytest.mark.parametrize("selected", [False, True])
def test_env_loader_rejects_symlink_before_decoding_secret(
    tmp_path, monkeypatch, selected
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    target = tmp_path / "secret-target"
    target.write_bytes(b"\xff")
    target.chmod(0o600)
    env_file = tmp_path / ".env.local"
    env_file.symlink_to(target)

    with pytest.raises(ValueError, match="env_file_symlink"):
        _load_secret_file(env_file, selected=selected)
    assert "CRYPTO_WIZARDS_API_KEY" not in os.environ


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("file_mode", [0o640, 0o604])
def test_env_loader_rejects_group_or_world_readable_file_before_decoding_secret(
    tmp_path, monkeypatch, selected, file_mode
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_bytes(b"\xff")
    env_file.chmod(file_mode)

    with pytest.raises(ValueError, match="env_file_permissions"):
        _load_secret_file(env_file, selected=selected)
    assert "CRYPTO_WIZARDS_API_KEY" not in os.environ


@pytest.mark.parametrize("selected", [False, True])
def test_env_loader_rejects_non_owner_file_before_importing_secret(
    tmp_path, monkeypatch, selected
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("CRYPTO_WIZARDS_API_KEY=not-imported\n", encoding="utf-8")
    env_file.chmod(0o600)
    actual_uid = os.getuid()

    with monkeypatch.context() as owner_probe:
        owner_probe.setattr(os, "getuid", lambda: actual_uid + 1)
        with pytest.raises(ValueError, match="env_file_not_owned_by_user"):
            _load_secret_file(env_file, selected=selected)
    assert "CRYPTO_WIZARDS_API_KEY" not in os.environ


@pytest.mark.parametrize("selected", [False, True])
def test_env_loader_rejects_file_replaced_between_checks(
    tmp_path, monkeypatch, selected
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("CRYPTO_WIZARDS_API_KEY=original\n", encoding="utf-8")
    env_file.chmod(0o600)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"\xff")
    replacement.chmod(0o600)
    real_open = os.open

    def replace_before_open(path, flags, *args, **kwargs):
        if path == env_file:
            replacement.replace(env_file)
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as replace_probe:
        replace_probe.setattr(os, "open", replace_before_open)
        with pytest.raises(ValueError, match="env_file_changed_before_read"):
            _load_secret_file(env_file, selected=selected)
    assert "CRYPTO_WIZARDS_API_KEY" not in os.environ


@pytest.mark.parametrize("selected", [False, True])
def test_env_loader_missing_file_remains_empty(tmp_path, selected):
    assert _load_secret_file(tmp_path / "missing.env", selected=selected) == {}
