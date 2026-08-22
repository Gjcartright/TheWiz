import os

from quant_platform.env import load_env_file, load_selected_env_keys


def test_load_env_file_reads_key_values_without_overriding_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "CRYPTO_WIZARDS_BASE_URL=https://api.example.test\n"
        "CRYPTO_WIZARDS_API_KEY='secret'\n"
        'DYDX_TESTNET_WALLET_ADDRESS="wallet"',
        encoding="utf-8",
    )
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
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    monkeypatch.delenv("HYPERLIQUID_AGENT_PRIVATE_KEY", raising=False)

    loaded = load_selected_env_keys(
        env_file,
        allowed_keys={"CRYPTO_WIZARDS_API_KEY"},
    )

    assert loaded == {"CRYPTO_WIZARDS_API_KEY": "wizard-key"}
    assert "HYPERLIQUID_AGENT_PRIVATE_KEY" not in os.environ
