import os

from quant_platform.env import load_env_file


def test_load_env_file_reads_key_values_without_overriding_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "CRYPTO_WIZARDS_BASE_URL=https://api.example.test",
                "CRYPTO_WIZARDS_API_KEY='secret'",
                'DYDX_TESTNET_WALLET_ADDRESS="wallet"',
            ]
        ),
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
