"""The old direct Crypto Wizards crawler cannot bypass governed capture."""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/crawl_crypto_wizards_with_curl.sh"


def test_legacy_crawler_exits_before_reading_credentials_or_calling_provider(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path / "missing.env")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 78
    assert "retired" in result.stderr
    assert not list(tmp_path.iterdir())
