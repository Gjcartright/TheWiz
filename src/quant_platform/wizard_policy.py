from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY_SCHEMA_VERSION = "wizard_discovery_policy.v1"


@dataclass(frozen=True)
class WizardDiscoveryPolicy:
    """Canonical thresholds for discovery display and paid-proof eligibility."""

    min_sharpe: float = 1.75
    min_returns_total_pct: float = 10.0
    min_closed_trades_for_proof: int = 5
    schema_version: str = POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_sharpe):
            raise ValueError("min_sharpe must be finite")
        if not math.isfinite(self.min_returns_total_pct) or self.min_returns_total_pct < 0:
            raise ValueError("min_returns_total_pct must be finite and non-negative")
        if self.min_closed_trades_for_proof < 1:
            raise ValueError("min_closed_trades_for_proof must be positive")
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported Wizard discovery policy schema: {self.schema_version}")

    @property
    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def policy_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


DEFAULT_WIZARD_DISCOVERY_POLICY = WizardDiscoveryPolicy()


def load_wizard_discovery_policy(root: Path = ROOT) -> WizardDiscoveryPolicy:
    """Load the versioned policy, falling back only when the file is absent.

    Temporary test roots intentionally use the canonical defaults. An existing
    malformed policy is a hard error so threshold drift cannot fail open.
    """

    path = root / "config" / "wizard_discovery_policy.json"
    if not path.exists():
        return DEFAULT_WIZARD_DISCOVERY_POLICY
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("raw_leaderboard") if isinstance(payload.get("raw_leaderboard"), dict) else {}
    proof = payload.get("proof_spend") if isinstance(payload.get("proof_spend"), dict) else {}
    return WizardDiscoveryPolicy(
        min_sharpe=float(raw["min_sharpe"]),
        min_returns_total_pct=float(raw["min_returns_total_pct"]),
        min_closed_trades_for_proof=int(proof["min_closed_trades"]),
        schema_version=str(payload.get("schema_version", "")),
    )
