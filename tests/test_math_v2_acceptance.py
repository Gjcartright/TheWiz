from __future__ import annotations

import json

import pandas as pd

from quant_platform.math_v2_acceptance import build_math_v2_acceptance


def test_math_v2_acceptance_marker_is_generated_from_passing_checks(tmp_path):
    result = build_math_v2_acceptance(root=tmp_path)
    marker = json.loads(result["marker"].read_text(encoding="utf-8"))
    reconciliation = pd.read_csv(result["reconciliation"])
    statistical = pd.read_csv(result["statistical_validity"])

    assert marker["status"] == "passed"
    assert marker["generated_by"] == "quant_platform.math_v2_acceptance"
    assert marker["all_checks_passed"] is True
    assert reconciliation["status"].eq("PASS").all()
    assert statistical["status"].eq("PASS").all()
    assert marker["wizard_exact_mode_parity"] == "BLOCKED"
    assert marker["execution_authority"] == "BLOCKED"
