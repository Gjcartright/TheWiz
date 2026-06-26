from pathlib import Path


def test_crypto_wizards_capture_script_records_passive_worker_metadata():
    script = Path("scripts/capture_crypto_wizards_pair_detail.js").read_text(encoding="utf-8")

    assert "attachPassiveWorkerCapture(worker)" in script
    assert "from_worker_messageerror" in script
    assert "worker_id" in script
    assert "script_url" in script
    assert "wasm_extracts" in script
    assert "tryWasmExtraction" in script
    assert "zscore_library-IlN0_w2C.js" in script
    assert "uniqueIdCandidates" in script
    assert "__CW_CAPTURE_SUMMARY__" in script
    assert "has_network_payloads" in script
    assert "has_wasm_extracts" in script
    assert "field_quality: captureFieldQuality()" in script
    assert "required_field_hits" in script
    assert "missing_baseline_fields" in script
    assert "missing_ecm_fields" in script
    assert "missing_two_leg_fields" in script
    assert "capture_ready_for_python_preflight" in script
    assert "__CW_CAPTURE_STATUS__" in script
    assert "console.table([status])" in script
    assert "next_capture_focus" in script
    assert "capture_operator_hint" in script
    assert "__CW_CAPTURE_RUNBOOK__" in script
    assert "click_refresh_or_recalculate_before_download" in script
    assert "capture.capture_summary" in script
    assert "maybeResponseText" in script
    assert "json: maybeParseJsonText(text)" in script
    assert "text," in script
