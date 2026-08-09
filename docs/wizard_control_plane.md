# Wizard Control Plane

The Wizard control plane is the authority boundary between scanner discovery and downstream research. It makes no paid API calls and never accepts a trade.

## Daily Sequence

```bash
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli crypto-wizards-full-sweep
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli crypto-wizards-full-sweep --execute-wizard-sweep
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli wizard-control-plane --max-wizard-age-hours 24
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli run-orchestrator --stage discovery
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli build-command-dashboard
```

The zero-credit first command previews request count and cost. The executed sweep is all-or-none at preflight: it starts only when the full matrix fits inside the known remaining allowance after the reserve.

## Ranking Gates

Wizard ranking authority is ready only when all of these hold:

- The sweep manifest, candidates, and summary exist.
- Every planned cell completed and the summary says `complete_discovery`.
- Required API fields and numeric discovery values validate.
- Every completed response has response/configuration hashes and a raw snapshot.
- At least one candidate has a source timestamp within the allowed age.
- Required ranking artifacts have complete 64-character configuration hashes.

A dry run, failed request, stale source, malformed API row, missing raw snapshot, or missing configuration identity blocks ranking. The evidence is retained for diagnosis.

## Outputs

- `reports/active/wizard_control_plane_health.csv`
- `reports/active/wizard_sweep_settings_capture_queue.csv`
- `reports/active/wizard_api_contract_report.csv`
- `reports/active/wizard_config_lineage_audit.csv`
- `reports/active/wizard_freshness_blockers.csv`
- `reports/active/wizard_control_plane_summary.json`
- `reports/active/wizard_control_plane_summary.md`

The settings queue applies the current discovery thresholds, Sharpe at least `1.75` and `returns_total` at least `0.10`. It derives the exact mode from the returned strategy and spread type, identifies every missing pair-page setting, and carries discovery and candidate configuration hashes.

## Authority Rule

Crypto Wizards nominates and diagnoses candidates. Exact captured settings plus local, costed Hyperliquid replay provide acceptance evidence. A ready Wizard control plane does not authorize paper or live execution.
