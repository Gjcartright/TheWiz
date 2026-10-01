# TheWiz nightly save points

The Mac mini has a local Codex automation, `thewiz-midnight-three-destination-save-point`, scheduled for **12:00 AM America/New_York**. It runs `/Users/gregc/Library/Application Support/TheWizBackup/nightly_savepoint.py`; the maintained copy is `scripts/ops/nightly_savepoint.py`. The existing `thewiz-nightly-github-backup` automation checks the receipt at 12:45 AM and retries a missing destination once. Both use the active Expansion checkout.

| Destination | Path | Coverage | Retention |
| --- | --- | --- | --- |
| Expansion | `/Volumes/Expansion/TheWizNightlySavepoints/YYYY-MM-DD/` | Active project plus read-only `TheWiz-LocalRuntime` | 30 daily save points |
| Mac mini internal drive | `/Users/gregc/Backups/TheWiz/savepoints/YYYY-MM-DD/` | Content-checked copy of the Expansion save point | 7 daily save points |
| Private GitHub | `Gjcartright/TheWiz-nightly-backup`, branch `codex/nightly-backup` | Versioned source and audit files listed by Git, including safe untracked files, excluding research paper PDFs | Git commit history |

The local copies include `.git`, data, reports, and the research runtime. Rebuildable Python environments, caches, macOS sidecars, transient locks, and credential-like files are excluded. Expansion uses ExFAT, so each dated copy is independent; the script uses hard-link deduplication only for the Mac internal copy. The GitHub mirror follows Git's tracked and nonignored file list from the frozen Expansion save point, except paper PDFs under `audit/quantopian_papers/`, which stay in the local copies because publisher rights may restrict redistribution. It refuses credential-like names except a reviewed credential-handling test filename, and scans file content for private-key/token patterns. It does not push to the public `Gjcartright/TheWiz` repository or grant research/trading authority.

Each run writes a dated JSON receipt under `/Users/gregc/Backups/TheWiz/state/`. `PASS` is recorded separately for Expansion, internal storage, and GitHub; local save points contain a SHA-256 file manifest. A failed `rsync` copy records its exit code and bounded stderr in the receipt. GitHub success requires the remote branch to return the commit that was pushed. A failed destination leaves successful copies in place and is retryable. Retention is applied only after all three destinations succeed.

To inspect the latest receipt:

```sh
/usr/bin/python3 '/Users/gregc/Library/Application Support/TheWizBackup/nightly_savepoint.py' --status
```

To restore, select a dated save point and copy its `project/` and `local_runtime/` folders into **new** directories first. Review the manifest and receipt before replacing any active checkout or runtime. Python environments can be rebuilt from `uv.lock`. GitHub contains the source and audit subset, while the two local destinations carry the broader data and report archive.

The Mac and local Codex scheduler must be available, and Expansion must be mounted. The older `com.thewiz.nightly-savepoint` LaunchAgent is disabled because macOS denied its `/usr/bin/python3` process removable-volume access on 2026-10-01. It is not part of the active backup schedule. A successful Codex run uses the app's existing removable-volume permission. Check dated receipts and `/Users/gregc/Library/Logs/TheWizBackup/` for any failure; the 12:45 AM Codex check can retry when the drive is available.
