# TheWiz nightly save points

The Mac mini runs `com.thewiz.nightly-savepoint` at **12:00 AM America/New_York**. The installed program is `/Users/gregc/Library/Application Support/TheWizBackup/nightly_savepoint.py`; its maintained copy is `scripts/ops/nightly_savepoint.py`. The existing Codex automation `thewiz-nightly-github-backup` checks the receipt at 12:45 AM and retries a missing destination once. It no longer targets the unmounted recovery checkout.

| Destination | Path | Coverage | Retention |
| --- | --- | --- | --- |
| Expansion | `/Volumes/Expansion/TheWizNightlySavepoints/YYYY-MM-DD/` | Active project plus read-only `TheWiz-LocalRuntime` | 30 daily save points |
| Mac mini internal drive | `/Users/gregc/Backups/TheWiz/savepoints/YYYY-MM-DD/` | Content-checked copy of the Expansion save point | 7 daily save points |
| Private GitHub | `Gjcartright/TheWiz-nightly-backup`, branch `codex/nightly-backup` | Versioned source and audit files listed by Git, including safe untracked files | Git commit history |

The local copies include `.git`, data, reports, and the research runtime. Rebuildable Python environments, caches, macOS sidecars, transient locks, and credential-like files are excluded. The GitHub mirror follows Git's tracked and nonignored file list from the frozen Expansion save point. It refuses credential-like names and detected private-key/token content. It does not push to the public `Gjcartright/TheWiz` repository or grant research/trading authority.

Each run writes a dated JSON receipt under `/Users/gregc/Backups/TheWiz/state/`. `PASS` is recorded separately for Expansion, internal storage, and GitHub; local save points contain a SHA-256 file manifest. GitHub success requires the remote branch to return the commit that was pushed. A failed destination leaves successful copies in place and is retryable. Retention is applied only after all three destinations succeed.

To inspect the latest receipt:

```sh
/usr/bin/python3 '/Users/gregc/Library/Application Support/TheWizBackup/nightly_savepoint.py' --status
```

To restore, select a dated save point and copy its `project/` and `local_runtime/` folders into **new** directories first. Review the manifest and receipt before replacing any active checkout or runtime. Python environments can be rebuilt from `uv.lock`. GitHub contains the source and audit subset, while the two local destinations carry the broader data and report archive.

The Mac must be on, the user LaunchAgent loaded, and Expansion mounted for a midnight run. If it is asleep or the drive is absent, check the dated receipt and LaunchAgent logs in `/Users/gregc/Library/Logs/TheWizBackup/`; the 12:45 AM Codex check can retry when the drive is available.
