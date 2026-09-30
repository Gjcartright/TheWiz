#!/usr/bin/env python3
"""Create local TheWiz save points and push a private, source-only GitHub mirror.

This script intentionally does not run project code, collectors, or trading services.
It is installed on the Mac's internal drive so a missing Expansion mount can be
reported rather than silently replaced by a directory on the internal drive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo


EXPANSION = Path("/Volumes/Expansion")
PROJECT = EXPANSION / "Crypto Wizard"
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
EXPANSION_BACKUPS = EXPANSION / "TheWizNightlySavepoints"
INTERNAL_BACKUPS = Path("/Users/gregc/Backups/TheWiz")
GITHUB_REMOTE = "https://github.com/Gjcartright/TheWiz-nightly-backup.git"
GITHUB_BRANCH = "codex/nightly-backup"
GIT_AUTHOR_NAME = "Gjcartright"
GIT_AUTHOR_EMAIL = "41594086+Gjcartright@users.noreply.github.com"
TIMEZONE = ZoneInfo("America/New_York")
EXCLUDES = (
    ".venv/", ".venv311/", ".venv312/", "__pycache__/", "*.pyc",
    ".pytest_cache/", ".ruff_cache/", ".mypy_cache/", ".DS_Store",
    "._*", ".runtime_locks/", ".runtime_tmp/", ".runtime-candidates/",
    ".env", ".env.local",
    ".env.*.local", "*.pem", "*.key", "*.p12", "*.pfx",
)
SENSITIVE_NAME = re.compile(
    r"(^|/)(\.env($|\.)|[^/]*(secret|credential|token)[^/]*|[^/]*\.(pem|key|p12|pfx)$)",
    re.IGNORECASE,
)
DATE_NAME = re.compile(r"\d{4}-\d{2}-\d{2}$")
MANIFEST_NAME = "SAVEPOINT_MANIFEST.json"
RECEIPT_NAME = "SAVEPOINT_RECEIPT.json"


def command(*args: str, cwd: Path | None = None, capture: bool = False) -> bytes:
    result = subprocess.run(
        args, cwd=str(cwd) if cwd else None, check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else b""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path) -> dict:
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(dirs)
        for name in sorted(files + [d for d in dirs if (Path(directory) / d).is_symlink()]):
            if name == ".DS_Store" or name.startswith("._"):
                continue
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative in (MANIFEST_NAME, RECEIPT_NAME):
                continue
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                rows.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
            elif stat.S_ISREG(mode):
                rows.append({"path": relative, "type": "file", "size": path.stat().st_size, "sha256": sha256(path)})
            else:
                raise RuntimeError("Unsupported snapshot entry: " + relative)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": "thewiz.savepoint.manifest.v1", "entries": rows,
            "entry_count": len(rows), "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def previous_snapshot(root: Path, today: str) -> Path | None:
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and DATE_NAME.fullmatch(p.name) and p.name < today)
    return candidates[-1] if candidates else None


def rsync(source: Path, target: Path, previous: Path | None = None) -> None:
    args = ["/usr/bin/rsync", "-a"]
    if previous is not None:
        args.append("--link-dest=" + str(previous))
    for pattern in EXCLUDES:
        args.extend(("--exclude", pattern))
    args.extend((str(source) + "/", str(target) + "/"))
    command(*args)


def prepare_expansion(today: str) -> tuple[Path, dict]:
    root = EXPANSION_BACKUPS
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = root / today
    if final.exists():
        manifest = json.loads((final / MANIFEST_NAME).read_text())
        if file_manifest(final)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError("Existing Expansion save point failed its manifest check")
        return final, manifest
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
    try:
        previous = previous_snapshot(root, today)
        for label, source in (("project", PROJECT), ("local_runtime", RUNTIME)):
            destination = temporary / label
            destination.mkdir()
            prior = previous / label if previous is not None else None
            rsync(source, destination, prior if prior and prior.exists() else None)
        manifest = file_manifest(temporary)
        write_json(temporary / MANIFEST_NAME, manifest)
        write_json(temporary / RECEIPT_NAME, {"date": today, "destination": "Expansion",
                   "source_paths": [str(PROJECT), str(RUNTIME)],
                   "excluded_rebuildable_or_sensitive_patterns": list(EXCLUDES),
                   "manifest_sha256": manifest["manifest_sha256"]})
        temporary.rename(final)
        return final, manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_internal(today: str, expansion: Path, expected: dict) -> Path:
    root = INTERNAL_BACKUPS / "savepoints"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = root / today
    if final.exists():
        actual = file_manifest(final)
        if actual["manifest_sha256"] != expected["manifest_sha256"]:
            raise RuntimeError("Existing Mac internal save point differs from Expansion")
        return final
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
    try:
        previous = previous_snapshot(root, today)
        rsync(expansion, temporary, previous)
        actual = file_manifest(temporary)
        if actual["manifest_sha256"] != expected["manifest_sha256"]:
            raise RuntimeError("Mac internal save point failed content comparison")
        temporary.rename(final)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def git(*args: str, cwd: Path, capture: bool = False) -> bytes:
    return command("/usr/bin/git", *args, cwd=cwd, capture=capture)


def github_paths(snapshot_project: Path) -> list[str]:
    raw = git("ls-files", "-co", "--exclude-standard", "-z", cwd=snapshot_project, capture=True)
    paths = sorted({os.fsdecode(item) for item in raw.split(b"\0") if item})
    for name in paths:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Unsafe GitHub mirror path")
        if name != ".env.example" and SENSITIVE_NAME.search(name):
            raise RuntimeError("Sensitive-looking GitHub mirror path: " + name)
        path = snapshot_project / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError("Unsupported GitHub mirror entry: " + name)
    return [name for name in paths if (snapshot_project / name).is_file()]


def secret_scan(path: Path) -> None:
    if path.suffix.lower() in (".pdf", ".jpg", ".jpeg", ".png", ".zip"):
        return
    private_key_header = re.compile(
        rb"(?m)^-----BEGIN (?:PRIVATE|OPENSSH PRIVATE|RSA PRIVATE) KEY-----\s*$"
    )
    token = re.compile(rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[opusr]_[A-Za-z0-9]{20,})")
    known_redaction_fixture = "ec02bd43d4c3ea6f3e993ba6fa3c378743038a101870b2dc810136c352ef437f"
    content = path.read_bytes()
    if private_key_header.search(content):
        raise RuntimeError("Potential credential content in GitHub mirror: " + str(path))
    for match in token.finditer(content):
        # This exact canary is an existing redaction test fixture, not a credential.
        if (path.name == "test_phase00_redaction.py"
                and hashlib.sha256(match.group()).hexdigest() == known_redaction_fixture):
            continue
        raise RuntimeError("Potential credential content in GitHub mirror: " + str(path))


def prepare_github(today: str, expansion: Path, manifest: dict) -> str:
    mirror = INTERNAL_BACKUPS / "github-mirror"
    mirror.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not (mirror / ".git").exists():
        git("init", "-b", GITHUB_BRANCH, cwd=mirror)
        git("remote", "add", "origin", GITHUB_REMOTE, cwd=mirror)
        git("config", "user.name", GIT_AUTHOR_NAME, cwd=mirror)
        git("config", "user.email", GIT_AUTHOR_EMAIL, cwd=mirror)
    remote = git("remote", "get-url", "origin", cwd=mirror, capture=True).decode().strip()
    branch = git("branch", "--show-current", cwd=mirror, capture=True).decode().strip()
    if remote != GITHUB_REMOTE or (branch and branch != GITHUB_BRANCH):
        raise RuntimeError("GitHub mirror remote or branch changed unexpectedly")
    source = expansion / "project"
    desired = github_paths(source)
    desired_set = set(desired)
    for directory, dirs, files in os.walk(mirror, topdown=True):
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in files:
            current = Path(directory) / name
            if current.relative_to(mirror).as_posix() not in desired_set:
                current.unlink()
    for name in desired:
        origin = source / name
        destination = mirror / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256(origin) != sha256(destination):
            shutil.copy2(origin, destination)
        secret_scan(destination)
    # The original checkout already tracks some paths matched by .gitignore.
    # This mirror contains only github_paths(), so force-add that reviewed set.
    git("add", "-f", "-A", cwd=mirror)
    git("commit", "-q", "--allow-empty", "-m",
        "Nightly TheWiz backup " + today + " " + manifest["manifest_sha256"][:12], cwd=mirror)
    git("push", "-q", "origin", "HEAD:refs/heads/" + GITHUB_BRANCH, cwd=mirror)
    local = git("rev-parse", "HEAD", cwd=mirror, capture=True).decode().strip()
    remote_head = git("ls-remote", "origin", "refs/heads/" + GITHUB_BRANCH,
                      cwd=mirror, capture=True).decode().split()[0]
    if local != remote_head:
        raise RuntimeError("GitHub remote did not report the pushed commit")
    return local


def prune(root: Path, keep: int) -> None:
    snapshots = sorted(p for p in root.iterdir() if p.is_dir() and DATE_NAME.fullmatch(p.name))
    for old in snapshots[:-keep]:
        shutil.rmtree(old)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Show the latest receipt")
    arguments = parser.parse_args()
    state = INTERNAL_BACKUPS / "state"
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    today = dt.datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    receipt_path = state / (today + ".json")
    if arguments.status:
        print(receipt_path.read_text() if receipt_path.exists() else json.dumps({"date": today, "status": "MISSING"}))
        return 0
    with (state / "nightly.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            if (prior.get("date") == today
                    and all(prior.get(key) == "PASS" for key in ("expansion", "mac_internal", "github"))
                    and Path(prior.get("expansion_path", "")).is_dir()
                    and Path(prior.get("mac_internal_path", "")).is_dir()):
                try:
                    remote_head = git("ls-remote", "origin", "refs/heads/" + GITHUB_BRANCH,
                                      cwd=INTERNAL_BACKUPS / "github-mirror", capture=True).decode().split()[0]
                    if remote_head == prior.get("github_commit"):
                        print(json.dumps(prior, sort_keys=True))
                        return 0
                except Exception:
                    pass
        receipt = {"schema_version": "thewiz.nightly.backup.v1", "date": today,
                   "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "expansion": "PENDING", "mac_internal": "PENDING", "github": "PENDING"}
        try:
            if not EXPANSION.is_mount() or PROJECT.resolve() != PROJECT or not RUNTIME.is_dir():
                raise RuntimeError("Expansion project mount or LocalRuntime is unavailable")
            if INTERNAL_BACKUPS.resolve().is_relative_to(EXPANSION):
                raise RuntimeError("Mac internal destination resolves onto Expansion")
            expansion, manifest = prepare_expansion(today)
            receipt["expansion"] = "PASS"
            receipt["expansion_path"] = str(expansion)
            receipt["manifest_sha256"] = manifest["manifest_sha256"]
            try:
                internal = prepare_internal(today, expansion, manifest)
                receipt["mac_internal"] = "PASS"
                receipt["mac_internal_path"] = str(internal)
            except Exception as error:
                receipt["mac_internal"] = "FAILED: " + str(error)
            try:
                receipt["github_commit"] = prepare_github(today, expansion, manifest)
                receipt["github"] = "PASS"
            except Exception as error:
                receipt["github"] = "FAILED: " + str(error)
            if receipt["mac_internal"] == "PASS" and receipt["github"] == "PASS":
                prune(EXPANSION_BACKUPS, 30)
                prune(INTERNAL_BACKUPS / "savepoints", 7)
        except Exception as error:
            if receipt["expansion"] == "PENDING":
                receipt["expansion"] = "FAILED: " + str(error)
            else:
                receipt["error"] = str(error)
        finally:
            write_json(receipt_path, receipt)
            print(json.dumps(receipt, sort_keys=True))
        return 0 if all(receipt[key] == "PASS" for key in ("expansion", "mac_internal", "github")) else 1


if __name__ == "__main__":
    sys.exit(main())
