#!/usr/bin/env python3
"""Build the tracked HEAD wheel on an internal staging volume.

The Expansion exFAT checkout is the source authority. Setuptools' wheel build
currently fails there at install_scripts, so the exact committed tree is
archived to a different filesystem for packaging. Untracked files are never
part of the wheel input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def run(*args: str, cwd: Path = ROOT, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=True, text=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--receipt", type=Path, help="Save the JSON build receipt at this path")
    args = parser.parse_args()

    repo = Path(run("git", "rev-parse", "--show-toplevel", stdout=subprocess.PIPE).stdout.strip())
    if repo != ROOT:
        raise SystemExit(f"script and Git root differ: {ROOT} != {repo}")
    if run("git", "status", "--porcelain", "--untracked-files=no", stdout=subprocess.PIPE).stdout:
        raise SystemExit("tracked source has uncommitted changes")
    head = run("git", "rev-parse", "HEAD", stdout=subprocess.PIPE).stdout.strip()
    tree = run("git", "rev-parse", "HEAD^{tree}", stdout=subprocess.PIPE).stdout.strip()

    staging_root = args.staging_root.expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    if os.stat(staging_root).st_dev == os.stat(ROOT).st_dev:
        raise SystemExit("staging root must be on a filesystem separate from the checkout")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="thewiz-wheel-", dir=staging_root) as temporary:
        stage = Path(temporary)
        archive = stage / "source.tar"
        checkout = stage / "checkout"
        checkout.mkdir()
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, stdout=stream, check=True)
        archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        with tarfile.open(archive) as stream:
            stream.extractall(checkout, filter="data")
        if not (checkout / "pyproject.toml").is_file():
            raise SystemExit("committed source archive lacks pyproject.toml")

        built = stage / "dist"
        run("uv", "build", "--wheel", "--offline", "--out-dir", str(built), cwd=checkout,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        wheels = list(built.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one wheel, found {len(wheels)}")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as package:
            if package.testzip() is not None:
                raise SystemExit("wheel ZIP integrity failed")
            names = package.namelist()
            if any(
                any(part.startswith("._") or part == "__MACOSX" for part in PurePosixPath(name).parts)
                for name in names
            ):
                raise SystemExit("wheel includes AppleDouble metadata")
        verification_env = stage / "verification-env"
        run("uv", "venv", "--offline", "--python", sys.executable, str(verification_env),
            cwd=checkout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        verification_python = verification_env / "bin" / "python"
        run("uv", "pip", "install", "--offline", "--python", str(verification_python),
            "--no-deps", str(wheel), cwd=checkout,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        run(str(verification_python), "-c", "import quant_platform", cwd=stage,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        target = output_dir / wheel.name
        with tempfile.NamedTemporaryFile(prefix=wheel.name + ".", dir=output_dir, delete=False) as stream:
            temporary_target = Path(stream.name)
        try:
            shutil.copyfile(wheel, temporary_target)
            os.replace(temporary_target, target)
        finally:
            temporary_target.unlink(missing_ok=True)

    receipt = {
        "source_commit": head,
        "source_tree": tree,
        "source_archive_sha256": archive_sha256,
        "wheel": str(target),
        "wheel_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "wheel_entries": len(names),
        "clean_install_and_import_verified": True,
        "source_authority": str(ROOT),
        "staging_root": str(staging_root),
    }
    if args.receipt is not None:
        receipt_path = args.receipt.expanduser().resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=receipt_path.parent,
                                         prefix=receipt_path.name + ".", delete=False) as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            temporary_receipt = Path(stream.name)
        os.replace(temporary_receipt, receipt_path)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
