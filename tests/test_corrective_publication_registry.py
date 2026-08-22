from __future__ import annotations

from pathlib import Path

from quant_platform.orchestration.corrective_publication_registry import (
    publication_surface_rows,
    unmigrated_publication_surface_ids,
    unpaired_staging_surface_ids,
)


def test_source_inventory_classifies_fenced_direct_and_lock_writers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
from pathlib import Path
from quant_platform.orchestration.corrective_runtime import atomic_write_text

def publish(path: Path) -> None:
    atomic_write_text(path, "safe")
    path.write_text("unsafe")

def acquire_lock(lock_path: Path) -> None:
    lock_path.write_text("coordination")
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)
    by_call = {(row["function"], row["call"]): row for row in rows}

    assert by_call[("publish", "atomic_write_text")]["migration_state"] == "MIGRATED"
    assert by_call[("publish", "atomic_write_text")]["authority_scope"] == "later_inventory"
    assert by_call[("publish", "path.write_text")]["migration_state"] == "UNMIGRATED"
    assert (
        by_call[("acquire_lock", "lock_path.write_text")]["migration_state"]
        == "EXCLUDED_COORDINATION"
    )


def test_to_csv_without_output_path_is_not_a_terminal_writer(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
def serialize(frame, path):
    text = frame.to_csv(index=False)
    frame.to_csv(path, index=False)
    return text
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["call"] == "frame.to_csv"
    assert rows[0]["migration_state"] == "UNMIGRATED"


def test_staging_writes_are_excluded_but_final_promotion_remains_blocking(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
def publish(frame, temporary, path):
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)
    by_call = {row["call"]: row for row in rows}

    assert by_call["frame.to_csv"]["migration_state"] == "EXCLUDED_STAGING"
    assert by_call["temporary.replace"]["migration_state"] == "UNMIGRATED"


def test_string_replace_is_not_a_publication_surface(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
def normalize(source):
    return source.replace("_", "-")
""".lstrip(),
        encoding="utf-8",
    )

    assert publication_surface_rows(tmp_path) == []


def test_link_to_temporary_is_staging_but_link_to_final_is_blocking(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
import os

def publish(source, temporary, path):
    os.link(source, temporary)
    os.link(temporary, path)
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)

    assert rows[0]["migration_state"] == "EXCLUDED_STAGING"
    assert rows[1]["migration_state"] == "UNMIGRATED"


def test_unmigrated_inventory_includes_later_scope(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def publish(path):\n    path.write_text('unfenced')\n",
        encoding="utf-8",
    )

    assert len(unmigrated_publication_surface_ids(tmp_path)) == 1


def test_staging_requires_explicit_governed_promotion(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
from quant_platform.orchestration.corrective_runtime import promote_staged_file

def unpaired(frame, temporary):
    frame.to_csv(temporary, index=False)

def paired(frame, temporary, path):
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)
    unpaired_ids = set(unpaired_staging_surface_ids(tmp_path))
    by_function = {row["function"]: row for row in rows if row["call_kind"] == "to_csv"}

    assert by_function["unpaired"]["surface_id"] in unpaired_ids
    assert by_function["paired"]["surface_id"] not in unpaired_ids


def test_fenced_writer_name_requires_canonical_import_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
from untrusted_helpers import atomic_write_text

def publish(path):
    atomic_write_text(path, "not proven")
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["classification"] == "unverified_fence_provenance"
    assert rows[0]["migration_state"] == "UNMIGRATED"
    assert rows[0]["blocker"] == "publication_fence_provenance_unverified"


def test_local_parameter_cannot_shadow_canonical_fenced_writer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
from quant_platform.orchestration.corrective_runtime import atomic_write_text

def publish(path, atomic_write_text):
    atomic_write_text(path, "shadowed")
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["migration_state"] == "UNMIGRATED"
    assert rows[0]["blocker"] == "publication_fence_provenance_unverified"


def test_canonical_alias_and_module_qualified_fences_are_accepted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
from quant_platform.orchestration.corrective_runtime import atomic_write_text as write
import quant_platform.orchestration.corrective_runtime as runtime

def publish(first, second):
    write(first, "direct alias")
    runtime.atomic_write_text(second, "module alias")
""".lstrip(),
        encoding="utf-8",
    )

    rows = publication_surface_rows(tmp_path)

    assert len(rows) == 2
    assert {row["migration_state"] for row in rows} == {"MIGRATED"}


def test_current_tree_has_no_unmigrated_or_unpaired_publication_surface() -> None:
    root = Path(__file__).resolve().parents[1]

    assert unmigrated_publication_surface_ids(root) == []
    assert unpaired_staging_surface_ids(root) == []
