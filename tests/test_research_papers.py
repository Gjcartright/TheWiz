from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
from pypdf import PdfWriter

from quant_platform.research_papers import ingest_paper_library, verify_paper_sources


def _pdf(path: Path, *, title: str) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": title, "/Author": "Test Author"})
    with path.open("wb") as stream:
        writer.write(stream)


def _javascript_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_js("app.alert('untrusted');")
    with path.open("wb") as stream:
        writer.write(stream)


def test_ingest_paper_library_preserves_sources_and_blocks_execution(tmp_path):
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    paper = source_dir / "pairs.pdf"
    _pdf(paper, title="Pairs Trading Test")
    original = paper.read_bytes()

    result = ingest_paper_library(source_dir=source_dir, root=tmp_path)

    assert paper.read_bytes() == original
    inventory = pd.read_csv(result.paths["paper_inventory"]).fillna("")
    assert len(inventory) == 1
    assert inventory.iloc[0]["title"] == "Pairs Trading Test"
    assert not bool(inventory.iloc[0]["instruction_authority"])
    assert not bool(inventory.iloc[0]["live_signal_eligible"])
    assert inventory.iloc[0]["promotion_authority"] == "local_point_in_time_validation_only"
    manifest = json.loads(Path(inventory.iloc[0]["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "provisional"
    assert manifest["notes"] == "untrusted_document;research_only;no_execution_authority"


def test_ingest_paper_library_ignores_sidecars_and_inventories_archives(tmp_path):
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    paper = source_dir / "paper.pdf"
    _pdf(paper, title="Unique Paper")
    (source_dir / "._paper.pdf").write_bytes(b"not a pdf")
    archive = source_dir / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("paper.pdf", paper.read_bytes())

    result = ingest_paper_library(source_dir=source_dir, root=tmp_path)

    inventory = pd.read_csv(result.paths["paper_inventory"])
    archives = pd.read_csv(result.paths["paper_archive_inventory"])
    assert len(inventory) == 1
    assert len(archives) == 1
    assert archives.iloc[0]["member_path"] == "paper.pdf"
    assert result.summary["archive_members"] == 1


def test_ingest_paper_library_quarantines_actual_pdf_javascript(tmp_path):
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    _javascript_pdf(source_dir / "scripted.pdf")

    result = ingest_paper_library(source_dir=source_dir, root=tmp_path)

    inventory = pd.read_csv(result.paths["paper_inventory"])
    assert bool(inventory.iloc[0]["pdf_javascript"])
    assert inventory.iloc[0]["extraction_status"] == "quarantined_pdf_javascript"
    assert result.summary["quarantined_documents"] == 1


def test_verify_paper_sources_matches_crossref_without_execution_authority(tmp_path, monkeypatch):
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    _pdf(source_dir / "paper.pdf", title="Matched Paper")
    ingest = ingest_paper_library(source_dir=source_dir, root=tmp_path)
    inventory = pd.read_csv(ingest.paths["paper_inventory"]).fillna("")
    inventory.loc[0, "doi"] = "10.1000/test"
    inventory.to_csv(ingest.paths["paper_inventory"], index=False)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "title": ["Matched Paper"],
                    "type": "journal-article",
                    "publisher": "Test Publisher",
                    "relation": {},
                }
            }

    monkeypatch.setattr("quant_platform.research_papers.requests.get", lambda *_, **__: Response())
    result = verify_paper_sources(root=tmp_path)

    verification = pd.read_csv(result.paths["paper_source_verification"])
    assert verification.iloc[0]["identity_status"] == "verified_crossref"
    assert result.summary["retraction_signals"] == 0
