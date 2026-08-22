from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
)
from quant_platform.research_ingestion import build_research_source_registry

PAPER_INVENTORY_COLUMNS = [
    "paper_id",
    "title",
    "author",
    "publisher",
    "source_path",
    "source_filename",
    "sha256",
    "size_bytes",
    "page_count",
    "encrypted",
    "pdf_javascript",
    "extraction_status",
    "extracted_pages",
    "extracted_characters",
    "duplicate_of",
    "doi",
    "source_version",
    "source_authenticity",
    "review_status",
    "instruction_authority",
    "live_signal_eligible",
    "promotion_authority",
    "manifest_path",
    "pages_path",
    "ingested_at",
]

REVIEW_QUEUE_COLUMNS = [
    "paper_id",
    "title",
    "priority",
    "relevance_tags",
    "review_status",
    "gap_analysis_status",
    "premortem_status",
    "red_team_status",
    "math_review_status",
    "local_replication_status",
    "blocking_reason",
    "evidence_path",
]


@dataclass(frozen=True)
class ExtractedPage:
    paper_id: str
    page_number: int
    text: str
    text_sha256: str
    character_count: int
    extraction_status: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _paper_base(root: Path) -> Path:
    return root / "data" / "external" / "research_sources" / "papers"


def _report_base(root: Path) -> Path:
    return root / "reports" / "research" / "papers"


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def _source_id(file_hash: str) -> str:
    return f"paper_{file_hash[:16]}"


def _pdf_paths(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*.pdf")
        if path.is_file() and not path.name.startswith("._")
    )


def _title_from_first_page(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ignored = {"research article", "research paper", "article", "conference discussion"}
    for line in lines[:12]:
        normalized = _normalize_text(line)
        if normalized.casefold() in ignored:
            continue
        if len(normalized) >= 12 and not normalized.casefold().startswith("downloaded from"):
            return normalized
    return fallback


def _author_from_first_page(text: str, title: str) -> str:
    lines = [_normalize_text(line) for line in text.splitlines() if _normalize_text(line)]
    title_index = next((index for index, line in enumerate(lines) if line == title), -1)
    candidates = lines[title_index + 1 : title_index + 5] if title_index >= 0 else lines[1:5]
    for candidate in candidates:
        lowered = candidate.casefold()
        if any(token in lowered for token in ("abstract", "department", "university", "keywords")):
            continue
        if 3 <= len(candidate) <= 300 and any(char.isalpha() for char in candidate):
            return candidate
    return ""


def _publisher(path: Path, metadata: dict[str, Any]) -> str:
    subject = _normalize_text(metadata.get("/Subject", ""))
    if subject:
        return subject
    lowered = path.name.casefold()
    if "sciencedirect" in str(path.parent).casefold():
        return "ScienceDirect"
    if lowered.startswith("ssrn-"):
        return "SSRN"
    if re.match(r"\d{4}\.\d+v\d+\.pdf$", lowered):
        return "arXiv"
    return "unknown"


def _doi(text: str) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE)
    return match.group(0).rstrip(".,;)") if match else ""


def _relevance_tags(text: str) -> list[str]:
    lowered = text.casefold()
    mapping = {
        "pairs_trading": ("pairs trading", "pair trading", "statistical arbitrage"),
        "cointegration": ("cointegration", "error correction", "ecm"),
        "copula": ("copula",),
        "reinforcement_learning": ("reinforcement learning", "policy gradient"),
        "deep_learning": ("deep learning", "transformer", "neural network"),
        "regime": ("regime", "structural break", "market crisis"),
        "tail_risk": ("tail risk", "catastrophic risk", "extreme value"),
        "crypto": ("cryptocurrency", "crypto", "defi", "dao"),
        "execution_costs": ("transaction cost", "slippage", "market impact", "no-trade region"),
        "portfolio": ("portfolio", "rebalanc"),
        "anomaly_detection": ("anomaly detection", "anomalous"),
        "graph_learning": ("graph neural", "graph representation", "graph embedding"),
        "sentiment": ("sentiment", "attention herding", "social network"),
    }
    return [tag for tag, terms in mapping.items() if any(term in lowered for term in terms)]


def _priority(tags: Iterable[str]) -> str:
    tag_set = set(tags)
    if tag_set & {"pairs_trading", "cointegration", "copula"}:
        return "must_review"
    if tag_set & {"reinforcement_learning", "regime", "tail_risk", "crypto", "execution_costs"}:
        return "high"
    if tag_set & {"deep_learning", "portfolio", "anomaly_detection", "graph_learning"}:
        return "useful"
    return "optional"


def _pdf_javascript(reader: Any) -> bool:
    def resolved(value: Any) -> Any:
        try:
            return value.get_object()
        except (AttributeError, TypeError, ValueError):
            return value

    def contains_javascript(value: Any, *, depth: int = 0) -> bool:
        if value is None or depth > 12:
            return False
        current = resolved(value)
        if isinstance(current, (list, tuple)):
            return any(contains_javascript(child, depth=depth + 1) for child in current)
        if hasattr(current, "items"):
            action_type = str(current.get("/S", ""))
            if action_type == "/JavaScript" or "/JS" in current:
                return True
            return any(contains_javascript(child, depth=depth + 1) for _, child in current.items())
        return False

    try:
        root = resolved(reader.trailer.get("/Root", {}))
        names = resolved(root.get("/Names", {})) if hasattr(root, "get") else {}
        candidates = [
            root.get("/OpenAction") if hasattr(root, "get") else None,
            root.get("/AA") if hasattr(root, "get") else None,
            names.get("/JavaScript") if hasattr(names, "get") else None,
        ]
        return any(contains_javascript(candidate) for candidate in candidates)
    except Exception:  # noqa: BLE001 - malformed PDF object graphs vary by producer
        return False


def _extract_pdf(path: Path, paper_id: str) -> tuple[dict[str, Any], list[ExtractedPage]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific
        raise RuntimeError("paper ingestion requires pypdf") from exc

    reader = PdfReader(str(path), strict=False)
    metadata = dict(reader.metadata or {})
    pages: list[ExtractedPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            status = "extracted" if text.strip() else "empty"
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - parser-specific failures
            text = ""
            status = f"error:{type(exc).__name__}"
        pages.append(
            ExtractedPage(
                paper_id=paper_id,
                page_number=page_number,
                text=text,
                text_sha256=sha256(text.encode("utf-8", errors="replace")).hexdigest(),
                character_count=len(text),
                extraction_status=status,
            )
        )
    first_page = pages[0].text if pages else ""
    compact_text = _normalize_text("\n".join(page.text for page in pages))
    title = _normalize_text(metadata.get("/Title", "")) or _title_from_first_page(
        first_page, path.stem
    )
    author = _normalize_text(metadata.get("/Author", "")) or _author_from_first_page(
        first_page, title
    )
    summary = {
        "title": title,
        "author": author,
        "publisher": _publisher(path, metadata),
        "page_count": len(reader.pages),
        "encrypted": bool(reader.is_encrypted),
        "pdf_javascript": _pdf_javascript(reader),
        "doi": _doi(compact_text[:30000]),
        "metadata": {str(key): _normalize_text(value) for key, value in metadata.items()},
        "relevance_tags": _relevance_tags(compact_text),
    }
    return summary, pages


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_text(path, encoded, encoding="utf-8")
    return path


def _archive_members(source_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for archive in sorted(source_dir.rglob("*.zip")):
        if archive.name.startswith("._"):
            continue
        try:
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if member.is_dir() or member.filename.startswith("._"):
                        continue
                    rows.append(
                        {
                            "archive_path": str(archive.resolve()),
                            "member_path": member.filename,
                            "member_size": member.file_size,
                            "member_crc32": f"{member.CRC:08x}",
                            "encrypted": bool(member.flag_bits & 0x1),
                        }
                    )
        except (OSError, zipfile.BadZipFile):
            rows.append(
                {
                    "archive_path": str(archive.resolve()),
                    "member_path": "",
                    "member_size": 0,
                    "member_crc32": "",
                    "encrypted": False,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["archive_path", "member_path", "member_size", "member_crc32", "encrypted"],
    )


def _title_similarity(left: str, right: str) -> float:
    normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def _embedded_source_ids(inventory_row: pd.Series) -> tuple[str, str]:
    pages = []
    try:
        with Path(str(inventory_row["pages_path"])).open(encoding="utf-8") as stream:
            for line in stream:
                if len(pages) >= 5:
                    break
                payload = json.loads(line)
                pages.append(str(payload.get("text", "")))
    except (OSError, ValueError, json.JSONDecodeError):
        return "", ""
    text = "\n".join(pages)
    ssrn = re.findall(r"(?:ssrn\.com/abstract=|abstract=)(\d{5,9})", text, re.IGNORECASE)
    arxiv = re.findall(r"arxiv(?: preprint)?[: ]+(\d{4}\.\d{4,5})", text, re.IGNORECASE)
    return (ssrn[0] if ssrn else "", arxiv[0] if arxiv else "")


def verify_paper_sources(
    *,
    root: Path = ROOT,
    timeout_seconds: float = 20.0,
    crossref_base_url: str = "https://api.crossref.org/works",
) -> CommandResult:
    inventory_path = _report_base(root) / "paper_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError("paper inventory missing; run ingest-paper-library first")
    inventory = pd.read_csv(inventory_path).fillna("")
    rows: list[dict[str, object]] = []
    for _, paper in inventory.iterrows():
        doi = str(paper["doi"]).strip()
        filename = str(paper["source_filename"])
        embedded_ssrn, embedded_arxiv = _embedded_source_ids(paper)
        registry_title = ""
        registry_type = ""
        registry_publisher = ""
        registry_status = "not_checked"
        title_similarity = 0.0
        relation = ""
        retraction_signal = False
        expected_id = ""
        embedded_id = ""
        identity_status = "unverified"
        error = ""

        if doi:
            try:
                response = requests.get(
                    f"{crossref_base_url.rstrip('/')}/{requests.utils.quote(doi, safe='')}",
                    timeout=timeout_seconds,
                    headers={"User-Agent": "TheWizResearch/0.1 (research source verification)"},
                )
                response.raise_for_status()
                message = response.json().get("message", {})
                titles = message.get("title") or []
                registry_title = str(titles[0]) if titles else ""
                registry_type = str(message.get("type", ""))
                registry_publisher = str(message.get("publisher", ""))
                relation_payload = message.get("relation") or {}
                relation = json.dumps(relation_payload, sort_keys=True)
                relation_text = relation.casefold()
                retraction_signal = any(
                    token in relation_text
                    for token in ("retract", "withdraw", "remove", "expression-of-concern")
                )
                title_similarity = _title_similarity(str(paper["title"]), registry_title)
                registry_status = "matched" if title_similarity >= 0.9 else "title_mismatch"
                identity_status = (
                    "verified_crossref"
                    if registry_status == "matched" and not retraction_signal
                    else "verification_conflict"
                )
            except (requests.RequestException, ValueError, KeyError) as exc:
                registry_status = "request_failed"
                error = f"{type(exc).__name__}:{exc}"
        elif filename.casefold().startswith("ssrn-"):
            match = re.search(r"ssrn-(\d+)", filename, re.IGNORECASE)
            expected_id = match.group(1) if match else ""
            embedded_id = embedded_ssrn
            registry_status = "embedded_id_check"
            identity_status = (
                "verified_embedded_id"
                if expected_id and expected_id == embedded_id
                else "identity_conflict"
            )
        elif re.match(r"\d{4}\.\d+v\d+\.pdf$", filename, re.IGNORECASE):
            match = re.match(r"(\d{4}\.\d+)v\d+\.pdf$", filename, re.IGNORECASE)
            expected_id = match.group(1) if match else ""
            embedded_id = embedded_arxiv
            registry_status = "embedded_id_check"
            identity_status = (
                "verified_embedded_id"
                if expected_id and expected_id == embedded_id
                else "identity_conflict"
            )

        verification_path = Path(str(paper["pages_path"])).parent / "source_verification.json"
        verification = {
            "paper_id": paper["paper_id"],
            "doi": doi,
            "expected_id": expected_id,
            "embedded_id": embedded_id,
            "registry_title": registry_title,
            "registry_type": registry_type,
            "registry_publisher": registry_publisher,
            "registry_status": registry_status,
            "title_similarity": title_similarity,
            "relation": relation,
            "retraction_signal": retraction_signal,
            "identity_status": identity_status,
            "error": error,
            "checked_at": _now_iso(),
        }
        atomic_write_text(verification_path, json.dumps(verification, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        rows.append({**verification, "verification_path": str(verification_path)})

    report = pd.DataFrame(rows)
    output = _report_base(root) / "paper_source_verification.csv"
    atomic_write_csv(report, output, index=False)
    status_by_id = report.set_index("paper_id")["identity_status"].to_dict()
    inventory["source_authenticity"] = inventory["paper_id"].map(status_by_id).fillna("unverified")
    atomic_write_csv(inventory, inventory_path, index=False)
    blocked = int(
        report["identity_status"].isin(["identity_conflict", "verification_conflict"]).sum()
    )
    return CommandResult(
        paths={"paper_source_verification": output, "paper_inventory": inventory_path},
        summary={
            "papers_checked": len(report),
            "verified": int(
                report["identity_status"].astype(str).str.startswith("verified_").sum()
            ),
            "identity_conflicts": blocked,
            "request_failures": int(report["registry_status"].eq("request_failed").sum()),
            "retraction_signals": int(report["retraction_signal"].astype(bool).sum()),
        },
    )


def ingest_paper_library(*, source_dir: Path, root: Path = ROOT) -> CommandResult:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"paper source directory does not exist: {source_dir}")

    report_base = _report_base(root)
    report_base.mkdir(parents=True, exist_ok=True)
    manifests = root / "data" / "external" / "research_sources" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    ingested_at = _now_iso()
    inventory_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}

    for source_path in _pdf_paths(source_dir):
        file_hash = _hash_file(source_path)
        paper_id = _source_id(file_hash)
        duplicate_of = hashes.get(file_hash, "")
        hashes.setdefault(file_hash, paper_id)
        paper_dir = _paper_base(root) / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        pages_path = paper_dir / "pages.jsonl"
        manifest_path = manifests / f"{paper_id}.json"
        source_manifest_path = paper_dir / "source_manifest.json"
        try:
            pdf, pages = _extract_pdf(source_path, paper_id)
            _write_jsonl(pages_path, (asdict(page) for page in pages))
            extracted_pages = sum(page.extraction_status == "extracted" for page in pages)
            extracted_characters = sum(page.character_count for page in pages)
            extraction_status = (
                "complete" if extracted_pages == len(pages) and extracted_characters else "partial"
            )
            if pdf["pdf_javascript"]:
                extraction_status = "quarantined_pdf_javascript"
        except Exception as exc:  # noqa: BLE001 - one malformed source must not abort the library
            pdf = {
                "title": source_path.stem,
                "author": "",
                "publisher": "unknown",
                "page_count": 0,
                "encrypted": False,
                "pdf_javascript": False,
                "doi": "",
                "metadata": {},
                "relevance_tags": [],
            }
            pages = []
            extracted_pages = 0
            extracted_characters = 0
            extraction_status = f"failed:{type(exc).__name__}"

        source_version = "duplicate" if duplicate_of else "unique"
        review_status = "blocked_duplicate" if duplicate_of else "intake_complete"
        quarantine_status = "quarantined" if pdf["pdf_javascript"] else "active"
        extended_manifest = {
            "source_id": paper_id,
            "source_type": "pdf",
            "title": pdf["title"],
            "author": pdf["author"],
            "channel_or_publisher": pdf["publisher"],
            "date_added": ingested_at,
            "source_path_or_url": str(source_path.resolve()),
            "topic_tags": ";".join(pdf["relevance_tags"]),
            "status": "provisional",
            "review_status": review_status,
            "quarantine_status": quarantine_status,
            "confidence": 0.0,
            "processed_at": ingested_at,
            "notes": "untrusted_document;research_only;no_execution_authority",
            "sha256": file_hash,
            "size_bytes": source_path.stat().st_size,
            "page_count": pdf["page_count"],
            "doi": pdf["doi"],
            "source_version": source_version,
            "duplicate_of": duplicate_of,
            "source_authenticity": "unverified",
            "instruction_authority": False,
            "live_signal_eligible": False,
            "promotion_authority": "local_point_in_time_validation_only",
            "metadata": pdf["metadata"],
            "pages_path": str(pages_path),
        }
        manifest_text = json.dumps(extended_manifest, indent=2, ensure_ascii=True, sort_keys=True)
        atomic_write_text(manifest_path, manifest_text, encoding="utf-8")
        atomic_write_text(source_manifest_path, manifest_text, encoding="utf-8")

        inventory_rows.append(
            {
                "paper_id": paper_id,
                "title": pdf["title"],
                "author": pdf["author"],
                "publisher": pdf["publisher"],
                "source_path": str(source_path.resolve()),
                "source_filename": source_path.name,
                "sha256": file_hash,
                "size_bytes": source_path.stat().st_size,
                "page_count": pdf["page_count"],
                "encrypted": pdf["encrypted"],
                "pdf_javascript": pdf["pdf_javascript"],
                "extraction_status": extraction_status,
                "extracted_pages": extracted_pages,
                "extracted_characters": extracted_characters,
                "duplicate_of": duplicate_of,
                "doi": pdf["doi"],
                "source_version": source_version,
                "source_authenticity": "unverified",
                "review_status": review_status,
                "instruction_authority": False,
                "live_signal_eligible": False,
                "promotion_authority": "local_point_in_time_validation_only",
                "manifest_path": str(manifest_path),
                "pages_path": str(pages_path),
                "ingested_at": ingested_at,
            }
        )
        tags = pdf["relevance_tags"]
        review_rows.append(
            {
                "paper_id": paper_id,
                "title": pdf["title"],
                "priority": _priority(tags),
                "relevance_tags": ";".join(tags),
                "review_status": review_status,
                "gap_analysis_status": "pending",
                "premortem_status": "pending",
                "red_team_status": "pending",
                "math_review_status": "pending",
                "local_replication_status": "not_started",
                "blocking_reason": "duplicate_source" if duplicate_of else "deep_review_required",
                "evidence_path": str(source_manifest_path),
            }
        )

    inventory = pd.DataFrame(inventory_rows, columns=PAPER_INVENTORY_COLUMNS)
    review_queue = pd.DataFrame(review_rows, columns=REVIEW_QUEUE_COLUMNS)
    inventory_path = report_base / "paper_inventory.csv"
    review_queue_path = report_base / "paper_review_queue.csv"
    archive_path = report_base / "paper_archive_inventory.csv"
    audit_path = report_base / "paper_intake_audit.md"
    atomic_write_csv(inventory, inventory_path, index=False)
    atomic_write_csv(review_queue, review_queue_path, index=False)
    archives = _archive_members(source_dir)
    atomic_write_csv(archives, archive_path, index=False)
    atomic_write_text(audit_path, _intake_audit_markdown(inventory, archives, source_dir), encoding="utf-8")
    registry_result = build_research_source_registry(root=root)
    return CommandResult(
        paths={
            "paper_inventory": inventory_path,
            "paper_review_queue": review_queue_path,
            "paper_archive_inventory": archive_path,
            "paper_intake_audit": audit_path,
            **registry_result.paths,
        },
        summary={
            "papers": len(inventory),
            "unique_papers": int(inventory["duplicate_of"].astype(str).eq("").sum()),
            "duplicate_papers": int(inventory["duplicate_of"].astype(str).ne("").sum()),
            "complete_extractions": int(inventory["extraction_status"].eq("complete").sum()),
            "quarantined_documents": int(inventory["pdf_javascript"].astype(bool).sum()),
            "archive_members": len(archives),
        },
    )


def _intake_audit_markdown(
    inventory: pd.DataFrame, archives: pd.DataFrame, source_dir: Path
) -> str:
    complete = (
        int(inventory["extraction_status"].eq("complete").sum()) if not inventory.empty else 0
    )
    duplicate = (
        int(inventory["duplicate_of"].astype(str).ne("").sum()) if not inventory.empty else 0
    )
    javascript = int(inventory["pdf_javascript"].astype(bool).sum()) if not inventory.empty else 0
    lines = [
        "# Paper Intake Audit",
        "",
        f"- source directory: `{source_dir}`",
        f"- substantive PDFs: {len(inventory)}",
        f"- complete text extractions: {complete}",
        f"- duplicate PDFs: {duplicate}",
        f"- PDF JavaScript quarantines: {javascript}",
        f"- archive members inventoried: {len(archives)}",
        "- original source files moved: no",
        "- document instruction authority: none",
        "- live-signal eligibility: none",
        "",
        "## Gate",
        "",
        (
            "All papers remain provisional and research-only until claim-level gap analysis, "
            "premortem, red-team, math review, and local replication are complete."
        ),
    ]
    return "\n".join(lines) + "\n"
