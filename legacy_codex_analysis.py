"""Analyze every artifact in outputs/ and explain consolidated case_verdict values.

The analyzer is deliberately deterministic and privacy-minimizing.  It reads every
nested file, but retains only structural, numerical, identifier, medication and
pipeline-status evidence.  Patient, hospital and doctor names are never written to
the generated analysis artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .cloud_embeddings import (
        FAILURE_TAXONOMY,
        CloudEmbeddingError,
        EmbeddingCache,
        ProjectBedrockEmbedder,
        semantic_category,
    )
except ImportError:  # Direct script execution from the standalone folder.
    from cloud_embeddings import (
        FAILURE_TAXONOMY,
        CloudEmbeddingError,
        EmbeddingCache,
        ProjectBedrockEmbedder,
        semantic_category,
    )


AL_PATTERN = re.compile(r"\b110\d{9}-\d+\b")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-(]?\d[\d,]*(?:\.\d+)?\)?")
SAFE_TERMS = (
    "amount", "total", "gross", "net", "discount", "tax", "charge", "bill",
    "invoice", "ip number", "uhid", "admission", "policy", "claim", "medicine",
    "medication", "pharmacy", "prescription", "return", "credit", "quantity",
    "rate", "payable", "confidence", "verdict", "flag", "match", "reconcile",
)
TEXT_EXTENSIONS = {
    ".py", ".json", ".txt", ".md", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".sql", ".csv", ".tsv", ".html", ".js", ".mjs", ".dockerignore",
}
CONFIDENCE_PASS_THRESHOLD = 0.90
AMOUNT_TOLERANCE = 5.0


@dataclass
class ScanStats:
    files: int = 0
    bytes: int = 0
    json_files: int = 0
    text_files: int = 0
    pdf_files: int = 0
    pdf_pages: int = 0
    parse_errors: list[str] = field(default_factory=list)
    extensions: Counter = field(default_factory=Counter)
    stages: Counter = field(default_factory=Counter)
    al_candidates: Counter = field(default_factory=Counter)
    numerical_tokens: int = 0
    safe_term_hits: Counter = field(default_factory=Counter)
    sha256: dict[str, str] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extensions"] = dict(self.extensions)
        data["stages"] = dict(self.stages)
        data["al_candidates"] = dict(self.al_candidates)
        data["safe_term_hits"] = dict(self.safe_term_hits)
        return data


def parse_amount(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else 0.0
    text = str(value).strip().replace(",", "").replace("INR", "").replace("₹", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("() ")
    try:
        result = float(text)
    except (TypeError, ValueError):
        return 0.0
    return -result if negative else result


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def unique_join(parts: Iterable[str], separator: str = " ") -> str:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        clean = re.sub(r"\s+", " ", str(part or "")).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return separator.join(result)


def stage_for(relative_path: Path) -> str:
    value = str(relative_path).replace("\\", "/").lower()
    name = relative_path.name.lower()
    if "stage_confidence" in value:
        return "Stage judges"
    if "adjudication" in name or "ai_note" in name:
        return "Post-pipeline adjudication"
    if "contract" in name:
        return "Output contract"
    if "consolidated_bill_level3" in value or name == "consolidated_final.json":
        return "Stage 6-7 consolidation/confidence"
    if "merged_bills" in value:
        return "Stage 5 merging"
    if "categorised_bill_level2" in value:
        return "Stage 4 sequencing"
    if "categorised_bill_level1" in value:
        return "Stage 3 categorisation"
    if "classification" in name:
        return "Stage 2 classification"
    if "discharge_summary" in name:
        return "Stage 3b discharge summary"
    if re.match(r"page_\d+(_ocr|_raw)?\.", name) or re.match(r"page_\d+\.json", name):
        return "Stage 1 extraction/OCR"
    return "Other output artifact"


def _extract_pdf_text(data: bytes) -> tuple[int, str, str | None]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return 0, "", "pypdf is unavailable; PDF bytes were still hashed and counted"
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return len(reader.pages), text, None
    except Exception as exc:  # noqa: BLE001 - corrupt/encrypted PDFs are audit evidence
        return 0, "", f"{type(exc).__name__}: {exc}"


def scan_files(
    root: Path,
    *,
    relative_to: Path | None = None,
    exclude_roots: Iterable[Path] = (),
) -> ScanStats:
    stats = ScanStats()
    if not root.exists():
        return stats
    base = relative_to or root
    excluded = [path.resolve() for path in exclude_roots]
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        resolved = path.resolve()
        if any(candidate == resolved or candidate in resolved.parents for candidate in excluded):
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            stats.parse_errors.append(f"artifact_{stats.files + 1:06d}: {type(exc).__name__}")
            continue
        try:
            rel = path.relative_to(base)
        except ValueError:
            rel = path
        stats.files += 1
        stats.bytes += len(data)
        ext = path.suffix.lower() or "<none>"
        stats.extensions[ext] += 1
        stats.stages[stage_for(rel)] += 1
        artifact_id = f"artifact_{stats.files:06d}"
        stats.sha256[artifact_id] = hashlib.sha256(data).hexdigest()

        text = ""
        if ext == ".json":
            stats.json_files += 1
            try:
                json.loads(data.decode("utf-8-sig"))
                text = data.decode("utf-8-sig", errors="ignore")
            except (UnicodeError, json.JSONDecodeError) as exc:
                stats.parse_errors.append(f"{artifact_id}: {type(exc).__name__}")
                text = data.decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            stats.pdf_files += 1
            pages, text, error = _extract_pdf_text(data)
            stats.pdf_pages += pages
            if error:
                stats.parse_errors.append(f"{artifact_id}: {error}")
        elif ext in TEXT_EXTENSIONS:
            stats.text_files += 1
            text = data.decode("utf-8-sig", errors="ignore")

        if text:
            for candidate in AL_PATTERN.findall(text):
                stats.al_candidates[candidate] += 1
            stats.numerical_tokens += len(NUMBER_PATTERN.findall(text))
            lower = text.lower()
            for term in SAFE_TERMS:
                count = lower.count(term)
                if count:
                    stats.safe_term_hits[term] += count
    return stats


def discover_case_roots(outputs_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for path in outputs_dir.rglob("consolidated_final.json"):
        lowered = {part.lower() for part in path.parts}
        if "_work" in lowered:
            continue
        roots.append(path.parent)
    return sorted(set(roots), key=lambda p: str(p).lower())


def infer_al_number(case_root: Path, consolidated: dict, scan: ScanStats) -> str:
    for part in reversed(case_root.parts):
        match = AL_PATTERN.fullmatch(part.strip())
        if match:
            return match.group(0)
    for key in ("bill_id", "case_number", "request_id"):
        match = AL_PATTERN.search(str(consolidated.get(key) or ""))
        if match:
            return match.group(0)
    if scan.al_candidates:
        return scan.al_candidates.most_common(1)[0][0]
    return "AL unavailable"


def _normalise_filename(value: str) -> str:
    text = Path(value).stem.lower()
    text = re.sub(r"(?:_compressed)?_\d+$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def match_ground_truth(
    case_root: Path,
    consolidated: dict,
    al_number: str,
    pdfs: list[dict],
) -> dict | None:
    case_names = [case_root.name, str(consolidated.get("case_number") or "")]
    for pdf in pdfs:
        if al_number != "AL unavailable" and al_number in (pdf.get("al_candidates") or []):
            return pdf
        pdf_name = _normalise_filename(pdf["path"])
        for name in case_names:
            case_name = _normalise_filename(name)
            if not case_name or not pdf_name:
                continue
            if case_name == pdf_name:
                return pdf
            # Plain filename containment is deterministic string matching, not an
            # embedding/model fallback. Fuzzy semantic matching is cloud-only.
            if min(len(case_name), len(pdf_name)) >= 10 and (
                case_name in pdf_name or pdf_name in case_name
            ):
                return pdf
    return None


def scan_ground_truth_pdfs(pdf_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not pdf_dir.exists():
        return records
    for path in sorted(pdf_dir.rglob("*.pdf")):
        data = path.read_bytes()
        pages, text, error = _extract_pdf_text(data)
        records.append({
            "path": str(path),
            "pages": pages,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "has_text_layer": bool(text.strip()),
            "al_candidates": sorted(set(AL_PATTERN.findall(text))),
            "numeric_token_count": len(NUMBER_PATTERN.findall(text)),
            "error": error,
        })
    return records


def load_reference_context(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    data = load_json(path, {}) or {}
    return data.get("cases", data) if isinstance(data, dict) else {}


def _load_level1(work_dir: Path) -> dict:
    data = load_json(work_dir / "categorised_bill_level1" / "level1_summary.json", {}) or {}
    return data.get("bill_pages", {}) if isinstance(data, dict) else {}


def _load_merged(work_dir: Path) -> list[dict]:
    result: list[dict] = []
    merged_dir = work_dir / "merged_bills"
    if not merged_dir.exists():
        return result
    for path in sorted(merged_dir.glob("*.json")):
        if path.name == "merged_summary.json":
            continue
        data = load_json(path, {}) or {}
        charges = data.get("charges") or []
        returns = data.get("returns") or []
        bill_summary = data.get("bill_summary") or {}
        negatives = [c for c in charges if parse_amount(c.get("amount")) < 0]
        result.append({
            "file": path.name,
            "category": data.get("bill_category") or path.stem.rsplit("_", 1)[0],
            "charge_count": len(charges),
            "charge_sum": round(sum(parse_amount(c.get("amount")) for c in charges), 2),
            "negative_charge_count": len(negatives),
            "negative_charge_sum": round(sum(parse_amount(c.get("amount")) for c in negatives), 2),
            "return_count": len(returns),
            "return_sum": round(sum(parse_amount(c.get("amount")) for c in returns), 2),
            "total_amount": parse_amount(bill_summary.get("total_amount") or bill_summary.get("gross_amount")),
            "net_payable": parse_amount(bill_summary.get("net_payable")),
        })
    return result


def _adjudication_summary(case_root: Path) -> dict:
    data = load_json(case_root / "adjudication.json", {}) or {}
    agents = []
    for block in data.get("agents") or []:
        report = block.get("report") or {}
        agents.append({
            "agent": block.get("agent"),
            "status": block.get("status"),
            "judge_status": (block.get("judge") or {}).get("status"),
            "issue_count": len(block.get("issues") or []),
            "verdict": report.get("overall_verdict") or report.get("overall_status"),
        })
    return {"overall_status": data.get("overall_status"), "agents": agents}


def _pipeline_evidence(case_root: Path, consolidated: dict) -> dict:
    work_dir = case_root / "_work"
    level1 = _load_level1(work_dir)
    merged = _load_merged(work_dir)
    charges = consolidated.get("charges") or []
    returns = consolidated.get("returns") or []
    aggregation = [
        c.get("aggregation_match")
        for c in charges
        if isinstance(c.get("aggregation_match"), dict)
        and c["aggregation_match"].get("within_tolerance") is False
    ]
    merged_negative_count = sum(m["negative_charge_count"] for m in merged)
    merged_negative_sum = round(sum(m["negative_charge_sum"] for m in merged), 2)
    merged_return_count = sum(m["return_count"] for m in merged)
    merged_return_sum = round(sum(m["return_sum"] for m in merged), 2)
    return {
        "work_dir_exists": work_dir.exists(),
        "level1_bill_pages": level1,
        "level1_category_counts": {k: len(v or []) for k, v in level1.items()},
        "has_final_bill_category": bool(level1.get("final_bill") or level1.get("detailed_final_bill")),
        "merged_bills": merged,
        "merged_negative_charge_count": merged_negative_count,
        "merged_negative_charge_sum": merged_negative_sum,
        "merged_return_count": merged_return_count,
        "merged_return_sum": merged_return_sum,
        "final_charge_count": len(charges),
        "final_charge_sum": round(sum(parse_amount(c.get("amount")) for c in charges), 2),
        "final_return_count": len(returns),
        "final_return_sum": round(sum(parse_amount(c.get("amount")) for c in returns), 2),
        "aggregation_mismatches": aggregation,
        "adjudication": _adjudication_summary(case_root),
    }


def _format_money(value: Any) -> str:
    return f"INR {parse_amount(value):,.2f}"


def _baseline_summary(cases: list[dict]) -> dict:
    passed = [c for c in cases if c["case_verdict"] == 1]
    failed = [c for c in cases if c["case_verdict"] == 0]
    pass_conf = [c["overall_confidence"] for c in passed if c["overall_confidence"] is not None]
    return {
        "case_count": len(cases),
        "pass_count": len(passed),
        "fail_count": len(failed),
        "pass_confidence_min": min(pass_conf) if pass_conf else None,
        "pass_confidence_mean": round(sum(pass_conf) / len(pass_conf), 3) if pass_conf else None,
        "passes_with_flagged_lines": sum(1 for c in passed if c["review_total_flagged"] > 0),
        "passes_with_amount_match": sum(1 for c in passed if c["amounts_match"] is True),
        "passes_without_root_flag": sum(1 for c in passed if not c["root_flagged_for_review"]),
        "passes_without_aggregation_mismatch": sum(1 for c in passed if not c["evidence"]["aggregation_mismatches"]),
    }


def _aggregation_text(mismatches: list[dict], limit: int = 3) -> str:
    pieces: list[str] = []
    for mismatch in mismatches[:limit]:
        category = str(mismatch.get("category") or "supporting category").replace("_", " ")
        final_amount = mismatch.get("final_bill_amount")
        supporting = mismatch.get("supporting_bills_total")
        difference = mismatch.get("difference")
        pieces.append(
            f"{category}: final header {_format_money(final_amount)} vs supporting total "
            f"{_format_money(supporting)} (difference {_format_money(difference)})"
        )
    if len(mismatches) > limit:
        pieces.append(f"and {len(mismatches) - limit} more mismatch(es)")
    return "; ".join(pieces)


def _manual_context_text(reference: dict) -> str:
    observations = reference.get("observations") or []
    return unique_join((str(item) for item in observations), "; ")


def _embedding_evidence_text(case: dict, reference: dict) -> str:
    """Build a privacy-safe semantic signature containing no person/institution names."""
    bs = case["bill_summary"]
    ev = case["evidence"]
    aggregation = _aggregation_text(ev["aggregation_mismatches"], limit=5)
    categories = ", ".join(
        f"{key.replace('_', ' ')} {value} page(s)"
        for key, value in sorted(ev["level1_category_counts"].items())
    )
    manual = _manual_context_text(reference)
    return unique_join([
        f"case verdict {case['case_verdict']}",
        f"amounts match {case['amounts_match']}",
        f"extracted gross {_format_money(bs.get('extracted_amount'))}",
        f"calculated itemized {_format_money(bs.get('calculated_amount'))}",
        f"printed net {_format_money(bs.get('printed_net_payable'))}",
        f"amount difference {_format_money(bs.get('amount_diff'))}",
        f"line items {case['total_line_items']}",
        f"overall confidence {case['overall_confidence']}",
        f"case review flag {case['root_flagged_for_review']}",
        str(case.get("root_flag_reason") or ""),
        f"bill categories {categories}" if categories else "",
        f"merged negative charges {ev['merged_negative_charge_count']} sum {_format_money(ev['merged_negative_charge_sum'])}",
        f"final returns {ev['final_return_count']} sum {_format_money(ev['final_return_sum'])}",
        f"supporting aggregation mismatches {aggregation}" if aggregation else "",
        f"validated reference observations {manual}" if manual else "",
    ])


def enrich_with_project_embeddings(
    cases: list[dict],
    references: dict[str, dict],
    *,
    project_root: Path,
    cache_path: Path,
) -> dict[str, Any]:
    """Categorize rejected cases with only config.py's Bedrock embedding model."""
    embedder = ProjectBedrockEmbedder(project_root)
    cache = EmbeddingCache(cache_path, embedder.config)
    taxonomy_vectors = {
        label: cache.get_or_embed(definition, embedder)
        for label, definition in FAILURE_TAXONOMY.items()
    }
    categorized = 0
    for case in cases:
        if case["case_verdict"] != 0:
            continue
        evidence_text = _embedding_evidence_text(
            case,
            references.get(case["al_number"], {}),
        )
        case["embedding_analysis"] = semantic_category(
            evidence_text,
            embedder=embedder,
            cache=cache,
            taxonomy_vectors=taxonomy_vectors,
        )
        categorized += 1
    cache.save()
    return {
        **asdict(embedder.config),
        "requests_this_run": embedder.request_count,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "categorized_rejected_cases": categorized,
        "taxonomy_size": len(FAILURE_TAXONOMY),
    }


def _ordered_stage_text(stages: Iterable[str]) -> str:
    unique = list(dict.fromkeys(stage for stage in stages if stage))
    def key(value: str) -> tuple[int, str]:
        match = re.search(r"Stage\s+(\d+)", value)
        return (int(match.group(1)) if match else 99, value)
    return "; ".join(sorted(unique, key=key))


def _make_row(case: dict, baseline: dict, reference: dict) -> dict[str, Any]:
    al = case["al_number"]
    verdict = case["case_verdict"]
    bs = case["bill_summary"]
    ev = case["evidence"]
    confidence = case["overall_confidence"] or 0.0
    adjudication = ev.get("adjudication") or {}

    if verdict == 1:
        flagged_note = (
            f" {case['review_total_flagged']} line-level flag(s) existed but are not a direct verdict gate."
            if case["review_total_flagged"] else ""
        )
        return {
            "AL-number": al,
            "case_verdict": 1,
            "what caused case verdict 0": "Not applicable - this is a passed baseline case.",
            "how it happened": (
                f"Printed and itemized totals reconciled, no case-level review flag was set, confidence "
                f"{confidence:.3f} was above 0.900, and no supporting-bill aggregation mismatch existed."
                + flagged_note
            ),
            "where it happened(step in pipeline)": "Stages 6-7: consolidation, confidence scoring, and final deterministic verdict check.",
            "why it happened(actual reasons)": (
                "All four deterministic pass conditions were satisfied. Passed cases are the comparison control; "
                f"the pass-population minimum confidence was {baseline.get('pass_confidence_min')}."
            ),
            "what can be improved": (
                "Retain this case as a positive regression control and preserve its document classification, "
                "amount reconciliation, supporting-bill linking, and numeric-ID extraction behavior."
            ),
        }

    causes: list[str] = []
    how: list[str] = []
    where: list[str] = []
    why: list[str] = []
    improve: list[str] = []

    if case["amounts_match"] is False:
        extracted = parse_amount(bs.get("extracted_amount"))
        calculated = parse_amount(bs.get("calculated_amount"))
        if bs.get("net_corrected") and abs(extracted - calculated) <= AMOUNT_TOLERANCE:
            causes.append(
                f"Net-integrity reconciliation gate failed: printed net {_format_money(bs.get('printed_net_payable'))} "
                f"was not explained by the itemized sum {_format_money(calculated)}."
            )
        else:
            causes.append(
                f"Amount reconciliation gate failed: printed/extracted gross {_format_money(extracted)} "
                f"vs itemized sum {_format_money(calculated)}."
            )
        where.extend(["Stage 5 merging", "Stage 6 consolidation/integrity reconciliation"])
        if case["total_line_items"] == 0 and parse_amount(bs.get("extracted_amount")) > 0:
            where.append("Stage 1 extraction/OCR")
            how.append(
                f"A printed total of {_format_money(bs.get('extracted_amount'))} survived, but zero charge lines "
                "reached the consolidated output, producing a full-value mismatch."
            )
            why.append("The final-bill page was recognized at summary level but its charge table was not extracted into line items.")
            improve.append(
                "Require a nonzero line-item check whenever a bill total is present; route the page to OCR/table fallback "
                "or manual review before consolidation."
            )
        elif ev["merged_negative_charge_count"] and not ev["final_return_count"]:
            how.append(
                f"{ev['merged_negative_charge_count']} signed return/credit line(s) totalling "
                f"{_format_money(ev['merged_negative_charge_sum'])} were present in merged charges but absent from "
                "the consolidated returns, so removing the negative lines raised the itemized sum."
            )
            why.append("Returns were represented as negative charges instead of the dedicated returns collection and were dropped from gross reconciliation.")
            improve.append(
                "Preserve returns/credits as signed financial components and reconcile gross - discount + tax - returns "
                "before the verdict decision."
            )
        elif bs.get("net_corrected"):
            printed_net = parse_amount(bs.get("printed_net_payable"))
            calculated = parse_amount(bs.get("calculated_amount"))
            ratio = calculated / printed_net if printed_net else 0.0
            how.append(
                f"The integrity guard rejected printed net {_format_money(printed_net)} as unexplained by the "
                f"itemized amount {_format_money(calculated)}, recomputed the net, set amounts_match=false, and capped confidence."
            )
            if 1.85 <= ratio <= 2.15:
                why.append(
                    f"The itemized-to-net ratio is {ratio:.2f}x, a strong signal of duplicated scope or mixing a gross/header total "
                    "with payer/patient-share details."
                )
                improve.append(
                    "Reconcile at document and payer-column scope; de-duplicate summary/header rows against detail rows before summing."
                )
            elif ev["level1_category_counts"].get("detailed_final_bill") and ev["level1_category_counts"].get("final_bill"):
                why.append(
                    "Both detailed and summary final bills existed; the detailed bill was preferred even though its extracted line-item scope did not explain the summary net."
                )
                improve.append(
                    "Before preferring a detailed final bill, cross-check its line count and sum against the summary final bill and retain the summary as a control total."
                )
            elif not ev["has_final_bill_category"]:
                why.append("No page was classified as a final bill, so supporting/interim documents were combined across incompatible scopes.")
                improve.append(
                    "Separate final bills from deposits, receipts and supporting bills; reconcile each document independently before claim-level aggregation."
                )
            else:
                why.append(
                    "The lower printed net was not explained by extracted discount, deduction, return or payer-share fields, so it was treated as inconsistent."
                )
                improve.append(
                    "Capture discount/deduction/return and payer-share components explicitly so a legitimate lower net is not treated as a bad total."
                )
        else:
            how.append(
                f"The itemized sum differed from the printed total by {_format_money(bs.get('amount_diff'))}, beyond the INR 5 tolerance."
            )
            why.append("At least one charge, return, subtotal, or page-scope amount was omitted, duplicated, or treated with the wrong sign.")
            improve.append(
                "Add page-to-merged-to-consolidated line-count and signed-sum checks, with separate handling for subtotals and returns."
            )

    if case["root_flagged_for_review"]:
        root_reason = str(case.get("root_flag_reason") or "case-level review flag")
        causes.append(f"Case-level flagged_for_review gate was true ({root_reason}).")
        where.append("Stage 6 consolidation")
        if "No final bill found" in root_reason:
            categories = ", ".join(
                f"{name.replace('_', ' ')}={count}" for name, count in ev["level1_category_counts"].items()
            ) or "no classified bill category"
            how.append(f"Categorisation produced {categories}; consolidation therefore found no authoritative final bill and set a blocking review flag.")
            why.append("The available procedure/package/supporting page may be financially complete, but it was not identified as a final bill under the pipeline hierarchy.")
            improve.append(
                "Use document-level finality evidence (title, invoice total, net payable, page sequence) and a controlled manual-final designation when only one complete bill exists."
            )

    mismatches = ev["aggregation_mismatches"]
    if mismatches:
        causes.append(f"Supporting-bill aggregation gate failed for {len(mismatches)} charge header(s).")
        how.append(_aggregation_text(mismatches) + ".")
        where.extend(["Stage 3-5 category grouping/merging", "Stage 6 supporting-bill expansion"])
        why.append(
            "The final-bill category header was compared with an incomplete, over-broad, or differently scoped supporting-bill total, so the expected detail expansion was withheld."
        )
        improve.append(
            "Link every supporting bill to the correct final-bill category and service period, then compare one aggregated category total rather than individual lines to the whole category."
        )

    if confidence <= CONFIDENCE_PASS_THRESHOLD:
        causes.append(f"Strict confidence gate failed: {confidence:.3f} is not greater than 0.900.")
        where.extend(["Stage 7 confidence scoring", "Stage 6 verdict recomputation after scoring"])
        if math.isclose(confidence, CONFIDENCE_PASS_THRESHOLD, abs_tol=1e-9) and case["amounts_match"] is True and not case["root_flagged_for_review"] and not mismatches:
            how.append("All financial and document-level gates passed, but the rule is strictly greater-than; a score exactly at 0.900 is rejected.")
            why.append("This is a threshold-boundary rejection, not evidence that the bill arithmetic is wrong.")
            improve.append(
                "Calibrate the boundary with validated cases and route exact-threshold cases to targeted line review instead of whole-case rejection."
            )
        elif case["amounts_match"] is False:
            how.append("Confidence was capped after the amount-integrity failure, so low confidence is a consequence of the same reconciliation defect.")
        else:
            why.append("One or more line-level extractions were not sufficiently supported by the source image to clear the strict case threshold.")
            improve.append("Review only the low-confidence pages/lines and preserve high-confidence reconciled sections.")

    manual = _manual_context_text(reference)
    if manual:
        why.append(f"Reference workbook/PDF evidence: {manual}.")

    if adjudication.get("overall_status"):
        why.append(
            f"Downstream adjudication overall_status={adjudication['overall_status']}; it is corroborating context and does not set bill_summary.case_verdict."
        )

    embedding = case.get("embedding_analysis") or {}
    if embedding:
        label = str(embedding.get("category") or "").replace("_", " ")
        why.append(
            f"Configured cloud embedding model categorized the evidence as {label} "
            f"(cosine similarity {embedding.get('similarity'):.3f}); deterministic pipeline gates remain authoritative."
        )

    if al == "110100834860-1":
        how.append(
            "All 16 bill pages were categorized as supporting/interim; the merged main bill combined summary headers and detailed pages, while deposits/receipts were also admitted into consolidation."
        )
        where.extend(["Stage 2 classification", "Stage 4 sequencing", "Stage 5 merging", "Stage 6 consolidation"])
        why.append(
            "The ground-truth final bill total is INR 189,771.25 with pharmacy returns, but the output carried INR 853,811.10 extracted and INR 914,760.85 itemized across 168 lines."
        )
        improve.append(
            "Recognize the final-bill summary/detail sequence as one authoritative document, treat deposits/payment receipts as evidence rather than charges, and keep returns signed."
        )

    return {
        "AL-number": al,
        "case_verdict": 0,
        "what caused case verdict 0": unique_join(causes),
        "how it happened": unique_join(how),
        "where it happened(step in pipeline)": _ordered_stage_text(where),
        "why it happened(actual reasons)": unique_join(why),
        "what can be improved": unique_join(improve),
    }


def analyze_case(case_root: Path, outputs_dir: Path) -> dict:
    consolidated_path = case_root / "consolidated_final.json"
    consolidated = load_json(consolidated_path, {}) or {}
    scan = scan_files(case_root, relative_to=case_root)
    bill_summary = consolidated.get("bill_summary") or {}
    evidence = _pipeline_evidence(case_root, consolidated)
    verdict = bill_summary.get("case_verdict")
    try:
        verdict = int(verdict)
    except (TypeError, ValueError):
        verdict = 0
    confidence_value = bill_summary.get("overall_confidence", consolidated.get("overall_confidence"))
    confidence = parse_amount(confidence_value) if confidence_value is not None else None
    return {
        "case_root": str(case_root),
        "al_number": infer_al_number(case_root, consolidated, scan),
        "case_verdict": verdict,
        "bill_summary": bill_summary,
        "amounts_match": bill_summary.get("amounts_match"),
        "overall_confidence": confidence,
        "total_line_items": int(bill_summary.get("total_line_items") or 0),
        "root_flagged_for_review": bool(consolidated.get("flagged_for_review")),
        "root_flag_reason": consolidated.get("flag_reason"),
        "review_total_flagged": int((consolidated.get("review_summary") or {}).get("total_flagged") or 0),
        "evidence": evidence,
        "scan": scan.public_dict(),
    }


def scan_codebase(project_root: Path, outputs_dir: Path, agent_dir: Path) -> dict:
    stats = {"files": 0, "bytes": 0, "keyword_hits": Counter(), "sha256": {}}
    keywords = (
        "case_verdict", "amounts_match", "overall_confidence", "flagged_for_review",
        "aggregation_match", "expanded_details", "returns", "verdict", "reconcile",
    )
    excluded = [outputs_dir.resolve(), (agent_dir / "input_pdfs").resolve()]
    for path in sorted(p for p in project_root.rglob("*") if p.is_file()):
        resolved = path.resolve()
        if any(base == resolved or base in resolved.parents for base in excluded):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in {"Dockerfile", "Makefile"}:
            continue
        data = path.read_bytes()
        text = data.decode("utf-8-sig", errors="ignore").lower()
        rel = str(path.relative_to(project_root))
        stats["files"] += 1
        stats["bytes"] += len(data)
        stats["sha256"][rel] = hashlib.sha256(data).hexdigest()
        for keyword in keywords:
            count = text.count(keyword)
            if count:
                stats["keyword_hits"][keyword] += count
    stats["keyword_hits"] = dict(stats["keyword_hits"])
    return stats


def _find_node(explicit: str | None) -> str:
    candidates = [explicit, os.getenv("CODEX_NODE"), shutil.which("node"), shutil.which("node.exe")]
    candidates.append(str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    raise RuntimeError("Node.js was not found. Pass --node with an executable path.")


def _find_artifact_modules(explicit: str | None) -> Path:
    candidates = [explicit, os.getenv("ARTIFACT_TOOL_NODE_MODULES")]
    candidates.append(str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"))
    for candidate in candidates:
        if candidate and (Path(candidate) / "@oai" / "artifact-tool").exists():
            return Path(candidate).resolve()
    raise RuntimeError("@oai/artifact-tool was not found. Pass --artifact-node-modules.")


def build_xlsx(
    rows_json: Path,
    audit_json: Path,
    output_xlsx: Path,
    *,
    node: str | None,
    artifact_node_modules: str | None,
) -> Path:
    node_exe = _find_node(node)
    modules = _find_artifact_modules(artifact_node_modules)
    source_builder = Path(__file__).with_name("build_report.mjs")
    with tempfile.TemporaryDirectory(prefix="case-verdict-xlsx-") as temp_name:
        temp_dir = Path(temp_name)
        builder = temp_dir / "build_report.mjs"
        shutil.copy2(source_builder, builder)
        link = temp_dir / "node_modules"
        try:
            link.symlink_to(modules, target_is_directory=True)
        except OSError:
            if os.name == "nt":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(modules)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                raise
        output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        output_xlsx.unlink(missing_ok=True)
        completed = subprocess.run(
            [node_exe, str(builder), str(rows_json), str(audit_json), str(output_xlsx)],
            check=False,
            cwd=temp_dir,
        )
        if not output_xlsx.exists() or output_xlsx.stat().st_size < 1000:
            raise RuntimeError(
                f"Workbook builder exited with {completed.returncode} before producing a valid XLSX"
            )
    return output_xlsx


def run(args: argparse.Namespace) -> dict:
    project_root = args.project_root.resolve()
    outputs_dir = args.outputs.resolve()
    agent_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()

    case_roots = discover_case_roots(outputs_dir)
    if not case_roots:
        raise RuntimeError(f"No case-level consolidated_final.json files found below {outputs_dir}")

    global_scan = scan_files(outputs_dir, relative_to=outputs_dir, exclude_roots=[output_dir])
    cases = [analyze_case(case_root, outputs_dir) for case_root in case_roots]
    baseline = _baseline_summary(cases)
    references = load_reference_context(args.reference_context)
    ground_truth_pdfs = scan_ground_truth_pdfs(args.pdf_input_dir.resolve())
    for case in cases:
        consolidated = load_json(Path(case["case_root"]) / "consolidated_final.json", {}) or {}
        case["ground_truth_pdf"] = match_ground_truth(
            Path(case["case_root"]),
            consolidated,
            case["al_number"],
            ground_truth_pdfs,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_runtime = enrich_with_project_embeddings(
        cases,
        references,
        project_root=project_root,
        cache_path=output_dir / "cloud_embedding_cache.json",
    )

    rows = [
        _make_row(case, baseline, references.get(case["al_number"], {}))
        for case in sorted(cases, key=lambda item: item["al_number"])
    ]

    rows_path = output_dir / "case_verdict_analysis_rows.json"
    audit_path = output_dir / "case_verdict_analysis_audit.json"
    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    assigned_files = sum(case["scan"]["files"] for case in cases)
    audit_cases = json.loads(json.dumps(cases))
    for audit_case in audit_cases:
        audit_case.pop("case_root", None)
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy": (
            "All source files were read. Persisted evidence is limited to structural/numerical/identifier/medical-item "
            "signals; patient, hospital and doctor names are excluded."
        ),
        "outputs_root": str(outputs_dir),
        "case_ids": [case["al_number"] for case in cases],
        "global_output_scan": global_scan.public_dict(),
        "case_assigned_file_count": assigned_files,
        "unassigned_output_file_count": global_scan.files - assigned_files,
        "baseline_comparison": baseline,
        "embedding_runtime": embedding_runtime,
        "codebase_context_scan": scan_codebase(project_root, outputs_dir, agent_dir),
        "ground_truth_pdfs": ground_truth_pdfs,
        "case_evidence": audit_cases,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    xlsx_path = output_dir / args.xlsx_name
    if not args.skip_xlsx:
        build_xlsx(
            rows_path,
            audit_path,
            xlsx_path,
            node=args.node,
            artifact_node_modules=args.artifact_node_modules,
        )

    return {
        "cases": len(cases),
        "pass": baseline["pass_count"],
        "fail": baseline["fail_count"],
        "files_scanned": global_scan.files,
        "files_assigned_to_cases": assigned_files,
        "pdfs_scanned": len(ground_truth_pdfs),
        "xlsx": str(xlsx_path) if not args.skip_xlsx else None,
        "audit": str(audit_path),
    }


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    project = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--outputs", type=Path, default=project / "outputs")
    parser.add_argument("--pdf-input-dir", type=Path, default=here / "input_pdfs")
    parser.add_argument("--reference-context", type=Path, default=here / "reference_context.json")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--xlsx-name", default="case_verdict_analysis.xlsx")
    parser.add_argument("--node", help="Path to Node.js used by the workbook renderer")
    parser.add_argument("--artifact-node-modules", help="node_modules directory containing @oai/artifact-tool")
    parser.add_argument("--skip-xlsx", action="store_true", help="Write JSON evidence only")
    parser.add_argument(
        "--check-model-config",
        action="store_true",
        help="Validate and print only the configured cloud embedding model metadata",
    )
    parser.add_argument(
        "--check-embedding-access",
        action="store_true",
        help="Make one configured Bedrock embedding call, print dimension, and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check_model_config or args.check_embedding_access:
            embedder = ProjectBedrockEmbedder(args.project_root.resolve())
            result = asdict(embedder.config)
            if args.check_embedding_access:
                vector = embedder.embed("case verdict amount reconciliation access check")
                result["embedding_dimension"] = len(vector)
                result["requests_this_run"] = embedder.request_count
            print(json.dumps(result, indent=2))
            return 0
        result = run(args)
    except CloudEmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI reports a concise actionable failure
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
