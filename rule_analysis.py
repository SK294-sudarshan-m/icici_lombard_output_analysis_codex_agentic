"""Standalone no-AI analysis mode for case verdict reporting.

This module exists for ICICI Lombard environments where the analyzer must run
without Codex, OpenAI, Bedrock, Node, or any local AI model.  It still reads the
same full corpus via ``CaseCorpusBuilder`` so every nested JSON field, raw text
line, page artifact and accepted PDF is represented in the audit.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .model_corpus import (
        ROW_HEADERS,
        CaseCorpusBuilder,
        EvidenceChunk,
        _case_verdict,
        redact_text,
    )
except ImportError:  # Direct script execution.
    from model_corpus import (
        ROW_HEADERS,
        CaseCorpusBuilder,
        EvidenceChunk,
        _case_verdict,
        redact_text,
    )


AMOUNT_TOLERANCE = 5.0
CONFIDENCE_THRESHOLD = 0.90


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _compact(value: Any) -> str:
    if value in (None, ""):
        return "not recorded"
    return str(value)


def _scan_chunks(chunks: list[EvidenceChunk]) -> dict[str, Any]:
    """Scan every chunk for safe pipeline/numeric evidence."""
    counters: Counter[str] = Counter()
    snippets: dict[str, list[str]] = {
        "amount": [],
        "confidence": [],
        "aggregation": [],
        "returns": [],
        "classification": [],
        "extraction": [],
    }
    patterns = {
        "amount": re.compile(r"amount|gross|net|discount|tax|payable|calculated|extracted", re.I),
        "confidence": re.compile(r"confidence|flagged_for_review|flag_reason|review", re.I),
        "aggregation": re.compile(r"aggregation_match|within_tolerance|merged_bills|bill_level", re.I),
        "returns": re.compile(r"return|credit|negative|refund", re.I),
        "classification": re.compile(r"classification|bill_category|page_type|document_markers", re.I),
        "extraction": re.compile(r"page_\d+|ocr|raw_text|line_item|charges", re.I),
    }
    for chunk in chunks:
        for line in chunk.text.splitlines():
            safe = redact_text(line.strip())
            if not safe:
                continue
            for label, pattern in patterns.items():
                if pattern.search(safe):
                    counters[label] += 1
                    if len(snippets[label]) < 8:
                        snippets[label].append(safe[:320])
    return {
        "keyword_counts": dict(counters),
        "evidence_snippets": snippets,
        "chunks_scanned": len(chunks),
        "chunk_sha256": [chunk.sha256 for chunk in chunks],
    }


def _contains_amount_mismatch(facts: dict[str, Any]) -> bool:
    diff = _as_float(facts.get("amount_diff"))
    return facts.get("amounts_match") is False or (diff is not None and abs(diff) > AMOUNT_TOLERANCE)


def _contains_confidence_issue(facts: dict[str, Any]) -> bool:
    confidence = _as_float(facts.get("overall_confidence"))
    return (
        confidence is not None
        and confidence <= CONFIDENCE_THRESHOLD
    ) or bool(facts.get("root_flagged_for_review")) or bool(facts.get("review_total_flagged"))


def _contains_extraction_gap(facts: dict[str, Any], evidence: dict[str, Any]) -> bool:
    line_items = _as_float(facts.get("total_line_items"))
    extracted = _as_float(facts.get("extracted_amount"))
    extraction_hits = (evidence.get("keyword_counts") or {}).get("extraction", 0)
    return (
        line_items is not None
        and line_items <= 0
        and extracted is not None
        and extracted > 0
    ) or extraction_hits == 0


def _verdict_zero_labels(facts: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if _contains_amount_mismatch(facts):
        labels.append("amount reconciliation mismatch")
    if facts.get("aggregation_mismatches"):
        labels.append("supporting bill aggregation mismatch")
    if _contains_confidence_issue(facts):
        labels.append("confidence/review flag")
    if bool(facts.get("net_corrected")):
        labels.append("net payable correction due to discount/return/deduction components")
    if _contains_extraction_gap(facts, evidence):
        labels.append("possible extraction or line-item coverage gap")
    if not labels:
        labels.append("consolidated output set case_verdict=0 through review or reconciliation guard")
    return labels


def _pipeline_stage(labels: list[str]) -> str:
    stages: list[str] = []
    if any("extraction" in label or "line-item" in label for label in labels):
        stages.append("stage_1_extraction_ocr")
    if any("aggregation" in label for label in labels):
        stages.append("stage_4_5_sequencing_merging")
    if any("amount" in label or "net payable" in label for label in labels):
        stages.append("stage_6_7_consolidation_confidence")
    if any("confidence" in label for label in labels):
        stages.append("confidence/adjudication review gate")
    return "; ".join(dict.fromkeys(stages)) or "consolidated_final.json verdict gate"


def _numeric_fact_sentence(facts: dict[str, Any]) -> str:
    return (
        "Stored gate facts: "
        f"amounts_match={_compact(facts.get('amounts_match'))}, "
        f"extracted_amount={_compact(facts.get('extracted_amount'))}, "
        f"calculated_amount={_compact(facts.get('calculated_amount'))}, "
        f"extracted_after_discount={_compact(facts.get('extracted_after_discount'))}, "
        f"calculated_after_discount={_compact(facts.get('calculated_after_discount'))}, "
        f"amount_diff={_compact(facts.get('amount_diff'))}, "
        f"printed_net_payable={_compact(facts.get('printed_net_payable'))}, "
        f"overall_confidence={_compact(facts.get('overall_confidence'))}, "
        f"total_line_items={_compact(facts.get('total_line_items'))}, "
        f"review_total_flagged={_compact(facts.get('review_total_flagged'))}."
    )


def _safe_join(items: list[str]) -> str:
    return "; ".join(redact_text(item) for item in items if item)


def _build_row(
    case_id: str,
    verdict: int,
    facts: dict[str, Any],
    evidence: dict[str, Any],
    *,
    standalone_pdf_count: int,
) -> dict[str, Any]:
    if verdict == 1:
        row = {
            "AL-number": case_id,
            "case_verdict": 1,
            "what caused case verdict 0": "Not applicable: stored case_verdict is 1, so no rejection gate fired.",
            "how it happened": _numeric_fact_sentence(facts),
            "where it happened(step in pipeline)": "consolidated_final.json verdict gate passed after consolidation/confidence checks",
            "why it happened(actual reasons)": (
                "The consolidated facts did not show a blocking amount mismatch or active review condition. "
                f"Full-artifact scan covered {evidence.get('chunks_scanned', 0)} evidence chunks."
            ),
            "what can be improved": (
                "Use this case as a positive control when comparing rejected cases with similar bill totals, "
                "line-item density, discounts, returns and confidence signals."
            ),
        }
        return row

    labels = _verdict_zero_labels(facts, evidence)
    snippets = evidence.get("evidence_snippets") or {}
    useful_snippets = []
    for key in ("amount", "aggregation", "confidence", "returns", "classification", "extraction"):
        useful_snippets.extend((snippets.get(key) or [])[:2])
    pdf_note = (
        f" Standalone PDF context accepted: {standalone_pdf_count} PDF(s)."
        if standalone_pdf_count
        else ""
    )
    row = {
        "AL-number": case_id,
        "case_verdict": 0,
        "what caused case verdict 0": _safe_join(labels),
        "how it happened": _numeric_fact_sentence(facts) + pdf_note,
        "where it happened(step in pipeline)": _pipeline_stage(labels),
        "why it happened(actual reasons)": (
            "The deterministic verdict gate rejected the case because the consolidated numeric/review facts "
            "showed: "
            + _safe_join(labels)
            + ". Supporting evidence patterns were found across the full artifact corpus: "
            + json.dumps(evidence.get("keyword_counts") or {}, ensure_ascii=False)
            + ". Sample safe evidence: "
            + _safe_join(useful_snippets[:6])
        ),
        "what can be improved": (
            "Operationally compare rejected cases against passed controls by bill scope, line-item coverage, "
            "discount/return handling, aggregation tolerance and confidence flags. Improve document/page "
            "triage, line-item completeness checks, return-credit sign handling, final-bill versus supporting-bill "
            "scope checks, and human-review routing for low-confidence or mismatch-heavy cases."
        ),
    }
    return row


def _validate_row(row: dict[str, Any]) -> dict[str, Any]:
    for header in ROW_HEADERS:
        if header not in row:
            row[header] = ""
    for header in ROW_HEADERS[2:]:
        row[header] = redact_text(str(row.get(header) or "").strip()) or "Not recorded by standalone rule analysis."
    return {header: row[header] for header in ROW_HEADERS}


def generate_rule_rows(
    cases: list[tuple[str, Path, list[Path]]],
    *,
    standalone_pdfs: list[Path],
    output_dir: Path,
    chunk_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate the seven-column report without any AI/model dependency."""
    _ = output_dir
    builder = CaseCorpusBuilder(chunk_chars=chunk_chars)
    corpus_audits: dict[str, dict[str, Any]] = {}
    rule_evidence: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    standalone_pdf_audit: dict[str, Any] | None = None
    standalone_pdf_evidence: dict[str, Any] | None = None
    if standalone_pdfs:
        pdf_chunks, pdf_audit = builder.build_external_files(
            "standalone_pdf_inputs",
            standalone_pdfs,
        )
        standalone_pdf_audit = pdf_audit.to_dict()
        standalone_pdf_evidence = _scan_chunks(pdf_chunks)

    for case_id, case_root, matched_pdfs in cases:
        verdict, facts = _case_verdict(case_root)
        chunks, corpus_audit = builder.build(case_id, case_root, extra_files=matched_pdfs)
        evidence = _scan_chunks(chunks)
        corpus_audits[case_id] = corpus_audit.to_dict()
        rule_evidence[case_id] = evidence
        rows.append(
            _validate_row(
                _build_row(
                    case_id,
                    verdict,
                    facts,
                    evidence,
                    standalone_pdf_count=len(standalone_pdfs),
                )
            )
        )

    if standalone_pdf_audit:
        corpus_audits["standalone_pdf_inputs"] = standalone_pdf_audit
    if standalone_pdf_evidence:
        rule_evidence["standalone_pdf_inputs"] = standalone_pdf_evidence

    verdict_counts = Counter(row["case_verdict"] for row in rows)
    audit = {
        "model_runtime": {
            "provider": "none",
            "analysis_mode": "standalone_rules",
            "language_model_id": None,
            "embedding_model_id": None,
            "settings_source": "case_verdict_analysis_agent rule engine",
            "imports_parent_project_config": False,
            "imports_parent_project_pipeline_clients": False,
            "uses_codex": False,
            "uses_external_ai": False,
            "uses_local_ai_model": False,
        },
        "embedding_runtime": {
            "provider": "none",
            "used": False,
            "uses_external_ai": False,
            "uses_local_ai_model": False,
        },
        "corpus_coverage": corpus_audits,
        "rule_evidence": rule_evidence,
        "cohort": {
            "case_count": len(rows),
            "verdict_0_count": int(verdict_counts.get(0, 0)),
            "verdict_1_count": int(verdict_counts.get(1, 0)),
            "standalone_pdf_context_count": len(standalone_pdfs),
        },
        "all_json_fields_and_raw_artifacts_model_processed": all(
            value["all_artifacts_represented"] for value in corpus_audits.values()
        ),
        "all_json_fields_and_raw_artifacts_processed": all(
            value["all_artifacts_represented"] for value in corpus_audits.values()
        ),
        "row_source": "standalone deterministic rule analysis; no Codex, external AI, Bedrock, or local AI model",
    }
    return rows, audit
