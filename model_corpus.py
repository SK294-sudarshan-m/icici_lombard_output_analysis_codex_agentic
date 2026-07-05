"""Full-artifact, model-driven claim verdict analysis.

Every nested JSON leaf and every line/page of text is redacted, chunked, and sent
to the project-configured Bedrock language model.  The project-configured Titan
embedding model is used for cross-case retrieval.  There is no local model and no
deterministic narrative fallback.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from .cloud_embeddings import (
        CloudEmbeddingError,
        EmbeddingCache,
        ProjectBedrockEmbedder,
        cosine_similarity,
    )
except ImportError:  # Direct script execution.
    from cloud_embeddings import (
        CloudEmbeddingError,
        EmbeddingCache,
        ProjectBedrockEmbedder,
        cosine_similarity,
    )


AL_PATTERN = re.compile(r"\b110\d{9}-\d+\b")
SENSITIVE_PATH = re.compile(
    r"(?:^|\.)(?:patient(?:_?name)?|hospital(?:_?name)?|doctor(?:_?name)?|"
    r"consultant|physician|surgeon(?:_?name)?|address|email|phone|mobile)(?:\.|$|\[)",
    re.IGNORECASE,
)
SENSITIVE_LINE = re.compile(
    r"\b(?:patient\s*name|hospital\s*name|doctor\s*name|consultant|physician|"
    r"treating\s*doctor|residential\s*address|postal\s*address)\b",
    re.IGNORECASE,
)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
LONG_BASE64 = re.compile(r"^[A-Za-z0-9+/=\s]{1000,}$")

ROW_HEADERS = [
    "AL-number",
    "case_verdict",
    "what caused case verdict 0",
    "how it happened",
    "where it happened(step in pipeline)",
    "why it happened(actual reasons)",
    "what can be improved",
]


class ModelAnalysisError(RuntimeError):
    """Raised when configured model analysis cannot complete safely."""


def redact_text(value: str) -> str:
    text = str(value or "")
    if SENSITIVE_LINE.search(text):
        return "[REDACTED SENSITIVE LINE]"
    text = EMAIL.sub("[REDACTED EMAIL]", text)
    text = PHONE.sub("[REDACTED PHONE]", text)
    return text


def stage_for(relative_path: Path) -> str:
    value = str(relative_path).replace("\\", "/").lower()
    name = relative_path.name.lower()
    if "stage_confidence" in value:
        return "stage_judges"
    if "adjudication" in name or "ai_note" in name:
        return "post_pipeline_adjudication"
    if "contract" in name:
        return "output_contract"
    if "consolidated_bill_level3" in value or name == "consolidated_final.json":
        return "stage_6_7_consolidation_confidence"
    if "merged_bills" in value:
        return "stage_5_merging"
    if "categorised_bill_level2" in value:
        return "stage_4_sequencing"
    if "categorised_bill_level1" in value:
        return "stage_3_categorisation"
    if "classification" in name:
        return "stage_2_classification"
    if "discharge_summary" in name:
        return "stage_3b_discharge_summary"
    if re.match(r"page_\d+(_ocr|_raw)?\.", name) or re.match(r"page_\d+\.json", name):
        return "stage_1_extraction_ocr"
    return "other_output_artifact"


def _safe_scalar(path: str, value: Any) -> str:
    if SENSITIVE_PATH.search(path):
        return '"[REDACTED]"'
    if isinstance(value, str):
        clean = redact_text(value)
        if LONG_BASE64.match(clean):
            digest = hashlib.sha256(clean.encode("utf-8", errors="ignore")).hexdigest()
            clean = f"[BINARY_OR_BASE64 length={len(clean)} sha256={digest}]"
        return json.dumps(clean, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, default=str)


def flatten_json(value: Any, path: str = "$") -> list[str]:
    """Return one line per JSON leaf, preserving every nested field path."""
    lines: list[str] = []
    if isinstance(value, dict):
        if not value:
            lines.append(f"{path} = {{}}")
        for key, child in value.items():
            escaped = str(key).replace("\\", "\\\\").replace(".", "\\.")
            lines.extend(flatten_json(child, f"{path}.{escaped}"))
    elif isinstance(value, list):
        if not value:
            lines.append(f"{path} = []")
        for index, child in enumerate(value):
            lines.extend(flatten_json(child, f"{path}[{index}]"))
    else:
        lines.append(f"{path} = {_safe_scalar(path, value)}")
    return lines


def _pdf_pages(data: bytes) -> tuple[list[str], str | None]:
    try:
        import fitz

        document = fitz.open(stream=data, filetype="pdf")
        pages = [redact_text(page.get_text("text") or "") for page in document]
        document.close()
        return pages, None
    except ImportError:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            return [redact_text(page.extract_text() or "") for page in reader.pages], None
        except ImportError:
            return [], "Neither project PyMuPDF nor optional pypdf is available"
        except Exception as exc:  # noqa: BLE001
            return [], f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


@dataclass
class EvidenceRecord:
    artifact_id: str
    relative_path: str
    stage: str
    content_type: str
    text: str
    json_leaf_count: int = 0
    raw_line_count: int = 0
    pdf_page_count: int = 0


@dataclass
class EvidenceChunk:
    case_id: str
    index: int
    text: str
    artifact_ids: list[str]
    sha256: str


@dataclass
class CorpusAudit:
    case_id: str
    files_read: int = 0
    bytes_read: int = 0
    json_files: int = 0
    json_leaf_paths: int = 0
    raw_text_files: int = 0
    raw_lines: int = 0
    pdf_files: int = 0
    pdf_pages: int = 0
    page_specific_files: int = 0
    other_files: int = 0
    parse_errors: list[str] = field(default_factory=list)
    stages: Counter = field(default_factory=Counter)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    chunks: int = 0
    artifacts_represented_in_chunks: int = 0
    all_artifacts_represented: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = dict(self.stages)
        return data


class CaseCorpusBuilder:
    def __init__(self, *, chunk_chars: int = 20000) -> None:
        if chunk_chars < 4000:
            raise ValueError("chunk_chars must be at least 4000")
        self.chunk_chars = chunk_chars

    def _record_for_file(
        self,
        path: Path,
        *,
        relative_path: Path,
        artifact_id: str,
        audit: CorpusAudit,
    ) -> EvidenceRecord:
        data = path.read_bytes()
        audit.files_read += 1
        audit.bytes_read += len(data)
        audit.artifact_hashes[artifact_id] = hashlib.sha256(data).hexdigest()
        stage = stage_for(relative_path)
        audit.stages[stage] += 1
        if re.match(r"page_\d+", path.name.lower()):
            audit.page_specific_files += 1

        extension = path.suffix.lower()
        header = (
            f"ARTIFACT {artifact_id}\nRELATIVE_PATH {relative_path.as_posix()}\n"
            f"PIPELINE_STAGE {stage}\n"
        )
        if extension == ".json":
            audit.json_files += 1
            try:
                parsed = json.loads(data.decode("utf-8-sig"))
                leaves = flatten_json(parsed)
                audit.json_leaf_paths += len(leaves)
                text = header + "CONTENT_TYPE json_flattened_all_leaves\n" + "\n".join(leaves)
                return EvidenceRecord(
                    artifact_id,
                    relative_path.as_posix(),
                    stage,
                    "json",
                    text,
                    json_leaf_count=len(leaves),
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                audit.parse_errors.append(f"{artifact_id}: {type(exc).__name__}")
                decoded = data.decode("utf-8", errors="replace")
                lines = [redact_text(line) for line in decoded.splitlines()]
                audit.raw_lines += len(lines)
                return EvidenceRecord(
                    artifact_id,
                    relative_path.as_posix(),
                    stage,
                    "invalid_json_text",
                    header + "CONTENT_TYPE invalid_json_text\n" + "\n".join(lines),
                    raw_line_count=len(lines),
                )
        if extension == ".txt":
            audit.raw_text_files += 1
            lines = [redact_text(line) for line in data.decode("utf-8-sig", errors="replace").splitlines()]
            numbered = [f"LINE {index + 1}: {line}" for index, line in enumerate(lines)]
            audit.raw_lines += len(lines)
            return EvidenceRecord(
                artifact_id,
                relative_path.as_posix(),
                stage,
                "raw_text",
                header + "CONTENT_TYPE raw_text_all_lines\n" + "\n".join(numbered),
                raw_line_count=len(lines),
            )
        if extension == ".pdf":
            audit.pdf_files += 1
            pages, error = _pdf_pages(data)
            if error:
                audit.parse_errors.append(f"{artifact_id}: {error}")
            audit.pdf_pages += len(pages)
            page_text = "\n".join(
                f"PDF_PAGE {index + 1}:\n{text or '[IMAGE-ONLY OR NO TEXT LAYER]'}"
                for index, text in enumerate(pages)
            )
            return EvidenceRecord(
                artifact_id,
                relative_path.as_posix(),
                stage,
                "pdf_pages",
                header + "CONTENT_TYPE pdf_all_pages\n" + page_text,
                pdf_page_count=len(pages),
            )

        audit.other_files += 1
        return EvidenceRecord(
            artifact_id,
            relative_path.as_posix(),
            stage,
            "binary_metadata",
            header
            + "CONTENT_TYPE binary_metadata\n"
            + f"BYTE_LENGTH {len(data)}\nSHA256 {hashlib.sha256(data).hexdigest()}",
        )

    def build(
        self,
        case_id: str,
        case_root: Path,
        *,
        extra_files: Iterable[Path] = (),
    ) -> tuple[list[EvidenceChunk], CorpusAudit]:
        audit = CorpusAudit(case_id=case_id)
        records: list[EvidenceRecord] = []
        files = sorted(path for path in case_root.rglob("*") if path.is_file())
        for index, path in enumerate(files, start=1):
            records.append(
                self._record_for_file(
                    path,
                    relative_path=path.relative_to(case_root),
                    artifact_id=f"artifact_{index:06d}",
                    audit=audit,
                )
            )
        offset = len(records)
        for extra_index, path in enumerate(sorted(set(extra_files)), start=1):
            records.append(
                self._record_for_file(
                    path,
                    relative_path=Path("external_pdf_inputs") / path.name,
                    artifact_id=f"artifact_{offset + extra_index:06d}",
                    audit=audit,
                )
            )

        chunks: list[EvidenceChunk] = []
        represented: set[str] = set()
        current_parts: list[str] = []
        current_ids: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current_parts, current_ids, current_len
            if not current_parts:
                return
            text = "\n\n".join(current_parts)
            chunks.append(
                EvidenceChunk(
                    case_id=case_id,
                    index=len(chunks) + 1,
                    text=text,
                    artifact_ids=list(dict.fromkeys(current_ids)),
                    sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
            represented.update(current_ids)
            current_parts, current_ids, current_len = [], [], 0

        for record in records:
            text = record.text
            # Split oversize artifacts without dropping a single character.
            parts = [text[i:i + self.chunk_chars] for i in range(0, len(text), self.chunk_chars)] or [""]
            for part_index, part in enumerate(parts, start=1):
                wrapper = (
                    f"ARTIFACT_PART {record.artifact_id} {part_index}/{len(parts)}\n{part}"
                    if len(parts) > 1
                    else part
                )
                if current_parts and current_len + len(wrapper) > self.chunk_chars:
                    flush()
                current_parts.append(wrapper)
                current_ids.append(record.artifact_id)
                current_len += len(wrapper)
        flush()
        audit.chunks = len(chunks)
        audit.artifacts_represented_in_chunks = len(represented)
        audit.all_artifacts_represented = len(represented) == len(records)
        if not audit.all_artifacts_represented:
            raise ModelAnalysisError(f"Corpus coverage failure for case {case_id}")
        return chunks, audit


def _extract_json_object(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "```" in cleaned:
        candidates = [part.strip().lstrip("json").strip() for part in cleaned.split("```")]
        cleaned = next((part for part in candidates if part.startswith("{")), cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    parsed = json.loads(cleaned[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON was not an object")
    return parsed


class ConfiguredModelRuntime:
    """Only the language and embedding models declared by the project config."""

    def __init__(self, project_root: Path, cache_dir: Path) -> None:
        self.embedder = ProjectBedrockEmbedder(project_root)
        try:
            from config import settings
            from pipeline.clients import create_bedrock_client
        except Exception as exc:  # noqa: BLE001
            raise ModelAnalysisError("Could not load configured Bedrock language model") from exc
        self.region = str(settings.aws_region)
        self.language_model_id = str(settings.model_id)
        self.embedding_model_id = str(settings.med_embedding_model)
        if not self.language_model_id or not self.embedding_model_id:
            raise ModelAnalysisError("Both configured language and embedding model IDs are required")
        self.max_tokens = int(settings.max_tokens_judge)
        self.client = create_bedrock_client(self.region)
        self.cache_path = cache_dir / "configured_llm_cache.json"
        self._cache_lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self.language_requests = 0
        self.language_cache_hits = 0
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if data.get("model_id") == self.language_model_id:
                    self._cache = data.get("items") or {}
            except (OSError, UnicodeError, json.JSONDecodeError):
                self._cache = {}

    def call_json(self, purpose: str, prompt: str, *, max_tokens: int | None = None) -> dict:
        key = hashlib.sha256(
            f"{self.language_model_id}\0{purpose}\0{prompt}".encode("utf-8")
        ).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(key)
            if isinstance(cached, dict):
                self.language_cache_hits += 1
                return cached
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                response = self.client.converse(
                    modelId=self.language_model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={
                        "maxTokens": max_tokens or self.max_tokens,
                        "temperature": 0.0,
                    },
                )
                text = "".join(
                    block.get("text", "")
                    for block in response["output"]["message"]["content"]
                    if isinstance(block, dict)
                )
                parsed = _extract_json_object(text)
                with self._cache_lock:
                    self._cache[key] = parsed
                    self.language_requests += 1
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise ModelAnalysisError(
            "Configured Bedrock language-model analysis failed; no local or Codex fallback was used"
        ) from last_error

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_lock:
            self.cache_path.write_text(
                json.dumps(
                    {"model_id": self.language_model_id, "items": self._cache},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "provider": "AWS Bedrock",
            "region": self.region,
            "language_model_id": self.language_model_id,
            "embedding_model_id": self.embedding_model_id,
            "credential_source": "boto3 default credential provider chain",
            "local_model": False,
            "codex_generated_analysis": False,
            "language_requests": self.language_requests,
            "language_cache_hits": self.language_cache_hits,
        }


MAP_SCHEMA = {
    "artifact_findings": [{"evidence_refs": ["artifact_id:path"], "observation": "text"}],
    "verdict_causes": ["text"],
    "how_it_happened": ["text"],
    "pipeline_steps": ["text"],
    "actual_reasons": ["text"],
    "improvements": ["text"],
    "numeric_checks": ["text"],
    "warnings": ["text"],
}


def map_prompt(chunk: EvidenceChunk, total_chunks: int, case_verdict: int) -> str:
    return f"""You are the evidence-map stage of a healthcare claim extraction audit agent.
Analyze ALL evidence in this chunk. It contains flattened JSON leaf paths and/or every raw text line/page.
The evidence has already had patient, hospital and doctor identity values redacted.

Rules:
- Focus on numeric IDs, invoice/IP/UHID/claim numbers, charges, medicines, prescriptions, returns, discounts,
  taxes, page classification, sequencing, merging, consolidation, confidence and adjudication.
- Do not infer a name or institution. Never output patient/hospital/doctor names.
- Distinguish the direct deterministic case_verdict gate from the earliest upstream cause.
- Cite artifact IDs and JSON paths/line numbers in evidence_refs.
- Do not omit evidence merely because it conflicts with another artifact.
- Return only valid JSON matching this shape: {json.dumps(MAP_SCHEMA)}

CASE_ID: {chunk.case_id}
STORED_CASE_VERDICT: {case_verdict}
CHUNK: {chunk.index}/{total_chunks}
CHUNK_SHA256: {chunk.sha256}

BEGIN COMPLETE CHUNK EVIDENCE
{chunk.text}
END COMPLETE CHUNK EVIDENCE
"""


def reduce_prompt(case_id: str, summaries: list[dict], round_index: int) -> str:
    return f"""You are the hierarchical reduce stage of a healthcare claim extraction audit agent.
Merge EVERY supplied child analysis into one evidence-preserving JSON object. Keep contradictory evidence,
numeric values, artifact references and pipeline stages. Do not introduce facts not present in the children.
Never output patient, hospital or doctor names. Return only JSON matching: {json.dumps(MAP_SCHEMA)}

CASE_ID: {case_id}
REDUCE_ROUND: {round_index}
CHILD_ANALYSES:
{json.dumps(summaries, ensure_ascii=False)}
"""


def _batch(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def analyze_case_chunks(
    runtime: ConfiguredModelRuntime,
    chunks: list[EvidenceChunk],
    *,
    case_verdict: int,
    workers: int,
) -> tuple[dict, dict[str, int]]:
    mapped: list[dict | None] = [None] * len(chunks)
    if workers <= 1:
        for index, chunk in enumerate(chunks):
            mapped[index] = runtime.call_json(
                "map_all_artifact_fields",
                map_prompt(chunk, len(chunks), case_verdict),
                max_tokens=2048,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    runtime.call_json,
                    "map_all_artifact_fields",
                    map_prompt(chunk, len(chunks), case_verdict),
                    max_tokens=2048,
                ): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                mapped[futures[future]] = future.result()
    summaries = [item for item in mapped if item is not None]
    if len(summaries) != len(chunks):
        raise ModelAnalysisError("Not every evidence chunk received a model analysis")

    reduce_calls = 0
    round_index = 0
    while len(summaries) > 1:
        round_index += 1
        reduced: list[dict] = []
        for group in _batch(summaries, 10):
            reduced.append(
                runtime.call_json(
                    "reduce_all_child_analyses",
                    reduce_prompt(chunks[0].case_id, group, round_index),
                    max_tokens=2048,
                )
            )
            reduce_calls += 1
        summaries = reduced
    return summaries[0], {"map_calls_or_hits": len(chunks), "reduce_calls_or_hits": reduce_calls}


def _case_verdict(case_root: Path) -> tuple[int, dict[str, Any]]:
    data = json.loads((case_root / "consolidated_final.json").read_text(encoding="utf-8-sig"))
    bill = data.get("bill_summary") or {}
    verdict = int(bill.get("case_verdict") or 0)
    aggregation = []
    for index, charge in enumerate(data.get("charges") or []):
        match = charge.get("aggregation_match")
        if isinstance(match, dict) and match.get("within_tolerance") is False:
            aggregation.append({"charge_index": index, **match})
    facts = {
        "case_verdict": verdict,
        "amounts_match": bill.get("amounts_match"),
        "overall_confidence": bill.get("overall_confidence", data.get("overall_confidence")),
        "extracted_amount": bill.get("extracted_amount"),
        "calculated_amount": bill.get("calculated_amount"),
        "extracted_after_discount": bill.get("extracted_after_discount"),
        "calculated_after_discount": bill.get("calculated_after_discount"),
        "amount_diff": bill.get("amount_diff"),
        "net_corrected": bill.get("net_corrected"),
        "printed_net_payable": bill.get("printed_net_payable"),
        "total_line_items": bill.get("total_line_items"),
        "root_flagged_for_review": bool(data.get("flagged_for_review")),
        "root_flag_reason": redact_text(str(data.get("flag_reason") or "")),
        "review_total_flagged": (data.get("review_summary") or {}).get("total_flagged"),
        "aggregation_mismatches": aggregation,
    }
    return verdict, facts


def _summary_text(summary: dict) -> str:
    return json.dumps(summary, ensure_ascii=False, sort_keys=True)[:12000]


def nearest_passed_cases(
    case_ids: list[str],
    verdicts: dict[str, int],
    summaries: dict[str, dict],
    *,
    runtime: ConfiguredModelRuntime,
    cache_path: Path,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    cache = EmbeddingCache(cache_path, runtime.embedder.config)
    vectors = {
        case_id: cache.get_or_embed(_summary_text(summaries[case_id]), runtime.embedder)
        for case_id in case_ids
    }
    result: dict[str, list[dict]] = {}
    passed = [case_id for case_id in case_ids if verdicts[case_id] == 1]
    for case_id in case_ids:
        if verdicts[case_id] != 0:
            result[case_id] = []
            continue
        ranked = sorted(
            (
                {
                    "AL-number": passed_id,
                    "cosine_similarity": round(cosine_similarity(vectors[case_id], vectors[passed_id]), 4),
                    "passed_case_model_summary": summaries[passed_id],
                }
                for passed_id in passed
            ),
            key=lambda item: item["cosine_similarity"],
            reverse=True,
        )
        result[case_id] = ranked[:3]
    cache.save()
    return result, {
        "model_id": runtime.embedding_model_id,
        "region": runtime.region,
        "requests_this_run": runtime.embedder.request_count,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "local_model": False,
    }


def final_row_prompt(
    case_id: str,
    verdict: int,
    gate_facts: dict,
    model_summary: dict,
    nearest_passed: list[dict],
    cohort: dict,
) -> str:
    schema = {header: "text" for header in ROW_HEADERS}
    schema["case_verdict"] = verdict
    schema["AL-number"] = case_id
    return f"""You are the final report stage of a healthcare claim extraction audit agent.
Produce exactly one Excel row from the model-generated evidence summary and deterministic stored facts.
All nested JSON leaves and every raw/page artifact were already processed by prior model map stages.

Rules:
- AL-number must be {case_id}; case_verdict must be {verdict}.
- For verdict 0, clearly separate: direct gate, mechanism, pipeline step, underlying actual reason, practical improvement.
- For verdict 1, state that no rejection gate fired and explain the positive-control comparison.
- Use numeric evidence and artifact references when useful.
- Treat nearest passed cases as comparison controls, not ground truth overrides.
- Recommendations must be conceptual operational improvements, not code edits.
- Never output patient, hospital or doctor names.
- Return only valid JSON with exactly these keys: {json.dumps(schema)}

COHORT_COUNTS:
{json.dumps(cohort)}

DETERMINISTIC_STORED_GATE_FACTS:
{json.dumps(gate_facts, ensure_ascii=False)}

MODEL_SUMMARY_DERIVED_FROM_ALL CASE ARTIFACT CHUNKS:
{json.dumps(model_summary, ensure_ascii=False)}

NEAREST_PASSED_CASES_FROM_CONFIGURED_TITAN_EMBEDDINGS:
{json.dumps(nearest_passed, ensure_ascii=False)}
"""


def validate_row(row: dict, *, case_id: str, verdict: int) -> dict:
    if set(row) != set(ROW_HEADERS):
        raise ModelAnalysisError(
            f"Final model row for {case_id} did not contain exactly the seven required columns"
        )
    row["AL-number"] = case_id
    row["case_verdict"] = verdict
    for header in ROW_HEADERS[2:]:
        value = redact_text(str(row.get(header) or "").strip())
        if not value:
            raise ModelAnalysisError(f"Final model row for {case_id} left {header!r} empty")
        row[header] = value
    return {header: row[header] for header in ROW_HEADERS}


def generate_model_rows(
    cases: list[tuple[str, Path, list[Path]]],
    *,
    project_root: Path,
    output_dir: Path,
    chunk_chars: int,
    workers: int,
) -> tuple[list[dict], dict[str, Any]]:
    cache_dir = output_dir / "model_cache"
    runtime = ConfiguredModelRuntime(project_root, cache_dir)
    builder = CaseCorpusBuilder(chunk_chars=chunk_chars)
    summaries: dict[str, dict] = {}
    verdicts: dict[str, int] = {}
    gate_facts: dict[str, dict] = {}
    corpus_audits: dict[str, dict] = {}
    call_audits: dict[str, dict] = {}

    for case_id, case_root, extra_files in cases:
        verdict, facts = _case_verdict(case_root)
        chunks, corpus_audit = builder.build(case_id, case_root, extra_files=extra_files)
        if not chunks:
            raise ModelAnalysisError(f"Case {case_id} produced no evidence chunks")
        summary, calls = analyze_case_chunks(
            runtime,
            chunks,
            case_verdict=verdict,
            workers=workers,
        )
        verdicts[case_id] = verdict
        gate_facts[case_id] = facts
        summaries[case_id] = summary
        corpus_audits[case_id] = corpus_audit.to_dict()
        call_audits[case_id] = calls

    case_ids = [case_id for case_id, _root, _extras in cases]
    nearest, embedding_audit = nearest_passed_cases(
        case_ids,
        verdicts,
        summaries,
        runtime=runtime,
        cache_path=cache_dir / "configured_titan_embeddings.json",
    )
    cohort = {
        "case_count": len(case_ids),
        "verdict_0_count": sum(1 for value in verdicts.values() if value == 0),
        "verdict_1_count": sum(1 for value in verdicts.values() if value == 1),
    }
    rows: list[dict] = []
    for case_id in case_ids:
        row = runtime.call_json(
            "final_excel_row_from_all_artifacts",
            final_row_prompt(
                case_id,
                verdicts[case_id],
                gate_facts[case_id],
                summaries[case_id],
                nearest[case_id],
                cohort,
            ),
        )
        rows.append(validate_row(row, case_id=case_id, verdict=verdicts[case_id]))

    runtime.save_cache()
    audit = {
        "model_runtime": runtime.safe_metadata(),
        "embedding_runtime": embedding_audit,
        "corpus_coverage": corpus_audits,
        "model_call_coverage": call_audits,
        "cohort": cohort,
        "all_json_fields_and_raw_artifacts_model_processed": all(
            value["all_artifacts_represented"] for value in corpus_audits.values()
        ),
        "row_source": "configured Bedrock model output only",
    }
    return rows, audit
