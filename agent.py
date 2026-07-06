"""Run the full-artifact, standalone-model case-verdict analysis agent.

The primary workflow is entirely agent-generated: every nested JSON leaf and every
raw/page artifact is fed through the standalone copied Bedrock model settings, and
the resulting seven-column workbook is authored by this program.  The preserved
``output_codex`` folder is never read, changed, or used as model evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .model_corpus import (
        AL_PATTERN,
        ROW_HEADERS,
        CloudEmbeddingError,
        ConfiguredModelRuntime,
        ModelAnalysisError,
        generate_model_rows,
    )
except ImportError:  # Direct script execution.
    from model_corpus import (
        AL_PATTERN,
        ROW_HEADERS,
        CloudEmbeddingError,
        ConfiguredModelRuntime,
        ModelAnalysisError,
        generate_model_rows,
    )


def discover_case_roots(jsons_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for path in jsons_dir.rglob("consolidated_final.json"):
        if "_work" in {part.lower() for part in path.parts}:
            continue
        roots.append(path.parent)
    return sorted(set(roots), key=lambda value: str(value).lower())


def infer_case_id(case_root: Path) -> str:
    for part in reversed(case_root.parts):
        if AL_PATTERN.fullmatch(part):
            return part
    candidates: Counter[str] = Counter()
    for path in sorted(item for item in case_root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in {".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        candidates.update(AL_PATTERN.findall(text))
    if candidates:
        return candidates.most_common(1)[0][0]
    raise ModelAnalysisError(
        "Could not infer an AL number from an output case; add the AL number to the folder or JSON artifacts"
    )


def _normalise_stem(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(value).stem.lower())


def assign_pdf_inputs(
    case_pairs: list[tuple[str, Path]],
    pdf_dir: Path,
) -> tuple[dict[str, list[Path]], list[Path]]:
    """Pair ground-truth PDFs without semantic/local filename models."""
    assignments = {case_id: [] for case_id, _root in case_pairs}
    pdfs = sorted(pdf_dir.rglob("*.pdf")) if pdf_dir.exists() else []
    unassigned: list[Path] = []
    for pdf in pdfs:
        match = AL_PATTERN.search(pdf.name)
        if match and match.group(0) in assignments:
            assignments[match.group(0)].append(pdf)
            continue
        pdf_stem = _normalise_stem(pdf.name)
        matched = False
        for case_id, case_root in case_pairs:
            source_stems = {_normalise_stem(case_root.name)}
            consolidated = json.loads(
                (case_root / "consolidated_final.json").read_text(encoding="utf-8-sig")
            )
            source_stems.add(_normalise_stem(str(consolidated.get("case_number") or "")))
            if pdf_stem and any(
                stem and min(len(stem), len(pdf_stem)) >= 10 and (
                    stem == pdf_stem or stem in pdf_stem or pdf_stem in stem
                )
                for stem in source_stems
            ):
                assignments[case_id].append(pdf)
                matched = True
                break
        if not matched:
            unassigned.append(pdf)

    # The supplied image-only PDF has a neutral staged filename. If there is exactly
    # one neutral PDF and one non-AL-named output case, pair them deterministically.
    # Do this even when other AL-named PDFs are unmatched, because those PDFs simply
    # belong to output cases that are not present in the current jsons corpus.
    non_al_cases = [
        (case_id, root)
        for case_id, root in case_pairs
        if not AL_PATTERN.fullmatch(root.name)
        and not assignments[case_id]
    ]
    neutral_unassigned = [path for path in unassigned if not AL_PATTERN.search(path.name)]
    if len(neutral_unassigned) == len(non_al_cases) == 1:
        neutral_pdf = neutral_unassigned[0]
        assignments[non_al_cases[0][0]].append(neutral_pdf)
        unassigned = [path for path in unassigned if path != neutral_pdf]

    return assignments, unassigned


def input_manifest(jsons_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in jsons_dir.rglob("*") if path.is_file())
    extensions: Counter[str] = Counter()
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        extensions[path.suffix.lower() or "<none>"] += 1
        total_bytes += len(data)
        digest.update(hashlib.sha256(data).digest())
    return {
        "files": len(files),
        "bytes": total_bytes,
        "extensions": dict(extensions),
        "aggregate_content_sha256": digest.hexdigest(),
    }


def build_xlsx(
    rows_json: Path,
    audit_json: Path,
    output_xlsx: Path,
    *,
    node: str | None = None,
    artifact_node_modules: str | None = None,
) -> Path:
    """Build the Excel report with Python only.

    ``node`` and ``artifact_node_modules`` are accepted for backward-compatible
    CLI parsing but intentionally ignored.  The standalone agent no longer needs
    Codex's artifact-tool or any parent-project code to create the workbook.
    """
    _ = (node, artifact_node_modules)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl is required for standalone XLSX output. "
            "Install case_verdict_analysis_agent/requirements.txt and retry."
        ) from exc

    rows = json.loads(rows_json.read_text(encoding="utf-8"))
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Case Verdict Analysis"
    worksheet.append(ROW_HEADERS)
    for row in rows:
        worksheet.append([row.get(header, "") for header in ROW_HEADERS])

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    wrap_top = Alignment(wrap_text=True, vertical="top")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row_cells in worksheet.iter_rows(min_row=2):
        for cell in row_cells:
            cell.alignment = wrap_top

    widths = [20, 14, 42, 48, 38, 54, 54]
    for index, width in enumerate(widths, start=1):
        worksheet.column_dimensions[get_column_letter(index)].width = width
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    meta = workbook.create_sheet("Run Audit")
    meta.append(["Field", "Value"])
    meta["A1"].font = header_font
    meta["B1"].font = header_font
    meta["A1"].fill = header_fill
    meta["B1"].fill = header_fill
    model_runtime = audit.get("model_analysis", {}).get("model_runtime", {})
    embedding_runtime = audit.get("model_analysis", {}).get("embedding_runtime", {})
    audit_rows = [
        ("generated_at", audit.get("generated_at")),
        ("input_folder", audit.get("input_folder")),
        ("case_count", audit.get("case_count")),
        ("pdf_input_count", audit.get("pdf_input_count")),
        ("unmapped_pdf_input_count", audit.get("unmapped_pdf_input_count")),
        ("language_model_id", model_runtime.get("language_model_id")),
        ("embedding_model_id", embedding_runtime.get("model_id") or model_runtime.get("embedding_model_id")),
        ("region", model_runtime.get("region")),
        ("settings_source", model_runtime.get("settings_source")),
        ("imports_parent_project_config", model_runtime.get("imports_parent_project_config")),
        ("imports_parent_project_pipeline_clients", model_runtime.get("imports_parent_project_pipeline_clients")),
        (
            "all_json_fields_and_raw_artifacts_model_processed",
            audit.get("model_analysis", {}).get("all_json_fields_and_raw_artifacts_model_processed"),
        ),
        ("output_codex_used_as_input", audit.get("output_codex_used_as_input")),
    ]
    for key, value in audit_rows:
        meta.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    meta.column_dimensions["A"].width = 48
    meta.column_dimensions["B"].width = 92
    for row_cells in meta.iter_rows():
        for cell in row_cells:
            cell.alignment = wrap_top

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.unlink(missing_ok=True)
    workbook.save(output_xlsx)
    if not output_xlsx.exists() or output_xlsx.stat().st_size < 1000:
        raise RuntimeError("Workbook was not produced as a valid XLSX")
    return output_xlsx


def validate_destinations(jsons_dir: Path, output_dir: Path, output_codex: Path) -> None:
    jsons = jsons_dir.resolve()
    output = output_dir.resolve()
    codex = output_codex.resolve()
    if output == codex or codex in output.parents or output in codex.parents:
        raise ModelAnalysisError("output_agent and output_codex must remain separate")
    if output == jsons or jsons in output.parents or output in jsons.parents:
        raise ModelAnalysisError("output_agent must not overlap the JSON input corpus")


def run(args: argparse.Namespace) -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    jsons_dir = args.jsons.resolve()
    output_dir = args.output_dir.resolve()
    output_codex = here / "output_codex"
    validate_destinations(jsons_dir, output_dir, output_codex)
    if not jsons_dir.exists():
        raise ModelAnalysisError(f"JSON/artifact input folder does not exist: {jsons_dir}")

    roots = discover_case_roots(jsons_dir)
    if not roots:
        raise ModelAnalysisError(f"No case-level consolidated_final.json files found under {jsons_dir}")
    case_pairs = [(infer_case_id(root), root) for root in roots]
    if len({case_id for case_id, _root in case_pairs}) != len(case_pairs):
        raise ModelAnalysisError("Duplicate AL numbers were inferred from the input corpus")
    pdf_assignments, unmapped_pdfs = assign_pdf_inputs(case_pairs, args.pdf_input_dir.resolve())
    if unmapped_pdfs and args.strict_pdf_mapping:
        names = ", ".join(path.name for path in unmapped_pdfs)
        raise ModelAnalysisError(
            f"Ground-truth PDF(s) could not be mapped to an AL case: {names}"
        )
    if unmapped_pdfs:
        names = ", ".join(path.name for path in unmapped_pdfs)
        print(
            "WARNING: Skipping ground-truth PDF(s) with no matching output case in jsons: "
            f"{names}",
            file=sys.stderr,
        )
    cases = [
        (case_id, root, pdf_assignments[case_id])
        for case_id, root in sorted(case_pairs)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows, model_audit = generate_model_rows(
        cases,
        output_dir=output_dir,
        chunk_chars=args.chunk_chars,
        workers=args.model_workers,
    )
    if len(rows) != len(cases):
        raise ModelAnalysisError("Configured model did not generate exactly one row per case")
    if any(list(row) != ROW_HEADERS for row in rows):
        raise ModelAnalysisError("Generated row column order does not match the requested Excel contract")

    rows_path = output_dir / "case_verdict_analysis_agent_rows.json"
    audit_path = output_dir / "case_verdict_analysis_agent_audit.json"
    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_folder": str(jsons_dir),
        "input_manifest": input_manifest(jsons_dir),
        "case_count": len(cases),
        "pdf_input_count": sum(len(paths) for paths in pdf_assignments.values()),
        "unmapped_pdf_input_count": len(unmapped_pdfs),
        "unmapped_pdf_inputs": [str(path) for path in unmapped_pdfs],
        "output_codex_preserved": output_codex.exists(),
        "output_codex_used_as_input": False,
        "excel_rows_generated_by": "standalone copied Bedrock model settings via agent code",
        "model_analysis": model_audit,
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
        "input_files": audit["input_manifest"]["files"],
        "all_artifacts_model_processed": model_audit["all_json_fields_and_raw_artifacts_model_processed"],
        "output_codex_preserved": True,
        "xlsx": str(xlsx_path) if not args.skip_xlsx else None,
        "audit": str(audit_path),
    }


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsons", type=Path, default=here / "jsons")
    parser.add_argument("--pdf-input-dir", type=Path, default=here / "input_pdfs")
    parser.add_argument("--output-dir", type=Path, default=here / "output_agent")
    parser.add_argument("--xlsx-name", default="case_verdict_analysis_agent.xlsx")
    parser.add_argument("--chunk-chars", type=int, default=60000)
    parser.add_argument("--model-workers", type=int, default=4)
    parser.add_argument("--node", help=argparse.SUPPRESS)
    parser.add_argument("--artifact-node-modules", help=argparse.SUPPRESS)
    parser.add_argument("--skip-xlsx", action="store_true")
    parser.add_argument("--check-model-config", action="store_true")
    parser.add_argument("--check-model-access", action="store_true")
    parser.add_argument(
        "--strict-pdf-mapping",
        action="store_true",
        help=(
            "Fail if any PDF under --pdf-input-dir does not map to a discovered AL/output case. "
            "By default, unmatched PDFs are recorded in the audit and skipped."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check_model_config or args.check_model_access:
            runtime = ConfiguredModelRuntime(args.output_dir.resolve() / "model_cache")
            result = runtime.safe_metadata()
            if args.check_model_access:
                vector = runtime.embedder.embed("case verdict artifact analysis access check")
                response = runtime.call_json(
                    "access_check",
                    'Return only this JSON: {"status":"ok"}',
                    max_tokens=32,
                )
                result.update({
                    "embedding_dimension": len(vector),
                    "language_model_response": response,
                })
            print(json.dumps(result, indent=2))
            return 0
        result = run(args)
    except (ModelAnalysisError, CloudEmbeddingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
