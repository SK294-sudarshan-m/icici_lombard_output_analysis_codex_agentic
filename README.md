# Full-Artifact Case Verdict Analysis Agent

This standalone Python agent creates a new model-generated Excel report without
using Codex-authored explanations. The earlier analysis remains unchanged in
`output_codex/`; the new agent writes only to `output_agent/`.

## Inputs processed by the agent

The default corpus is `jsons/`. For every discovered AL case the agent reads every
nested file, including the entire `_work` tree.

- Every JSON object/list is recursively flattened to one record per nested leaf
  path. Empty objects/lists are retained too, so every nested field is represented.
- Every line of every raw TXT page is included with its line number.
- Every page-specific extraction, OCR, classification, level-1/level-2 grouping,
  merged-bill, consolidation, confidence, contract and adjudication artifact is
  included according to its path in the corpus.
- Every PDF page text layer is included. Image-only pages are explicitly recorded;
  their corresponding page OCR/extraction artifacts in `jsons/` remain full inputs.
- Other binary files are represented by length and SHA-256.

Patient, hospital and doctor identity values are redacted before model calls. Numeric
IDs, amounts, charges, medicines, prescriptions, returns and pipeline evidence are
retained.

Ground-truth PDFs in `input_pdfs/` are attached to matching output cases only.
If a PDF filename contains an AL number, that same AL must exist under `jsons/`.
Neutral staged names such as `ground_truth_001.pdf` are mapped only when there is
exactly one non-AL-named output case. PDFs with no matching output case are skipped
by default and listed in the audit; use `--strict-pdf-mapping` to fail instead.

## Model-only analysis flow

The agent uses only models declared by the parent project's `config.py`:

1. `settings.model_id` (configured Qwen Bedrock model) analyzes every evidence chunk.
2. Hierarchical Qwen reduce calls combine every chunk analysis without dropping
   conflicting evidence.
3. `settings.med_embedding_model` (configured Titan Bedrock embedding model) compares
   rejected cases with the closest verdict-1 controls.
4. Qwen produces one strict seven-column JSON row per AL case.
5. The Python process passes those model-generated rows to `build_report.mjs`, which
   writes the Excel workbook with `@oai/artifact-tool`.

There is no local language model, local embedder, Codex fallback, or hard-coded
narrative fallback. AWS credentials come only from boto3's normal project role,
profile or environment chain; the agent accepts and logs no access/secret keys.

The audit records file/field/line/chunk coverage, configured model IDs, model/cache
call counts, and `all_json_fields_and_raw_artifacts_model_processed=true`. A failed
model call exits with code 2 before a new workbook is produced.

## Run

Use the project's Python environment:

```powershell
python case_verdict_analysis_agent\agent.py
```

The default output is:

```text
case_verdict_analysis_agent/output_agent/case_verdict_analysis_agent.xlsx
```

Useful checks:

```powershell
python case_verdict_analysis_agent\agent.py --check-model-config
python case_verdict_analysis_agent\agent.py --check-model-access
```

Optional capacity controls (they do not change model IDs):

```powershell
python case_verdict_analysis_agent\agent.py --chunk-chars 60000 --model-workers 4
```

`output_codex/` is not read or used as evidence by the new workflow.
