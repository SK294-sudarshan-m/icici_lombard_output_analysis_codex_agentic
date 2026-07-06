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

## Standalone model-only analysis flow

The agent uses standalone copied settings in `standalone_settings.py`. These
values match the parent project's model defaults, but the agent does not import
the parent project's `config.py`, `.env`, settings objects, or `pipeline.clients`.

1. `LANGUAGE_MODEL_ID = qwen.qwen3-vl-235b-a22b` analyzes every evidence chunk.
2. Hierarchical Qwen reduce calls combine every chunk analysis without dropping
   conflicting evidence.
3. `EMBEDDING_MODEL_ID = amazon.titan-embed-text-v2:0` compares
   rejected cases with the closest verdict-1 controls.
4. Qwen produces one strict seven-column JSON row per AL case.
5. Python/openpyxl writes the Excel workbook.

There is no local language model, local embedder, Codex fallback, or hard-coded
narrative fallback. AWS credentials come only from boto3's normal AWS role,
profile or environment chain; the agent accepts and logs no access/secret keys.

The audit records file/field/line/chunk coverage, configured model IDs, model/cache
call counts, and `all_json_fields_and_raw_artifacts_model_processed=true`. A failed
model call exits with code 2 before a new workbook is produced.

## Run

Use a standalone Python environment:

```powershell
cd C:\Users\User\Desktop\icici-lombard-claims-extraction-uat\case_verdict_analysis_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python agent.py
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
