# Full-Artifact Case Verdict Analysis Agent

This standalone Python agent creates a new Excel report without using Codex-authored
explanations or Codex runtime dependencies. The earlier analysis remains unchanged in
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

Ground-truth PDFs in `input_pdfs/` are always accepted. If a PDF filename maps to
an output AL case, it is attached to that case. If it does not map, it is scanned
as standalone PDF context and listed in the audit. It does not need a matching
output folder unless you intentionally run with `--strict-pdf-mapping`.

## Standalone analysis modes

Default mode is `rules`, which uses no external AI service and no local AI model:

- Python scans every nested output artifact and accepted PDF.
- A deterministic rule engine categorizes the actual `case_verdict=0` causes
  from consolidated gate facts plus full-corpus evidence patterns.
- Python/openpyxl writes the Excel workbook.

Optional mode is `bedrock`, which uses standalone copied settings in
`standalone_settings.py`. These values match the parent project's model defaults,
but the agent does not import the parent project's `config.py`, `.env`, settings
objects, or `pipeline.clients`.

- `LANGUAGE_MODEL_ID = qwen.qwen3-vl-235b-a22b`
- `EMBEDDING_MODEL_ID = amazon.titan-embed-text-v2:0`
- `AWS_REGION = ap-south-1`

There is no Codex dependency, OpenAI dependency, Node dependency, or local AI
model dependency in either mode. AWS credentials are needed only if you explicitly
run `--analysis-mode bedrock`.

The audit records file/field/line/chunk coverage, configured model IDs, model/cache
coverage and `all_json_fields_and_raw_artifacts_processed=true`. In Bedrock mode,
a failed model call exits with code 2 before a new workbook is produced.

## Run

Use a standalone Python environment:

```powershell
cd C:\Users\User\Desktop\icici-lombard-claims-extraction-uat\case_verdict_analysis_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python agent.py
```

To explicitly use Bedrock/Qwen/Titan instead of the no-AI rule engine:

```powershell
python agent.py --analysis-mode bedrock
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
