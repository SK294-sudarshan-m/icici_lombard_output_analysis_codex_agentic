# Case Verdict Analysis Agent

This isolated agent explains why `bill_summary.case_verdict` is `0` or `1` without
modifying the claim extraction pipeline.

It recursively reads every file under every discovered output case, including root
artifacts and `_work` JSON/TXT/PDF files. It also scans the project pipeline and
agent source files for verdict and calculation context, compares rejected cases with
passed cases, reads downstream `adjudication.json` as corroborating context, and
uses the PDFs under `input_pdfs/` as ground truth.

## Model policy

Semantic categorization uses only `settings.med_embedding_model` and
`settings.aws_region` from the parent project's `config.py`. In the current project
that is Amazon Titan Embed Text v2 through AWS Bedrock. The analyzer reuses the
project's `pipeline.clients.create_bedrock_client` and boto3 credential provider
chain. It does not accept access keys, secret keys, session tokens, model-ID
overrides, or API-key arguments.

There is deliberately no local embedding fallback: no sentence-transformers,
Transformers, Ollama, FAISS embedder, hashing vectorizer, or substitute model. If
the configured Bedrock model cannot be called, the analysis exits with code 2 before
rewriting the report. The optional cache contains only vectors previously returned
by that same configured model and is invalidated when the model ID or region changes.
Because the run is fail-closed, a previously generated workbook may remain untouched
after an authentication failure; a successfully regenerated audit always records the
cloud model ID, region, request/cache counts, and `local_fallback: false`.

Privacy behavior: the scan retains only structural evidence, numerical values,
identifiers, charge/medicine/prescription/return signals, stage status, and file
hashes. Patient, hospital and doctor names are never written to the report or audit.

Run from the project root:

```powershell
python case_verdict_analysis_agent\agent.py
```

The command must run in the project's Python environment, which provides boto3,
pydantic-settings, and the rest of `requirements.txt`.

Validate the configured model without making an embedding request:

```powershell
python case_verdict_analysis_agent\agent.py --check-model-config
```

Validate live Bedrock access with one safe embedding request:

```powershell
python case_verdict_analysis_agent\agent.py --check-embedding-access
```

Optional explicit runtime paths:

```powershell
python case_verdict_analysis_agent\agent.py `
  --node C:\path\to\node.exe `
  --artifact-node-modules C:\path\to\node_modules
```

The default output folder is `case_verdict_analysis_agent/output/`. It contains the
Excel report plus privacy-safe JSON rows and an audit proving file coverage. The
main worksheet has the seven requested columns in the requested order; verdict-1
cases are included as positive comparison controls.
