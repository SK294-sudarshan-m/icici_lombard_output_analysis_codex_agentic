import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [rowsPath, auditPath, outputPath] = process.argv.slice(2);
if (!rowsPath || !auditPath || !outputPath) {
  throw new Error("Usage: build_report.mjs <rows.json> <audit.json> <output.xlsx>");
}

const rows = JSON.parse(await fs.readFile(rowsPath, "utf8"));
const audit = JSON.parse(await fs.readFile(auditPath, "utf8"));
const headers = [
  "AL-number",
  "case_verdict",
  "what caused case verdict 0",
  "how it happened",
  "where it happened(step in pipeline)",
  "why it happened(actual reasons)",
  "what can be improved",
];

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Case Verdict Analysis");
sheet.showGridLines = false;
sheet.getRange(`A1:G${rows.length + 1}`).values = [
  headers,
  ...rows.map((row) => headers.map((header) => row[header] ?? "")),
];

const header = sheet.getRange("A1:G1");
header.format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 11 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "outside", style: "medium", color: "#17365D" },
};
header.format.rowHeight = 34;

const body = sheet.getRange(`A2:G${rows.length + 1}`);
body.format = {
  font: { color: "#1F2937", size: 10 },
  verticalAlignment: "top",
  wrapText: true,
  borders: {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#D9E2F3" },
  },
};
body.format.rowHeight = 108;
sheet.getRange(`A2:B${rows.length + 1}`).format.horizontalAlignment = "center";

const widths = [20, 13, 48, 50, 40, 58, 58];
for (let col = 0; col < widths.length; col += 1) {
  sheet.getRangeByIndexes(0, col, rows.length + 1, 1).format.columnWidth = widths[col];
}

const verdictRange = sheet.getRange(`B2:B${rows.length + 1}`);
verdictRange.format.font = { bold: true, color: "#FFFFFF" };
verdictRange.conditionalFormats.add("cellIs", {
  operator: "equal",
  formula: 0,
  format: { fill: "#C00000", font: { bold: true, color: "#FFFFFF" } },
});
verdictRange.conditionalFormats.add("cellIs", {
  operator: "equal",
  formula: 1,
  format: { fill: "#548235", font: { bold: true, color: "#FFFFFF" } },
});

const table = sheet.tables.add(`A1:G${rows.length + 1}`, true, "CaseVerdictAnalysisTable");
table.style = "TableStyleMedium2";
table.showFilterButton = true;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);

const baseline = audit.baseline_comparison || {};
const embedding = audit.embedding_runtime || {};
const noteCell = sheet.getRange(`G${rows.length + 1}`);
await workbook.comments.setSelf({ displayName: "User" });
workbook.comments.addThread(
  { cell: noteCell },
  `Audit: ${baseline.case_count ?? rows.length} cases; ${baseline.fail_count ?? 0} rejected; ` +
    `${baseline.pass_count ?? 0} passed. Cloud semantic categorization: ${embedding.model_id ?? "not recorded"} ` +
    `in ${embedding.region ?? "unknown region"}; local fallback=${embedding.local_fallback ?? "unknown"}. ` +
    `All nested output files were scanned; report text excludes patient, hospital and doctor names.`,
);

const inspect = await workbook.inspect({
  kind: "table",
  range: `Case Verdict Analysis!A1:G${rows.length + 1}`,
  include: "values,formulas",
  tableMaxRows: rows.length + 1,
  tableMaxCols: 7,
  maxChars: 16000,
});
console.log(inspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const preview = await workbook.render({
  sheetName: "Case Verdict Analysis",
  range: `A1:G${rows.length + 1}`,
  scale: 1,
  format: "png",
});
await fs.writeFile(
  outputPath.replace(/\.xlsx$/i, "_preview.png"),
  new Uint8Array(await preview.arrayBuffer()),
);
