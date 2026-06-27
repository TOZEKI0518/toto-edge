import fs from "fs";
import path from "path";

const inputPath = path.join(process.cwd(), "data/imports/current-toto-matches.csv");
const outputPath = path.join(process.cwd(), "data/generated/currentMatches.ts");

const parseCsvLine = (line) => {
  const result = [];
  let current = "";
  let insideQuotes = false;

  for (const char of line) {
    if (char === '"') {
      insideQuotes = !insideQuotes;
    } else if (char === "," && !insideQuotes) {
      result.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current);
  return result.map((v) => v.trim());
};

const text = fs.readFileSync(inputPath, "utf8").trim();
const [headerLine, ...lines] = text.split(/\r?\n/);
const headers = parseCsvLine(headerLine);

const rows = lines.map((line) => {
  const values = parseCsvLine(line);
  return Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""]));
});

fs.mkdirSync(path.dirname(outputPath), { recursive: true });

fs.writeFileSync(
  outputPath,
  `import { normalizeMatches } from "@/lib/normalizers/normalizeMatch";
import type { RawTotoMatch } from "@/types/rawToto";

const rawMatches: RawTotoMatch[] = ${JSON.stringify(rows, null, 2)};

export const currentMatches = normalizeMatches(rawMatches);
`,
  "utf8"
);

console.log(`Generated ${rows.length} normalized matches.`);