import fs from "fs";
import path from "path";

const fileName = process.argv[2] ?? "rakutenSchedule.html";
const keyword = process.argv[3] ?? "第";

const filePath = path.join(process.cwd(), "data/snapshots", fileName);

if (!fs.existsSync(filePath)) {
  console.error(`File not found: ${filePath}`);
  process.exit(1);
}

const html = fs.readFileSync(filePath, "utf8");

const plainText = html
  .replace(/<script[\s\S]*?<\/script>/gi, "")
  .replace(/<style[\s\S]*?<\/style>/gi, "")
  .replace(/<[^>]+>/g, "\n")
  .replace(/&nbsp;/g, " ")
  .replace(/&amp;/g, "&")
  .replace(/\s+/g, " ")
  .trim();

console.log(`File: ${fileName}`);
console.log(`HTML length: ${html.length}`);
console.log(`Text length: ${plainText.length}`);
console.log(`Keyword: ${keyword}`);
console.log("");

let index = 0;
let count = 0;

while ((index = plainText.indexOf(keyword, index)) !== -1 && count < 20) {
  const start = Math.max(0, index - 80);
  const end = Math.min(plainText.length, index + 180);

  console.log(`--- Match ${count + 1} ---`);
  console.log(plainText.slice(start, end));
  console.log("");

  index += keyword.length;
  count += 1;
}

if (count === 0) {
  console.log("No keyword matches found.");
}