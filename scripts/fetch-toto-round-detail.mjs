import fs from "fs";
import path from "path";

const holdCntId = process.argv[2] ?? "1637";

const url = `https://store.toto-dream.com/dcs/subos/screen/pi04/spin011/PGSPIN01101LnkHoldCntLotResultLsttoto.form?popupDispDiv=disp&holdCntId=${holdCntId}`;

const outputDir = path.join(process.cwd(), "data/snapshots");
const outputPath = path.join(outputDir, `totoRound-${holdCntId}.html`);

fs.mkdirSync(outputDir, { recursive: true });

console.log(`Fetching ${url}`);

const res = await fetch(url, {
  headers: {
    "User-Agent": "Mozilla/5.0",
  },
});

if (!res.ok) {
  console.error(`Failed: ${res.status}`);
  process.exit(1);
}

const html = await res.text();

fs.writeFileSync(outputPath, html, "utf8");

console.log(`Saved ${outputPath}`);
console.log(`HTML length: ${html.length}`);