// Experimental helper: fetches the public toto vote page HTML and prints text snippets.
// Use this first to inspect a round, then copy vote rates into CSV until parser is hardened.
// Example: npm run fetch:toto-votes -- 1636

const round = process.argv[2];
if (!round) {
  console.error("Usage: npm run fetch:toto-votes -- <round_no_number>");
  process.exit(1);
}

const url = `https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501ForwardVotetotoSP.form?commodityId=01&fromId=SSIN026&gameAssortment=A&holdCntId=${round}`;
const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 TOTO EDGE research" } });
if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
const html = await res.text();
const text = html
  .replace(/<script[\s\S]*?<\/script>/g, " ")
  .replace(/<style[\s\S]*?<\/style>/g, " ")
  .replace(/<[^>]+>/g, " ")
  .replace(/&nbsp;/g, " ")
  .replace(/\s+/g, " ")
  .trim();
console.log(text.slice(0, 4000));
console.log("\nSource:", url);
