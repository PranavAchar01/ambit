/**
 * Re-capture the documentation screenshots against the running UI.
 *
 * Kept in the repo rather than run ad hoc because every previous set went stale
 * silently -- the powerline shots survived a full product rescope and a visual
 * rebuild before anyone noticed they showed a different product.
 *
 *   node web/capture_screenshots.mjs [baseUrl]   (run from web/, playwright lives there)
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";

const BASE = process.argv[2] ?? "http://localhost:3000";
const OUT = new URL("../docs/screenshots/", import.meta.url).pathname;
const DEMO = new URL("./public/demo/", import.meta.url).pathname;

const settle = (page, ms = 1200) => page.waitForTimeout(ms);

const shots = [
  {
    name: "20-electronics-routed",
    async run(page) {
      await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
      await page.setInputFiles('input[type="file"]', `${DEMO}pcb1_defect.png`);
      // The agent takes several seconds: wait for the verdict, not a fixed delay.
      await page.waitForSelector(".badge-defect", { timeout: 180000 });
      await settle(page);
    },
  },
  {
    name: "21-electronics-refused",
    async run(page) {
      await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
      await page.setInputFiles('input[type="file"]', `${DEMO}pcb4_unroutable.png`);
      await page.waitForSelector(".badge-unroutable, .panel-alert", { timeout: 180000 });
      await settle(page);
    },
  },
  {
    name: "22-electronics-registry",
    async run(page) {
      await page.goto(`${BASE}/registry`, { waitUntil: "networkidle" });
      await page.waitForSelector(".grid-cards article", { timeout: 30000 });
      await settle(page);
    },
  },
  {
    name: "23-electronics-trends",
    async run(page) {
      await page.goto(`${BASE}/trends`, { waitUntil: "networkidle" });
      await page.waitForSelector("svg", { timeout: 30000 });
      await settle(page);
    },
  },
];

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
  colorScheme: "dark",
});

await mkdir(OUT, { recursive: true });
for (const shot of shots) {
  try {
    await shot.run(page);
    await page.screenshot({ path: `${OUT}${shot.name}.png`, fullPage: true });
    console.log(`  captured ${shot.name}`);
  } catch (err) {
    console.error(`  FAILED   ${shot.name}: ${err.message}`);
    process.exitCode = 1;
  }
}
await browser.close();
