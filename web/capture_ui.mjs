/**
 * Drive the GridSight UI end to end and capture the evidence screenshots.
 *
 * Runs the real user path against the real services: upload an in-registry
 * defect frame, upload a frame from the withheld asset class, cold-start a
 * specialist through the inline uploader, and read the trends dashboard.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const WEB = process.env.WEB_URL ?? "http://localhost:3000";
const OUT = path.resolve(process.argv[2] ?? "../artifacts/screenshots");
const DEMO = path.resolve("public/demo");

const shot = async (page, name) => {
  const file = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: file, fullPage: true });
  console.log(`  saved ${file}`);
};

const waitForResult = async (page, timeout = 180_000) => {
  await page.waitForFunction(() => /Agent narrative|no model in the registry covers/i.test(document.body.innerText), {
    timeout,
  });
  await page.waitForTimeout(1200);
};

async function run(theme) {
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 2,
    colorScheme: theme,
  });
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`    [browser error] ${m.text()}`);
  });

  const suffix = theme === "dark" ? "" : "-light";

  // ---- 1. in-registry defect frame ---------------------------------------
  console.log(`[${theme}] inspect: in-registry insulator defect frame`);
  await page.goto(WEB, { waitUntil: "networkidle" });
  await page.setInputFiles('input[type="file"]', path.join(DEMO, "insulator_defect.png"));
  await page.waitForTimeout(2500);
  await shot(page, `01-agent-steps${suffix}`);
  await waitForResult(page);
  await shot(page, `02-defect-overlay${suffix}`);

  if (theme === "dark") {
    // ---- 2. withheld class -> refusal + cold-start ------------------------
    console.log(`[${theme}] inspect: withheld rail frame (expect refusal)`);
    await page.goto(WEB, { waitUntil: "networkidle" });
    await page.setInputFiles('input[type="file"]', path.join(DEMO, "rail_defect.png"));
    await waitForResult(page);
    await shot(page, "03-unroutable-refusal");

    console.log("[dark] cold start: supplying 8 reference images");
    await page.fill('input[type="text"]', "rail_surface");
    const refs = Array.from({ length: 8 }, (_, i) =>
      path.join(DEMO, "rail_refs", `${String(i).padStart(5, "0")}.png`),
    );
    const inputs = page.locator('input[type="file"]');
    await inputs.nth(1).setInputFiles(refs);
    await page.waitForTimeout(600);
    await shot(page, "04-coldstart-ready");

    await page.getByRole("button", { name: /cold-start a specialist/i }).click();
    await page.waitForFunction(
      () => /New model minted/i.test(document.body.innerText),
      { timeout: 300_000 },
    );
    await page.waitForTimeout(1500);
    await shot(page, "05-coldstart-complete");

    // ---- 3. same frame again: must route to the new model -----------------
    console.log("[dark] re-inspect the same frame: expect routing, not cold start");
    await page.goto(WEB, { waitUntil: "networkidle" });
    await page.setInputFiles('input[type="file"]', path.join(DEMO, "rail_defect.png"));
    await waitForResult(page);
    await shot(page, "06-routes-to-new-model");

    // ---- 4. registry -------------------------------------------------------
    await page.goto(`${WEB}/registry`, { waitUntil: "networkidle" });
    await page.waitForTimeout(2500);
    await shot(page, "07-registry");
  }

  // ---- 5. trends ----------------------------------------------------------
  console.log(`[${theme}] trends dashboard`);
  await page.goto(`${WEB}/trends`, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  await shot(page, `08-trends${suffix}`);

  // ---- 6. responsive check ------------------------------------------------
  if (theme === "dark") {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: "networkidle" });
    await page.waitForTimeout(2000);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
    );
    console.log(`  mobile horizontal overflow: ${overflow}`);
    await shot(page, "09-trends-mobile");
  }

  await browser.close();
}

await mkdir(OUT, { recursive: true });
await run("dark");
await run("light");
console.log("done");
