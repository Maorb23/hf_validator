import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const origin = process.env.PROTOTYPE_URL || 'http://127.0.0.1:4317';
const browser = await chromium.launch({ headless: true });
const results = [];
await mkdir(new URL('../screenshots/', import.meta.url), { recursive: true });

try {
  for (const theme of ['dark', 'light']) {
    for (const [device, viewport] of Object.entries({ desktop: { width: 1440, height: 1000 }, mobile: { width: 390, height: 844 } })) {
      const page = await browser.newPage({ viewport, deviceScaleFactor: 1, reducedMotion: 'reduce', isMobile: device === 'mobile', hasTouch: device === 'mobile' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${origin}/${theme}`, { waitUntil: 'networkidle' });
      await page.evaluate(() => document.fonts.ready);
      await page.waitForFunction(() => [...document.querySelectorAll('.integration-brand img')].every(img => img.complete && img.naturalWidth > 0));
      assert.equal(await page.getByRole('button', { name: 'Open menu' }).count(), device === 'mobile' ? 1 : 0);
      assert.equal(await page.locator('h1').innerText(), 'Better inference.\nBacked by\nevidence.');
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${theme}/${device} overflows horizontally`);
      await page.screenshot({ path: new URL(`../screenshots/${theme}-${device}.png`, import.meta.url).pathname, fullPage: true, animations: 'disabled' });
      await page.screenshot({ path: new URL(`../screenshots/${theme}-${device}-hero.png`, import.meta.url).pathname, animations: 'disabled' });

      await page.getByRole('button', { name: 'Search landscape', exact: true }).click();
      assert.equal(await page.getByRole('button', { name: 'Search landscape', exact: true }).getAttribute('aria-pressed'), 'true');
      await page.getByRole('button', { name: /^throughput$/i }).click();
      assert.equal(await page.getByRole('button', { name: 'Pareto frontier', exact: true }).getAttribute('aria-pressed'), 'true');
      await page.getByRole('button', { name: /Nebius Jobs Remote execution/ }).click();
      assert.equal(await page.locator('.node-details h3').innerText(), 'Run the workload remotely.');
      await page.getByRole('button', { name: 'Follow the workflow' }).click();
      assert.equal(await page.locator('.node-details h3').innerText(), 'Compare what was measured.');
      await page.getByRole('button', { name: 'SGLang', exact: false }).click();
      assert.ok(await page.getByText('chunked_prefill_size', { exact: false }).isVisible());
      await page.getByRole('button', { name: 'TensorRT-LLM', exact: false }).click();
      assert.ok(await page.getByText('kv_cache_config', { exact: false }).isVisible());

      if (device === 'mobile') {
        await page.getByRole('button', { name: 'Open menu' }).click();
        await page.locator('.mobile-links').getByRole('button', { name: 'Methodology' }).click();
      } else {
        await page.getByRole('button', { name: 'How we compare' }).click();
      }
      assert.ok(await page.getByRole('dialog').isVisible());
      await page.keyboard.press('Escape');
      assert.equal(await page.getByRole('dialog').count(), 0);
      assert.deepEqual(errors, [], `${theme}/${device} browser errors`);
      results.push({ theme, device, viewport, passed: true, checks: ['render', 'no horizontal overflow', 'chart switching', 'objective selection', 'architecture details', 'walkthrough', 'engine tabs', 'methodology dialog', 'Escape dismissal', 'no browser errors'] });
      await page.close();
    }
  }
  await writeFile(new URL('../screenshots/verification.json', import.meta.url), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
} finally {
  await browser.close();
}
