import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const origin=process.env.INFIRO_URL||'http://127.0.0.1:4321';
const browser=await chromium.launch();
const results=[];
const combinations=[['openbmb/MiniCPM5-2B','Baseline'],['google/gemma-4-12B-it','Baseline'],['google/gemma-4-12B-it','DSpark'],['Qwen/Qwen3-4B','Baseline'],['Qwen/Qwen3-4B','DSpark'],['Qwen/Qwen3-8B','Baseline'],['Qwen/Qwen3-8B','DSpark']];
await mkdir(new URL('../screenshots/',import.meta.url),{recursive:true});
try {
 for(const viewport of (process.argv.includes('--mobile')?[{width:390,height:844}]:[{width:1440,height:1000},{width:390,height:844}])) {
  const page=await browser.newPage({viewport,reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(origin+'/benchmarks',{waitUntil:'networkidle'});
  await page.getByLabel('Measured model',{exact:true}).waitFor();
  for(const [model,variant] of combinations) {
   await page.getByLabel('Measured model',{exact:true}).selectOption(model);
   await page.getByLabel('Measured variant',{exact:true}).selectOption(variant);
   await page.getByRole('button',{name:'Apply filters',exact:true}).click();
   assert.ok(await page.locator('.sample-point').count()>0);
   await page.locator('.sample-point').first().focus();
   await page.keyboard.press('Enter');
   assert.ok(await page.locator('.result-inspector').getByText(`${model} · ${variant}`,{exact:true}).isVisible());
   const record=page.locator('.result-inspector').getByText(/^Run metadata · /);
   await record.click();
   assert.ok(await page.locator('.result-inspector').getByRole('link',{name:'Original run record ↗'}).isVisible());
   assert.ok(await page.locator('.result-inspector').getByText(/Termination reason: Not reported/).isVisible());
   if(variant==='DSpark') {
    const details=page.locator('.result-inspector .technical-details');
    assert.match(await details.textContent(),/draftRevision/);
   }
   const details=page.locator('.result-inspector').getByText('Trial measurements, timings and logs',{exact:true});
   await details.click();
   const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth?[...document.querySelectorAll('main *')].filter(e=>!e.closest('.table-scroll')&&e.getBoundingClientRect().right>innerWidth).slice(0,30).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,80),width:e.getBoundingClientRect().width})):[]);
   if(overflow.length)await page.screenshot({path:new URL('../screenshots/benchmark-overflow.png',import.meta.url).pathname,fullPage:true});
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`${model}/${variant}/${viewport.width} overflow ${JSON.stringify(await page.evaluate(()=>({width:document.documentElement.scrollWidth,inner:innerWidth,scrollX})))}`);
   results.push({model,variant,width:viewport.width,keyboardSelection:true,sourceLink:true,metadata:true,noOverflow:true});
  }
  for(const id of ['0bxo3s5e','x911x1uq','by7lkkyu','7hksyocg']) {
   await page.getByLabel('Measured run',{exact:true}).selectOption(id);
   await page.getByRole('button',{name:'Apply filters',exact:true}).click();
   if(id==='7hksyocg') {
    assert.equal(await page.locator('.sample-point').count(),0);
    await page.locator('.measured-replay').getByText('Recorded context and missing metadata',{exact:true}).click();
    await page.locator('.measured-replay').getByText(`Run metadata · ${id}`,{exact:true}).click();
    assert.ok(await page.locator('.measured-replay').getByText(/Duration: 0 seconds/).isVisible());
   } else {
    assert.equal(await page.locator('.sample-point').count(),2);
    await page.locator('.sample-point').first().focus();await page.keyboard.press('Space');
    await page.locator('.result-inspector').getByText(`Run metadata · ${id}`,{exact:true}).click();
    assert.ok(await page.locator('.result-inspector').getByText(/Termination reason: Not reported/).isVisible());
   }
  }
  await page.screenshot({path:new URL(`../screenshots/benchmark-import-${viewport.width}.png`,import.meta.url).pathname,fullPage:true});
  assert.deepEqual(errors,[]);
  await page.close();
 }
 await writeFile(new URL('../screenshots/benchmark-import-verification.json',import.meta.url),JSON.stringify(results,null,2));
 console.log(JSON.stringify({checks:results.length,shortStudies:4,errors:0,results},null,2));
} finally {await browser.close();}
