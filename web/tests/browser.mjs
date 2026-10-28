import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';

const origin=process.env.INFIRO_URL || 'http://127.0.0.1:4321';
const routes=['/','/technology','/architecture','/benchmarks','/methodology','/evaluation-guide','/demo','/demo/search','/demo/pareto','/demo/journey','/demo/studies'];
const browser=await chromium.launch();
const results=[];
await mkdir(new URL('../screenshots/',import.meta.url),{recursive:true});
try {
  const readiness=await browser.newPage();
  let ready=false;
  for(let attempt=0;attempt<30;attempt++){
    try{const response=await readiness.request.get(origin,{timeout:3000});if(response.ok()){ready=true;break;}}catch{}
    await readiness.waitForTimeout(1000);
  }
  await readiness.close();
  assert.ok(ready,'Local preview did not become ready');
  for(const [device,viewport] of Object.entries({desktop:{width:1440,height:1000},mobile:{width:390,height:844}})){
    const context=await browser.newContext({viewport,reducedMotion:'reduce',deviceScaleFactor:1,isMobile:device==='mobile',hasTouch:device==='mobile'});
    const page=await context.newPage();
    const errors=[];page.on('pageerror',e=>{errors.push(e.message);console.error('Browser error:',e.message);});
    page.on('console',message=>{if(message.type()==='error')console.error('Browser console:',message.text());});
    const localLinks=new Set();
    for(const route of routes){
      const response=await page.goto(origin+route,{waitUntil:'networkidle'});
      console.log('Checking',device,route);
      assert.equal(response.status(),200,route);
      await page.evaluate(()=>document.fonts.ready);
      if(['/demo/search','/benchmarks','/demo/pareto'].includes(route))await page.waitForSelector('.lab-root .control-bar');
      assert.equal(await page.locator('main h1').count(),1,`${route} needs one h1`);
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth ? [...document.querySelectorAll('main *')].map(el=>({tag:el.tagName,cls:el.getAttribute('class'),right:Math.round(el.getBoundingClientRect().right)})).filter(el=>el.right>innerWidth).slice(0,8) : []);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`${route}/${device} horizontal overflow: ${JSON.stringify(overflow)}`);
      for(const href of await page.locator('a[href^="/"]').evaluateAll(links=>links.map(a=>a.getAttribute('href'))))localLinks.add(href.split('#')[0].split('?')[0]);
      const name=route==='/'?'home':route.slice(1).replaceAll('/','-');
      await page.screenshot({path:new URL(`../screenshots/${name}-${device}.png`,import.meta.url).pathname,fullPage:true,animations:'disabled'});
      results.push({route,device,render:true,noOverflow:true});
    }
    await page.goto(origin+'/demo/search',{waitUntil:'networkidle'});
    await page.getByRole('button',{name:'Step →',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.playback output')?.textContent==='13 / 24');
    await page.getByLabel('Seed',{exact:true}).selectOption('29');
    await page.waitForFunction(()=>document.querySelector('.playback output')?.textContent==='0 / 24');
    await page.getByLabel('Objective',{exact:true}).selectOption('hills');
    await page.getByRole('button',{name:'Gaussian process',exact:true}).click();
    await page.getByLabel('Observed trials',{exact:true}).fill('24');
    assert.ok(await page.getByRole('heading',{name:'Trial 24 · Gaussian process',exact:true}).isVisible());
    await page.getByRole('button',{name:'Reset',exact:true}).click();
    await page.getByRole('button',{name:'Play',exact:false}).click();
    await page.waitForFunction(()=>document.querySelector('.playback output')?.textContent!=='0 / 24');
    await page.getByRole('button',{name:'Pause',exact:false}).click();
    await page.goto(origin+'/benchmarks',{waitUntil:'networkidle'});
    await page.waitForSelector('.control-bar');
    await page.getByRole('button',{name:'Simulated examples',exact:true}).click();
    await page.getByLabel('Engine label',{exact:true}).selectOption('vLLM');
    const checks=page.getByRole('checkbox',{name:/Compare short-/});
    await checks.nth(0).check();await checks.nth(1).check();
    assert.ok(await page.locator('.comparison').getByText('2 / 3',{exact:true}).isVisible());
    await page.getByLabel('Workload',{exact:true}).selectOption('long');
    assert.ok(await page.locator('.comparison').getByText('0 / 3',{exact:true}).isVisible());
    await page.getByLabel('Latency ceiling',{exact:true}).fill('80');
    assert.ok(await page.getByText('No examples meet this latency ceiling.',{exact:false}).isVisible());
    await page.getByLabel('Pareto points only',{exact:true}).check();
    await page.getByRole('button',{name:'Published measurements',exact:true}).click();
    await page.waitForSelector('.measured-controls');
    assert.ok(await page.getByRole('heading',{name:'Throughput and p95 latency.',exact:true}).isVisible());
    const initialBest=await page.locator('.measured-best-value').textContent();
    await page.getByLabel('Measured engine',{exact:true}).selectOption('TensorRT-LLM');
    assert.equal(await page.getByLabel('Measured cohort',{exact:true}).inputValue(),'all');
    assert.equal(await page.locator('.measured-best-value').textContent(),initialBest,'Unapplied changes must not change results');
    assert.ok(await page.getByText('Changes pending. Apply filters to update results.',{exact:true}).isVisible());
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
    assert.equal(await page.locator('.measured-best-value').textContent(),'7,911.5 output tok/s');
    await page.getByRole('button',{name:'Inspect highest-throughput trial',exact:true}).click();
    assert.ok(await page.locator('.result-inspector').getByText('9y5kjcb9-13 · COMPLETE',{exact:true}).isVisible());
    assert.ok(await page.getByRole('group',{name:'Measured output throughput versus p95 end-to-end latency',exact:true}).isVisible());
    assert.ok(await page.locator('.measured-explorer tbody tr').first().getByText(/^TensorRT-LLM · /).isVisible());
    await page.screenshot({path:new URL(`../screenshots/benchmarks-trtllm-${device}.png`,import.meta.url).pathname,fullPage:true});
    const specificGroup=await page.getByLabel('Measured cohort',{exact:true}).locator('option').nth(1).getAttribute('value');
    await page.getByLabel('Measured cohort',{exact:true}).selectOption(specificGroup);
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
    assert.notEqual(await page.locator('.measured-best-value').textContent(),'7,911.5 output tok/s');
    await page.getByLabel('Measured cohort',{exact:true}).selectOption('all');
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
    assert.equal(await page.locator('.measured-best-value').textContent(),'7,911.5 output tok/s');
    for(const engine of ['vLLM','SGLang']){
      await page.getByLabel('Measured engine',{exact:true}).selectOption(engine);
      await page.getByRole('button',{name:'Apply filters',exact:true}).click();
      assert.ok(await page.getByRole('group',{name:'Measured output throughput versus p95 end-to-end latency',exact:true}).isVisible());
      assert.ok(await page.locator('.measured-explorer tbody tr').first().getByText(new RegExp(`^${engine} · `)).isVisible());
    }
    await page.getByLabel('Measured engine',{exact:true}).selectOption('all');
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
    const measuredChecks=page.getByRole('checkbox',{name:/Compare measured /});
    await measuredChecks.nth(0).check();await page.locator('input[aria-label^="Compare measured "]:not(:disabled)').nth(1).check();
    assert.ok(await page.locator('.measured-explorer .comparison').getByText('2 / 3',{exact:true}).isVisible());
    await page.getByLabel('Measured model',{exact:true}).selectOption('Qwen/Qwen3-4B');
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
    assert.ok(await page.locator('.measured-explorer .comparison').getByText('0 / 3',{exact:true}).isVisible());
    const runId=await page.getByLabel('Measured run',{exact:true}).locator('option').nth(1).getAttribute('value');
    await page.getByLabel('Measured run',{exact:true}).selectOption(runId);
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
    await page.getByLabel('Measured replay progress',{exact:true}).fill('0');
    assert.ok(await page.getByText('No completed measurements with valid throughput and p95 latency in this selection.',{exact:false}).isVisible());
    await page.getByRole('button',{name:'Step trial →',exact:true}).click();
    assert.match(await page.locator('.measured-replay output').textContent(),/^1 \/ /);
    await page.getByRole('button',{name:'Show all trials',exact:true}).click();
    await page.getByRole('button',{name:'Play replay',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.measured-replay output')?.textContent.startsWith('1 /'));
    await page.getByRole('button',{name:'Pause replay',exact:true}).click();
    await page.screenshot({path:new URL(`../screenshots/benchmarks-measured-${device}.png`,import.meta.url).pathname,fullPage:true});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Measured controls overflow');
    await page.goto(origin+'/demo/journey',{waitUntil:'networkidle'});
    await page.getByRole('button',{name:/Next step/}).click();
    assert.ok(await page.getByRole('heading',{name:'Separate evidence from assumptions.',exact:true}).isVisible());
    await page.getByRole('button',{name:/Compare and validate/}).click();
    await page.getByRole('button',{name:/Start again/}).click();
    assert.ok(await page.getByRole('heading',{name:'Start with the decision.',exact:true}).isVisible());
    await page.goto(origin+'/architecture',{waitUntil:'networkidle'});
    await page.getByRole('button',{name:/Nebius Jobs Remote execution/}).click();
    assert.ok(await page.getByRole('heading',{name:'Run the workload remotely.',exact:true}).isVisible());
    if(device==='mobile'){
      await page.getByRole('button',{name:'Open navigation',exact:true}).click();
      assert.ok(await page.getByRole('navigation',{name:'Mobile navigation',exact:true}).isVisible());
      await page.keyboard.press('Escape');
      assert.equal(await page.getByRole('navigation',{name:'Mobile navigation',exact:true}).count(),0);
    }
    const missing=await page.request.get(origin+'/a-page-that-does-not-exist');assert.equal(missing.status(),404);
    for(const link of localLinks){const r=await page.request.get(origin+link);assert.ok(r.status()<400,`${link} broken: ${r.status()}`);}
    assert.deepEqual(errors,[],`${device} browser errors`);
    results.push({device,interactions:true,localLinks:localLinks.size,browserErrors:errors});
    await context.close();
  }
  // An unavailable dataset must produce a recoverable state, not an empty chart.
  const page=await browser.newPage();
  await page.route('**/data/simulations.json',route=>route.fulfill({status:503,body:'unavailable'}));
  await page.goto(origin+'/demo/search',{waitUntil:'networkidle'});
  assert.ok(await page.getByRole('heading',{name:'The dataset could not be loaded.',exact:true}).isVisible());
  await page.unroute('**/data/simulations.json');
  await page.getByRole('button',{name:'Try again',exact:true}).click();
  await page.waitForSelector('.playback');
  await page.close();
  await writeFile(new URL('../screenshots/verification.json',import.meta.url),JSON.stringify(results,null,2));
  console.log(JSON.stringify({pages:routes.length,viewports:2,checks:'render, overflow, interactions, downloads, local links, 404, dataset recovery',results},null,2));
}finally{await browser.close();}
