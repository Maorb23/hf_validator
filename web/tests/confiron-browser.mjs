import assert from 'node:assert/strict';
import {readFile,mkdir} from 'node:fs/promises';
import {chromium} from '@playwright/test';
import {objectiveUplifts} from '../src/components/uplift.mjs';
const publication=JSON.parse(await readFile(new URL('../public/data/publication.json',import.meta.url),'utf8'));
const baselines=JSON.parse(await readFile(new URL('../public/data/uplift-baselines.json',import.meta.url),'utf8')).rows;
const origin=process.env.INFIRO_URL||'http://127.0.0.1:8432';
const fmt=n=>n.toLocaleString('en',{maximumFractionDigits:1});
await mkdir(new URL('../screenshots/',import.meta.url),{recursive:true});
const browser=await chromium.launch();
try{
 for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
  const page=await browser.newPage({viewport,reducedMotion:'reduce'}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(origin,{waitUntil:'networkidle'});
  await page.getByRole('heading',{name:'Configuration is boring. Let Confiron do it for you.'}).waitFor();
  assert.match(await page.title(),/Confiron/);
  assert.equal(await page.getByRole('link',{name:'Confiron home',exact:true}).count(),1);
  await page.getByRole('heading',{name:'Automatic optimization. Your objective.',exact:true}).waitFor();
  await page.getByRole('heading',{name:'Configuration matters.',exact:true}).waitFor();
  assert.equal(await page.locator('.user-experience h3').count(),3);
  const landing=page.getByRole('region',{name:'Best results versus measured baseline'});
  await landing.waitFor();
  for(const model of baselines.map(b=>b.model)){
   await landing.getByLabel('Uplift model',{exact:true}).selectOption(model);
   const rows=publication.rows.filter(r=>r.context.model===model);
   assert.deepEqual(await landing.locator('.uplift-metric>strong').allTextContents(),objectiveUplifts(rows,baselines).map(s=>s.percent===null?'Unavailable':`${fmt(s.percent)}%`));
  }
  await page.evaluate(()=>document.querySelectorAll('[data-reveal]').forEach(n=>n.classList.add('visible')));
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'home overflow');
  await page.screenshot({path:new URL(`../screenshots/confiron-home-${viewport.width}.png`,import.meta.url).pathname,fullPage:true});
  await page.goto(origin+'/benchmarks',{waitUntil:'networkidle'});
  const panel=page.getByRole('region',{name:'Best results versus measured baseline'});
  for(const model of [...new Set(publication.rows.map(r=>r.context.model))]){
   await page.getByLabel('Measured model',{exact:true}).selectOption(model);
   await page.getByRole('button',{name:'Apply filters',exact:true}).click();
   await panel.waitFor();
   const rows=publication.rows.filter(r=>r.context.model===model);
   assert.deepEqual(await panel.locator('.uplift-metric>strong').allTextContents(),objectiveUplifts(rows,baselines).map(s=>s.percent===null?'Unavailable':`${fmt(s.percent)}%`));
   const engine=rows.find(r=>r.plotEligible)?.engine;
   if(engine){
    await page.getByLabel('Measured engine',{exact:true}).selectOption(engine);
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
    assert.deepEqual(await panel.locator('.uplift-metric>strong').allTextContents(),objectiveUplifts(rows,baselines).map(s=>s.percent===null?'Unavailable':`${fmt(s.percent)}%`));
    await page.getByLabel('Measured engine',{exact:true}).selectOption('all');
    await page.getByRole('button',{name:'Apply filters',exact:true}).click();
   }
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`${model} overflow`);
  }
  await panel.scrollIntoViewIfNeeded();
  await page.screenshot({path:new URL(`../screenshots/confiron-benchmarks-${viewport.width}.png`,import.meta.url).pathname,fullPage:true});
  assert.deepEqual(errors,[]);await page.close();
 }
 console.log('PASS: Confiron branding and landing copy; best per-model metrics across frameworks versus measured baselines for all five models; desktop/mobile layout.');
}finally{await browser.close()}
