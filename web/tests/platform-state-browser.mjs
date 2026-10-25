// Isolated UI regression checks: mock private APIs; no credentials, database or provider jobs.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {extname,resolve} from 'node:path';
import {chromium} from '@playwright/test';

const dist=resolve('dist');
const types={'.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2'};
const server=createServer(async(req,res)=>{
 try{
  const path=new URL(req.url,'http://localhost').pathname;
  const file=path.startsWith('/app')?resolve(dist,'app/index.html'):resolve(dist,'.'+path);
  if(!file.startsWith(dist+'/')){res.writeHead(404).end();return}
  res.setHeader('Content-Type',types[extname(file)]||'text/html');
  res.end(await readFile(file));
 }catch{res.writeHead(404).end()}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin='http://127.0.0.1:'+server.address().port;
let browser;
try{
 browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1200,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const choices={trials:1,dry_run:true,num_prompts:100,input_tokens:512,output_tokens:128,concurrency:24,seed:1};
 const previous={id:'previous',name:'Previous study',recommendation_id:'rec',choices,summary:{},status:'prepared',configuration_files:[]};
 const data={user:{username:'test'},connections:[{id:'connection',kind:'nebius_inference',name:'Test inference'}],datasets:[],jobs:[],recommendations:[{id:'rec',selected:{model_id:'org/model',revision:'a'.repeat(40)},connection_refs:{nebius_inference:'connection'}}],prepared:[previous],live_enabled:true,tracking_configuration:{project:'tracking-test'}};
 let submitted,prepared;
 await page.route('**/app/api/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('/bootstrap'))return route.fulfill({json:data});
  if(path.endsWith('/prepared')&&route.request().method()==='POST'){
   submitted=route.request().postDataJSON();
   prepared={...previous,id:'new',name:'New study',choices:submitted.choices,status:'queued',summary:{model:'org/model',dry_run:true,optuna:{trials:submitted.choices.trials}}};
   data.prepared.unshift(prepared);
   return route.fulfill({status:202,json:{prepared,job:{id:'preparation'}}});
  }
  if(path.endsWith('/diagnostics'))return route.fulfill({json:{frameworks:[{framework:'vllm',log_available:true,log_tail:'<script>window.logExecuted=true</script>\nSuccessful requests: 99',failure_reasons:['99/100 requests succeeded. Benchmark excluded.']}]}});
  return route.fulfill({json:{}});
 });
 await page.goto(origin+'/app/configure/rec');
 const trials=page.getByLabel('Trials per framework',{exact:true});
 await trials.fill('2');
 // A bootstrap refresh must not reset an edited count to the previous study's 1.
 await page.waitForTimeout(4200);
 assert.equal(await trials.inputValue(),'2');
 await page.getByRole('button',{name:'Prepare configuration',exact:true}).click();
 await page.getByRole('heading',{name:'Preparing your configuration…',exact:true}).waitFor();
 assert.equal(submitted.choices.trials,2);
 assert.equal(await page.getByRole('list',{name:'Study workflow'}).count(),0);
 assert.equal(await page.getByRole('link',{name:/Download draft/}).count(),0);
 assert.ok(await page.getByText('2 trials per framework.',{exact:false}).isVisible());
 prepared.status='prepared';
 await page.getByRole('heading',{name:'Configuration ready. Opening approval…',exact:true}).waitFor({timeout:12000});
 await page.getByRole('heading',{name:'New study',exact:true}).waitFor();
 assert.ok(await page.getByRole('button',{name:'Approve dry run',exact:true}).isDisabled());
 const job={id:'study',kind:'study',target_id:'new',state:'completed',outcome:'partial_success',trial_failures:[{framework:'vllm',reason:'1 trial failed.'}],progress:{recovered_provider_error:{message:'Old timeout'},jobs:[{framework:'vllm',state:'completed'}]},result:{summary:{candidates:[],trial_state_counts:{vllm:{COMPLETE:0,FAIL:1},sglang:{COMPLETE:1,FAIL:0}}}},artifacts:[]};
 data.jobs=[job];
 await page.goto(origin+'/app/studies/study');
 await page.getByText('Completed with failed trials',{exact:true}).waitFor();
 assert.equal(await page.getByRole('alert').count(),0);
 await page.getByRole('button',{name:'Load framework logs',exact:true}).click();
 await page.getByText('99/100 requests succeeded. Benchmark excluded.',{exact:true}).waitFor();
 await page.getByText('Execution log',{exact:true}).click();
 assert.ok(await page.getByText('<script>window.logExecuted=true</script>',{exact:false}).isVisible());
 assert.equal(await page.evaluate(()=>window.logExecuted),undefined);
 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
 prepared.status='failed';
 data.jobs.push({id:'preparation',kind:'prepare',target_id:'new',state:'failed',error:'Runtime check failed',progress:{}});
 await page.goto(origin+'/app/approval/new');
 await page.getByRole('heading',{name:'Preparation stopped',exact:true}).waitFor();
 assert.ok(await page.getByText('Runtime check failed',{exact:true}).isVisible());
 assert.ok(await page.getByRole('button',{name:'Review configuration',exact:true}).isVisible());
 assert.equal(await page.getByRole('button',{name:'Cancel preparation',exact:true}).count(),0);
 prepared.status='prepared';prepared.choices.dry_run=false;
 await page.goto(origin+'/app/approval/new');
 await page.getByText('This configuration has no W&B tracking, although the server has it configured.',{exact:false}).waitFor();
 assert.deepEqual(errors,[]);
 console.log('PASS: trial count survives polling and submission; simplified preparation and ready transition; explicit approval; partial success; recovered errors; safe logs; mobile layout.');
}finally{
 await browser?.close();
 await new Promise(resolve=>server.close(resolve));
}
