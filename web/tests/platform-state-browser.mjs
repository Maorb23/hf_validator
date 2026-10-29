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
 const previous={id:'previous',name:"Previous run",recommendation_name:'Coding research',recommendation_id:'rec',choices,summary:{},status:'prepared',configuration_files:[{id:'request',name:'request.json',download_url:'/app/api/artifacts/request'}]};
 const labels={output_throughput:'Output throughput (tokens/second)',p95_e2e_latency_ms:'95th-percentile end-to-end latency (ms)',p95_ttft_ms:'95th-percentile time to first token (ms)'};
 const keys=Object.keys(labels),optimization_objectives=[...keys.map(k=>[k]),[keys[0],keys[1]],[keys[0],keys[2]],[keys[1],keys[2]],keys].map(names=>({value:names.join(','),label:(names.length>1?'Pareto · ':'')+names.map(m=>labels[m]).join(' + ')}));
 const data={optimization_objectives,user:{username:'test'},connections:[{id:'connection',kind:'nebius_inference',name:'Test inference'}],datasets:[],jobs:[],recommendations:[{id:'rec',name:'Coding research',requirements:{request:'A coding assistant'},recommendations:[],status:'recommended',selected:{model_id:'org/model',revision:'a'.repeat(40)},connection_refs:{nebius_inference:'connection'}}],prepared:[previous],live_enabled:true,tracking_configuration:{project:'tracking-test'}};
 let submitted,prepared,datasetSubmission,directSubmission,editedSubmission,approvedSubmission;
 let delayedBootstrap=false,releaseBootstrap,failBootstrap=false;
 let editorRequest={model:'org/model',timeout_minutes:60,optuna:{trials:2,sampler:'TPESampler',objectives:['mean_ttft_ms']},server_args:{vllm:['batch=[1]']}};
 await page.addInitScript(()=>Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>{window.copiedTemplate=text}}}));
 await page.route('**/app/api/**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('/bootstrap')){
   if(failBootstrap)return route.fulfill({status:503,json:{error:'Offline'}});
   if(delayedBootstrap){delayedBootstrap=false;const snapshot=JSON.parse(JSON.stringify(data));await new Promise(resolve=>{releaseBootstrap=resolve});return route.fulfill({json:snapshot}).catch(()=>{})}
   return route.fulfill({json:data});
  }
  if(route.request().method()==='DELETE'){
   if(path.endsWith('/trials')){
    const job=data.jobs.find(j=>path.includes('/'+j.id+'/'));
    const key=route.request().postDataJSON().trial_key;
    const match=t=>`${t.framework}:${t.trial_id??t.configuration_id}`!==key;
    job.result.summary.trials=job.result.summary.trials.filter(match);
    job.result.summary.candidates=job.result.summary.candidates.filter(match);
    job.result.analysis.pareto_front=job.result.analysis.pareto_front.filter(match);
   }else if(path.includes('/jobs/'))data.jobs=data.jobs.filter(j=>!path.endsWith('/'+j.id));
   else if(path.includes('/recommendations/'))data.recommendations=data.recommendations.filter(r=>!path.endsWith('/'+r.id));
   return route.fulfill({json:{deleted:true}});
  }
  if(path.endsWith('/models')&&route.request().method()==='POST'){
   directSubmission=route.request().postDataJSON();
   const rec={id:'direct',name:directSubmission.model.model_id,requirements:{request:'Direct model selection'},recommendations:[],selection_source:'direct',status:'ready',selected:{model_id:directSubmission.model.model_id,revision:'d'.repeat(40)},connection_refs:{}};
   data.recommendations.unshift(rec);
   return route.fulfill({status:202,json:{recommendation:rec,job:{id:'pin-model'}}});
  }
  if(path.endsWith('/request')){
   const source=data.prepared.find(p=>path.includes('/'+p.id+'/request'));
   if(route.request().method()==='GET')return route.fulfill({json:{request:editorRequest,digest:source.digest}});
   editedSubmission=route.request().postDataJSON();
   if(editedSubmission.request.model!==editorRequest.model)return route.fulfill({status:400,json:{error:'Protected fields changed: model.'}});
   source.invalidated=true;editorRequest=editedSubmission.request;
   const revised={...source,id:'edited',name:"Edited run",invalidated:false,digest:'new-digest',choices:{...source.choices,trials:editorRequest.optuna.trials,timeout_minutes:editorRequest.timeout_minutes},summary:{...source.summary,optuna:editorRequest.optuna,timeout_minutes:editorRequest.timeout_minutes,maximum_gpu_cost_usd:2}};
   data.prepared.unshift(revised);
   return route.fulfill({status:201,json:{prepared:revised}});
  }
  if(path.endsWith('/prepared/edited')&&route.request().method()==='POST'){
   approvedSubmission=route.request().postDataJSON();
   const job={id:'edited-study',kind:'study',target_id:'edited',state:'dry_run',progress:{},result:{},artifacts:[]};
   data.jobs.push(job);return route.fulfill({status:202,json:job});
  }
  if(path.endsWith('/recommendations/rec')&&route.request().method()==='PATCH'){
   data.recommendations[0].name=route.request().postDataJSON().name.trim();
   previous.recommendation_name=data.recommendations[0].name;
   return route.fulfill({json:data.recommendations[0]});
  }
  if(path.endsWith('/datasets')&&route.request().method()==='POST'){
   datasetSubmission=route.request().postDataJSON();
   data.datasets.push({id:'frozen',name:'Custom coding source',config:datasetSubmission.config,status:'ready',identity:{dataset:{repo_id:datasetSubmission.config.repo_id,revision:'b'.repeat(40),config:null,split:'train',data_file:datasetSubmission.config.data_file},sha256:'f'.repeat(64),tokenizer_revision:'a'.repeat(40),sample_count:100,seed:1},preview:[{prompt:'<script>window.promptExecuted=true</script>',prompt_len:512,output_len:128}]});
   return route.fulfill({status:202,json:{dataset:data.datasets.at(-1)}});
  }
  if(path.endsWith('/prepared')&&route.request().method()==='POST'){
   submitted=route.request().postDataJSON();
   prepared={...previous,id:'new',name:"New run",choices:submitted.choices,status:'queued',summary:{model:'org/model',dry_run:true,optuna:{trials:submitted.choices.trials}}};
   data.prepared.unshift(prepared);
   return route.fulfill({status:202,json:{prepared,job:{id:'preparation'}}});
  }
  if(path.endsWith('/rerun')){
   const original=data.jobs.find(j=>path.includes('/'+j.id+'/'));
   const source=data.prepared.find(p=>p.id===original.target_id);
   const revised={...source,id:'rerun-review',approved_at:null,digest:'rerun-digest',summary:{...source.summary,rerun_of:original.id}};
   data.prepared.unshift(revised);
   return route.fulfill({status:202,json:{prepared:revised}});
  }
  if(path.endsWith('/diagnostics'))return route.fulfill({json:{frameworks:[{framework:'vllm',log_available:true,log_tail:'<script>window.logExecuted=true</script>\nSuccessful requests: 99',failure_reasons:['99/100 requests succeeded. Benchmark excluded.']}]}});
  return route.fulfill({json:{}});
 });
 await page.goto(origin+'/app/connections');
 for(const label of ['Nebius inference','Nebius jobs','Hugging Face','object storage','Tavily']){
  const summary=page.getByText('Set up '+label,{exact:true});
  await summary.click();
  const help=summary.locator('..');
  assert.ok(await help.getByRole('list').isVisible());
  await help.getByRole('button',{name:'Copy command template',exact:true}).click();
  assert.ok((await page.evaluate(()=>window.copiedTemplate)).length>10);
 }
 await page.getByText('Set up shared W&B tracking',{exact:true}).click();
 const tracking=page.getByText('Set up shared W&B tracking',{exact:true}).locator('..');
 await tracking.getByRole('button',{name:'Copy command template',exact:true}).click();
 assert.match(await page.evaluate(()=>window.copiedTemplate),/INFIRO_WANDB/);
 assert.match(await page.evaluate(()=>window.copiedTemplate),/--prepared <PREPARED_CONFIGURATION_UUID>/);
 await page.getByRole('button',{name:'Add connection',exact:true}).click();
 await page.getByLabel('API key',{exact:true}).fill('typed-test-credential-never-copy');
 await page.getByLabel('How to set up api key',{exact:true}).click();
 assert.ok(await page.getByText('Create an API key in this provider’s console (Nebius Token Factory or Tavily). Paste the key, without a Bearer prefix.',{exact:true}).isVisible());
 await page.getByLabel('Provider',{exact:true}).selectOption('storage');
 assert.equal(await page.getByLabel('Session token (optional)',{exact:true}).count(),0);
 await page.getByLabel('Provider',{exact:true}).selectOption('nebius_jobs');
 await page.getByLabel('How to set up project id',{exact:true}).click();
 assert.ok(await page.getByText('Select the Nebius project in the console and copy its project ID. The service account must have Admin access in this project for this deployment.',{exact:true}).isVisible());
 await page.getByLabel('Provider',{exact:true}).selectOption('nebius_inference');
 const inferenceHelp=page.getByText('Set up Nebius inference',{exact:true}).locator('..');
 await inferenceHelp.getByRole('button',{name:/Template copied|Copy command template/}).click();
 assert.ok(!(await page.evaluate(()=>window.copiedTemplate)).includes('typed-test-credential'));
 data.jobs=[
  {id:'discovery-running',kind:'discover',target_id:'rec',state:'running',progress:{}},
  {id:'discovery-failed',kind:'discover',target_id:'rec',state:'failed',error:'A required worker dependency is missing. See the safe technical diagnostics.',progress:{}},
  {id:'discovery-cancelled',kind:'discover',target_id:'rec',state:'cancelled',progress:{}}
 ];
 await page.goto(origin+'/app/recommendations');
 await page.getByRole('heading',{name:'Background operations',exact:true}).waitFor();
 assert.equal(await page.getByLabel('Recommendation name',{exact:true}).count(),0,'Renaming stays collapsed');
 assert.equal(await page.getByText('A required worker dependency is missing. See the safe technical diagnostics.',{exact:true}).count(),0);
 assert.equal(await page.getByText('Operation cancelled.',{exact:true}).count(),0);
 assert.equal(await page.locator('.recommendations-operations .operation-row').count(),1);
 assert.equal(await page.locator('.recommendations-operations').getByRole('button',{name:'Retry',exact:true}).count(),0);
 const desktop=await page.evaluate(()=>({content:document.querySelector('.recommendations-content').getBoundingClientRect().toJSON(),operations:document.querySelector('.recommendations-operations').getBoundingClientRect().toJSON()}));
 assert.ok(desktop.operations.left>=desktop.content.right,'Operations sit to the right');
 assert.ok(desktop.operations.width<=280,'Operations remain compact');
 await page.setViewportSize({width:390,height:844});
 const mobile=await page.evaluate(()=>({content:document.querySelector('.recommendations-content').getBoundingClientRect().toJSON(),operations:document.querySelector('.recommendations-operations').getBoundingClientRect().toJSON()}));
 assert.ok(mobile.operations.top>=mobile.content.bottom,'Operations stack below on mobile');
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Recommendations mobile overflow');
 await page.setViewportSize({width:1200,height:900});
 // Hold a pre-action poll. Its stale response must never undo the successful action.
 delayedBootstrap=true;
 await page.evaluate(()=>window.dispatchEvent(new Event('focus')));

 for(let attempt=0;!releaseBootstrap&&attempt<100;attempt++)await new Promise(resolve=>setTimeout(resolve,50));
 assert.ok(releaseBootstrap,'A workspace poll started');
 const recommendationName='<script>window.nameExecuted=true</script>';
 await page.getByRole('button',{name:'Rename',exact:true}).click();
 await page.getByLabel('Recommendation name',{exact:true}).fill('  '+recommendationName+'  ');
 await page.getByRole('button',{name:'Save recommendation name',exact:true}).click();
 await page.getByRole('heading',{name:recommendationName,exact:true}).waitFor();
 releaseBootstrap();
 await page.waitForTimeout(200);
 assert.ok(await page.getByRole('heading',{name:recommendationName,exact:true}).isVisible());
 failBootstrap=true;
 await page.evaluate(()=>window.dispatchEvent(new Event('focus')));
 await page.getByRole('button',{name:'Retry updates',exact:true}).waitFor();
 failBootstrap=false;
 await page.getByRole('button',{name:'Retry updates',exact:true}).click();
 await page.getByRole('button',{name:'Retry updates',exact:true}).waitFor({state:'detached'});
 assert.equal(await page.evaluate(()=>window.nameExecuted),undefined);
 await page.goto(origin+'/app/overview');
 await page.getByRole('button',{name:recommendationName,exact:true}).waitFor();
 await page.goto(origin+'/app/configure/rec');
 await page.getByRole('heading',{name:'Previous configuration files',exact:true}).waitFor();
 await page.getByRole('button',{name:'Validate a custom dataset repository and JSON file',exact:true}).click();
 assert.equal(await page.getByLabel('Hugging Face repository ID or URL').inputValue(),'');
 assert.equal(await page.getByLabel('JSON data file',{exact:true}).inputValue(),'');
 await page.getByLabel('Hugging Face repository ID or URL').fill('org/small-conversations');
 await page.getByLabel('Revision',{exact:true}).fill('reviewed-branch');
 assert.equal(await page.getByLabel('Configuration (optional)',{exact:true}).count(),0);
 assert.equal(await page.getByLabel('Split',{exact:true}).count(),0);
 assert.equal(await page.getByLabel('Dataset format',{exact:true}).inputValue(),'sharegpt');
 await page.getByLabel('Dataset format',{exact:true}).selectOption('alpaca');
 await page.getByLabel('Dataset format',{exact:true}).selectOption('messages');
 await page.getByLabel('JSON data file',{exact:true}).fill('conversations.json');
 await page.getByRole('button',{name:'Validate & freeze workload',exact:true}).click();
 await page.getByText('Bounded preview',{exact:true}).waitFor();
 assert.equal(datasetSubmission.config.format,'messages');
 assert.equal(datasetSubmission.config.model,'org/model');
 assert.equal(datasetSubmission.config.model_revision,'a'.repeat(40));
 assert.equal(datasetSubmission.config.revision,'reviewed-branch');
 assert.equal(datasetSubmission.config.data_file,'conversations.json');
 assert.equal(datasetSubmission.config.config,undefined);
 assert.equal(datasetSubmission.config.split,undefined);
 assert.equal(await page.evaluate(()=>window.promptExecuted),undefined);
 // Closing inline validation keeps Configure's workload controls unambiguous.
 await page.getByRole('button',{name:'Validate a custom dataset repository and JSON file',exact:true}).click();
 await page.goto(origin+'/app/datasets');
 await page.getByRole('button',{name:'Add HF dataset',exact:true}).click();
 assert.equal(await page.getByLabel('Hugging Face repository ID or URL').inputValue(),'');
 assert.equal(await page.getByLabel('JSON data file',{exact:true}).inputValue(),'');
 await page.getByLabel('Dataset name',{exact:true}).fill('Draft');
 await page.getByLabel('Hugging Face repository ID or URL').fill('org/draft');
 await page.getByLabel('JSON data file',{exact:true}).fill('draft.json');
 await page.getByLabel('Dataset format',{exact:true}).selectOption('alpaca');
 await page.getByRole('button',{name:'Add HF dataset',exact:true}).click();
 assert.equal(await page.getByLabel('Dataset name',{exact:true}).inputValue(),'');
 assert.equal(await page.getByLabel('Hugging Face repository ID or URL').inputValue(),'');
 assert.equal(await page.getByLabel('JSON data file',{exact:true}).inputValue(),'');
 assert.equal(await page.getByLabel('Dataset format',{exact:true}).inputValue(),'sharegpt');
 assert.equal(data.datasets.length,1);
 await page.getByRole('heading',{name:'Custom coding source',exact:true}).waitFor();
 await page.goto(origin+'/app/configure/rec');
 const source=page.getByLabel('Dataset source',{exact:true});
 await source.selectOption('sharegpt');
 assert.equal(await source.locator('option[value="random"]').count(),0);
 await source.selectOption('frozen');
 assert.ok(await page.getByText('f'.repeat(64),{exact:true}).isVisible());

 const trials=page.getByLabel('Trials per framework',{exact:true});
 const objective=page.getByLabel('Optimization objective',{exact:true});
 assert.equal(await objective.locator('option').count(),7);
 for(const option of optimization_objectives){await objective.selectOption(option.value);assert.equal(await objective.inputValue(),option.value)}
 await page.getByLabel('Run name',{exact:true}).fill('P95 run');
 await objective.selectOption('p95_ttft_ms');
 await page.getByLabel('Sampler',{exact:true}).selectOption('tpe_multivariate');
 await page.getByLabel('Pruner',{exact:true}).selectOption('MedianPruner');
 await objective.selectOption(await objective.locator('option').last().getAttribute('value'));
 assert.ok(await page.getByLabel('Pruner',{exact:true}).isDisabled());
 assert.equal(await page.getByLabel('Pruner',{exact:true}).inputValue(),'NopPruner');
 await objective.selectOption('p95_ttft_ms');
 await page.getByLabel('Pruner',{exact:true}).selectOption('MedianPruner');
 await trials.fill('2');
 // A bootstrap refresh must not reset an edited count to the previous study's 1.
 await page.waitForTimeout(4200);
 assert.equal(await trials.inputValue(),'2');
 await page.getByRole('button',{name:'Prepare configuration',exact:true}).click();
 await page.getByRole('heading',{name:'Preparing your configuration…',exact:true}).waitFor();
 assert.equal(submitted.name,'P95 run');
 assert.equal(submitted.choices.trials,2);
 assert.deepEqual(submitted.choices.objectives,['p95_ttft_ms']);
 assert.equal(submitted.choices.sampler,'TPESampler');
 assert.equal(submitted.choices.tpe_multivariate,true);
 assert.equal(submitted.choices.pruner,'MedianPruner');
 assert.equal(submitted.dataset_id,'frozen');
 assert.equal(submitted.choices.workload_mode,'sharegpt');
 assert.equal(await page.getByRole('list',{name:"Run workflow"}).count(),0);
 assert.equal(await page.getByRole('link',{name:/Download draft/}).count(),0);
 assert.ok(await page.getByText('2 trials per framework.',{exact:false}).isVisible());
 prepared.status='prepared';
 await page.getByRole('heading',{name:'Configuration ready. Opening approval…',exact:true}).waitFor({timeout:12000});
 await page.getByRole('heading',{name:"New run",exact:true}).waitFor();
 assert.ok(await page.getByText('TPE (multivariate)',{exact:true}).isVisible());
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
 const measured=[{framework:'vllm',trial_id:'0',configuration_id:'c0',configuration:{batch:4},metrics:{p95_e2e_latency_ms:32,p95_ttft_ms:12,output_throughput:120},timings:{startup_to_ready_s:6.5,benchmark_s:2}},
 {framework:'sglang',trial_id:'1',configuration_id:'c1',configuration:{batch:8},metrics:{p95_e2e_latency_ms:24,p95_ttft_ms:10,output_throughput:100},timings:{}}];
 job.result.summary.candidates=measured;
 job.result.summary.trials=[...measured.map(c=>({...c,status:'COMPLETE'})),{framework:'vllm',trial_id:'2',status:'FAIL',metrics:{},configuration:{},reason:'Startup failed',timings:{}}];
 job.result.summary.framework_timings={vllm:{queue_wait_s:4.5,startup_to_ready_s:6.5,benchmark_s:2}};
 job.result.analysis={pareto_front:measured};
 prepared.choices.objectives=keys;
 await page.goto(origin+'/app/studies/study');
 await page.getByLabel('Plot vertical objective',{exact:true}).selectOption('p95_ttft_ms');
 await page.getByLabel('Plot horizontal objective',{exact:true}).selectOption('p95_e2e_latency_ms');
 assert.equal(await page.getByLabel('Plot vertical objective',{exact:true}).inputValue(),'p95_ttft_ms');
 const points=page.getByRole('button',{name:/Inspect .*Trial .*95th/});
 await points.first().waitFor();
 assert.equal(await points.count(),2);
 const fills=await points.locator('.point-mark').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('fill')));
 assert.notEqual(fills[0],fills[1]);
 assert.equal(await points.nth(1).locator('.point-mark').getAttribute('stroke'),'#fff');
 await points.first().focus();
 assert.match(await page.getByRole('status').last().innerText(),/vllm.*Trial 0.*12/);
 await page.keyboard.press('Enter');
 const inspector=page.getByRole('region',{name:'Trial inspector'});
 await inspector.waitFor();
 assert.match(await inspector.innerText(),/batch/);
 assert.ok(await inspector.getByRole('heading',{name:'vllm · Trial 0'}).evaluate(n=>n===document.activeElement));
 await page.keyboard.press('Escape');
 assert.equal(await inspector.count(),0);
 assert.ok(await points.first().evaluate(n=>n===document.activeElement));
 await page.getByRole('button',{name:'Inspect vllm trial 2',exact:true}).click();
 await inspector.getByText('Startup failed',{exact:true}).waitFor();
 assert.match(await inspector.innerText(),/Unavailable/);
 await inspector.getByRole('button',{name:'Load inspector logs',exact:true}).click();
 await inspector.getByText('<script>window.logExecuted=true</script>',{exact:false}).waitFor();
 assert.equal(await page.evaluate(()=>window.logExecuted),undefined);
 await page.screenshot({path:'/tmp/optimizer-study-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 await page.getByRole('button',{name:'Close inspector',exact:true}).click();
 await page.getByRole('button',{name:'Inspect sglang trial 1',exact:true}).click();
 await inspector.getByRole('heading',{name:'sglang · Trial 1'}).waitFor();
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
 await page.screenshot({path:'/tmp/optimizer-study-mobile.png',fullPage:true});
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

 // Direct model selection works on mobile without submitting a recommendation.
 await page.setViewportSize({width:1200,height:900});
 await page.goto(origin+'/app/studies/new');
 await page.getByLabel('Model ID',{exact:true}).fill('org/my-model');
 await page.getByLabel('Model revision',{exact:true}).fill('release');
 await page.screenshot({path:'/tmp/direct-model-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 await page.getByRole('button',{name:'Use this model',exact:true}).click();
 await page.getByRole('heading',{name:"Configure your run",exact:true}).waitFor();
 assert.deepEqual(directSubmission.model,{model_id:'org/my-model',revision:'release'});
 assert.ok(await page.getByText('d'.repeat(40),{exact:true}).isVisible());
 assert.ok(await page.getByText('Direct model selection · discovery was skipped.',{exact:true}).isVisible());
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'direct model mobile overflow');

 // Identical selections remain separate runs; deleting one persists across reloads.
 data.recommendations.unshift({...data.recommendations.find(r=>r.id==='direct'),id:'direct-copy'});
 await page.goto(origin+'/app/studies');
 await page.getByRole('heading',{name:'Runs',exact:true}).waitFor();
 assert.equal(await page.getByRole('link',{name:/^Runs(?:\s*\d+)?$/,includeHidden:true}).count(),1);
 const selections=page.locator('section.panel').filter({has:page.getByRole('heading',{name:'Direct model selections',exact:true})});
 assert.equal(await selections.getByRole('button',{name:'Delete model selection run',exact:true}).count(),2);
 await selections.getByRole('button',{name:'Delete model selection run',exact:true}).first().click();
 await selections.getByRole('button',{name:'Keep history',exact:true}).click();
 assert.equal(data.recommendations.filter(r=>r.selection_source==='direct').length,2);
 await selections.getByRole('button',{name:'Delete model selection run',exact:true}).first().click();
 await selections.getByRole('button',{name:'Confirm deletion',exact:true}).click();
 assert.ok(!data.recommendations.some(r=>r.id==='direct-copy'));
 assert.ok(data.recommendations.some(r=>r.id==='direct'));
 await page.reload();
 await selections.getByRole('button',{name:'Delete model selection run',exact:true}).waitFor();
 assert.equal(await selections.getByRole('button',{name:'Delete model selection run',exact:true}).count(),1);
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'direct model deletion mobile overflow');

 // Unsaved or invalid request edits cannot reuse previous approval consent.
 const editable={...previous,id:'editable',name:"Editable run",digest:'old-digest',choices:{...choices,concurrency:1,dry_run:true},summary:{model:'org/model',dry_run:true,optuna:editorRequest.optuna},invalidated:false};
 data.prepared.unshift(editable);
 await page.goto(origin+'/app/approval/editable');
 await page.getByLabel('I reviewed the pinned model, workload, allocation, connections and estimated costs. I explicitly approve this exact configuration.').check();
 await page.getByRole('button',{name:'Edit request.json',exact:true}).click();
 const editor=page.getByLabel('Request JSON editor',{exact:true});
 await editor.waitFor();
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'open request editor mobile overflow');
 await page.screenshot({path:'/tmp/request-editor-mobile.png',fullPage:true});
 await page.setViewportSize({width:1200,height:900});
 await editor.focus();
 assert.ok(await editor.evaluate(n=>n===document.activeElement));
 await page.screenshot({path:'/tmp/request-editor-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.getByRole('button',{name:'Approve dry run',exact:true}).isDisabled());
 await editor.fill('{broken');
 await page.getByRole('button',{name:'Validate & save request',exact:true}).click();
 await page.getByText('Request JSON is invalid. Check its syntax.',{exact:true}).waitFor();
 await editor.fill(JSON.stringify({...editorRequest,model:'org/other'}));
 await page.getByRole('button',{name:'Validate & save request',exact:true}).click();
 await page.getByRole('alert').filter({hasText:'Protected fields changed: model.'}).first().waitFor();
 const uploaded={...editorRequest,timeout_minutes:120,optuna:{...editorRequest.optuna,trials:3}};
 await page.getByLabel('Upload request.json',{exact:true}).setInputFiles({name:'request.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(uploaded))});
 await page.getByRole('button',{name:'Validate & save request',exact:true}).click();
 await page.getByRole('heading',{name:"Edited run",exact:true}).waitFor();
 assert.equal(editedSubmission.digest,'old-digest');
 assert.equal(editedSubmission.request.optuna.trials,3);
 assert.equal(await page.getByRole('button',{name:'Approve dry run',exact:true}).isDisabled(),true);
 assert.equal(data.jobs.filter(j=>j.target_id==='edited').length,0);
 assert.ok(await page.getByText('new-digest',{exact:true}).isVisible());
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'request editor mobile overflow');
 await page.getByLabel('I reviewed the pinned model, workload, allocation, connections and estimated costs. I explicitly approve this exact configuration.').check();
 await page.getByRole('button',{name:'Approve dry run',exact:true}).click();
 await page.getByRole('heading',{name:"Edited run",exact:true}).waitFor();
 assert.equal(approvedSubmission.digest,'new-digest');
 // Rerun opens a new review without launching a study.
 await page.goto(origin+'/app/studies/study');
 const studyCount=data.jobs.filter(j=>j.kind==='study').length;
 await page.getByRole('button',{name:'Rerun / edit and rerun',exact:true}).click();
 await page.waitForURL('**/app/approval/rerun-review');
 assert.equal(data.jobs.filter(j=>j.kind==='study').length,studyCount);
 await page.getByRole('button',{name:'Edit request.json',exact:true}).waitFor();
 assert.ok(data.jobs.some(j=>j.id==='study'));
 // Delete one trial, then its past study, then a saved recommendation.
 await page.goto(origin+'/app/studies/study');
 await page.getByRole('button',{name:'Inspect vllm trial 0',exact:true}).click();
 await page.getByRole('button',{name:'Delete trial',exact:true}).click();
 await page.getByRole('button',{name:'Keep history',exact:true}).click();
 assert.ok(await page.getByRole('button',{name:'Inspect vllm trial 0',exact:true}).isVisible());
 await page.getByRole('button',{name:'Delete trial',exact:true}).click();
 await page.getByRole('button',{name:'Confirm deletion',exact:true}).click();
 await page.getByRole('button',{name:'Inspect vllm trial 0',exact:true}).waitFor({state:'detached'});
 assert.equal(await page.getByRole('button',{name:'Inspect vllm trial 0',exact:true}).count(),0);
 await page.reload();
 await page.getByRole('button',{name:'Inspect sglang trial 1',exact:true}).waitFor();
 await page.getByRole('button',{name:'Inspect vllm trial 0',exact:true}).waitFor({state:'detached'});
 assert.equal(await page.getByRole('button',{name:'Inspect vllm trial 0',exact:true}).count(),0);
 await page.goto(origin+'/app/studies');
 for(const viewport of [{width:1200,height:900},{width:390,height:844}]){
  await page.setViewportSize(viewport);
  const historyLink=page.locator(".history-study-row > button:first-child").first();
  await historyLink.waitFor();
  assert.equal(await historyLink.evaluate(el=>getComputedStyle(el).backgroundColor),'rgba(0, 0, 0, 0)',"Run navigation must not use native grey button styling");
  assert.equal(await historyLink.evaluate(el=>getComputedStyle(el).display),'flex');
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),"Run history overflow");
  await page.screenshot({path:`/tmp/study-history-${viewport.width}.png`,fullPage:true});
 }

 await page.goto(origin+'/app/studies/study');
 await page.getByRole('button',{name:"Delete run",exact:true}).click();
 await page.getByRole('button',{name:'Confirm deletion',exact:true}).click();
 await page.getByRole('heading',{name:"Runs",exact:true}).waitFor();
 assert.ok(!data.jobs.some(j=>j.id==='study'));
 data.jobs=data.jobs.filter(j=>j.kind!=='discover');
 await page.goto(origin+'/app/recommendations');
 assert.equal(await page.getByRole('heading',{name:'Background operations',exact:true}).count(),0);
 await page.getByRole('button',{name:'Delete recommendation',exact:true}).first().click();
 await page.getByRole('button',{name:'Confirm deletion',exact:true}).click();
 assert.ok(!data.recommendations.some(r=>r.id==='rec'));
 await page.reload();
 await page.getByRole('heading',{name:'Recommendations',exact:true}).waitFor();
 assert.equal(await page.getByRole('button',{name:'Delete recommendation',exact:true}).count(),0);
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'History controls mobile overflow');
 assert.deepEqual(errors,[]);
 console.log('PASS: direct model selection; request editing and upload; fresh approval digest and consent; responsive desktop/mobile forms; compact operation sidebar; recommendation rename; safe rendering; connection templates; datasets; trial selection, persistence and logs.');
}finally{
 await browser?.close();
 await new Promise(resolve=>server.close(resolve));
}
