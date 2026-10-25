import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {fileURLToPath} from 'node:url';
import {generateKeyPairSync} from 'node:crypto';

const origin=process.env.INFIRO_URL||'http://127.0.0.1:8000';
const root=fileURLToPath(new URL('../../',import.meta.url));
const python=process.env.INFIRO_PYTHON||'/tmp/infiro-platform-env/bin/python';
const username='browser-'+Date.now(),password='Local-browser-test-'+crypto.randomUUID();
const shots=fileURLToPath(new URL('../screenshots/',import.meta.url));
await mkdir(shots,{recursive:true});
const browser=await chromium.launch({headless:true});
const results=[];
let ownerStudy,ownerPrepared;
async function seed(){const {stdout}=await promisify(execFile)(python,['web/manage.py','platform_demo','--username',username],{cwd:root,env:process.env});const match=stdout.match(/\/app\/approval\/([a-f0-9-]+)/);assert.ok(match,stdout);return match[1]}
async function api(page,path,body){return page.evaluate(async({path,body})=>{const csrf=document.cookie.split('; ').find(s=>s.startsWith('csrftoken='))?.split('=')[1];const response=await fetch('/app/api/'+path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-CSRFToken':csrf||''},body:body?JSON.stringify(body):undefined});return {status:response.status,data:await response.json()}},{path,body})}
try{
 for(const [device,viewport] of Object.entries({desktop:{width:1440,height:1000},mobile:{width:390,height:844}})){
  const context=await browser.newContext({viewport,reducedMotion:'reduce',isMobile:device==='mobile',hasTouch:device==='mobile'}),page=await context.newPage(),errors=[],consoleErrors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')consoleErrors.push(m.text())});
  await page.goto(origin+'/app/');assert.ok(page.url().endsWith('/app/login'));
  if(device==='desktop'){await page.goto(origin+'/app/register');await page.getByLabel('Username',{exact:true}).fill(username);await page.getByLabel('Password',{exact:true}).fill(password);await page.getByRole('button',{name:'Create workspace'}).click()}
  else{await page.getByLabel('Username',{exact:true}).fill(username);await page.getByLabel('Password',{exact:true}).fill(password);await page.getByRole('button',{name:'Sign in',exact:false}).click()}
  await page.getByRole('heading',{name:'Your inference workspace',exact:true}).waitFor();
  const screenshot=async(name)=>{await page.evaluate(()=>document.fonts.ready);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`${device}/${name}: horizontal overflow`);await page.screenshot({path:`${shots}/platform-${name}-${device}.png`,fullPage:true});results.push({device,screen:name,noOverflow:true})};
  await screenshot('overview');
  if(device==='mobile'){await page.getByRole('button',{name:'Toggle workspace navigation'}).click();await page.getByRole('navigation',{name:'Workspace navigation'}).getByRole('link',{name:'Connections',exact:true}).click()}
  else await page.getByRole('navigation',{name:'Workspace navigation'}).getByRole('link',{name:'Connections',exact:true}).click();
  await page.getByRole('heading',{name:'Connections',exact:true}).waitFor();
  await page.getByRole('button',{name:'Add Nebius jobs',exact:false}).click();
  assert.equal(await page.getByLabel('Authentication',{exact:true}).inputValue(),'service_account');
  await page.getByLabel('Connection name',{exact:true}).fill('Service account fixture '+device);
  await page.getByLabel('Nebius project ID',{exact:true}).fill('project-test');
  await page.getByLabel('Service account ID',{exact:true}).fill('serviceaccount-test');
  await page.getByLabel('Public key ID',{exact:true}).fill('publickey-test');
  const {privateKey}=generateKeyPairSync('rsa',{modulusLength:2048,privateKeyEncoding:{type:'pkcs8',format:'pem'},publicKeyEncoding:{type:'spki',format:'pem'}});
  await screenshot('nebius-service-account');
  await page.getByLabel('Private key (PEM or base64 PEM)',{exact:true}).fill(Buffer.from(privateKey).toString('base64'));
  await page.getByRole('button',{name:'Save encrypted connection'}).click();
  await page.getByRole('status').filter({hasText:'Connection saved encrypted'}).waitFor();
  const saSnapshot=await api(page,'bootstrap');
  assert.ok(!JSON.stringify(saSnapshot.data).includes(privateKey));
  assert.ok(!JSON.stringify(saSnapshot.data).includes(Buffer.from(privateKey).toString('base64')));
  const saConnection=saSnapshot.data.connections.find(c=>c.name==='Service account fixture '+device);
  assert.deepEqual(saConnection.credentials,{private_key:'••••••••'});
  const saDelete=await page.evaluate(async id=>{const csrf=document.cookie.split('; ').find(s=>s.startsWith('csrftoken='))?.split('=')[1];return (await fetch('/app/api/connections/'+id,{method:'DELETE',headers:{'X-CSRFToken':csrf}})).status},saConnection.id);
  assert.equal(saDelete,200);
  await page.reload();await page.getByRole('heading',{name:'Connections',exact:true}).waitFor();
  await page.getByRole('button',{name:'Add connection',exact:true}).click();await page.getByLabel('Provider',{exact:true}).selectOption('huggingface');
  await page.getByLabel('Connection name',{exact:true}).fill('Browser test token — '+device);await page.getByLabel('HF read token',{exact:true}).fill('hf-local-nonprovider-fixture-'+device);
  await page.getByLabel('SecretStash secret or version ID (required only for live workers)',{exact:true}).fill('mbsec-browser123');
  await page.getByRole('button',{name:'Save encrypted connection'}).click();await page.getByRole('status').filter({hasText:'Connection saved encrypted'}).waitFor();
  const snapshot=await api(page,'bootstrap');assert.equal(snapshot.status,200);assert.ok(!JSON.stringify(snapshot.data).includes('hf-local-nonprovider-fixture-'));
  const connection=snapshot.data.connections.find(c=>c.name==='Browser test token — '+device);
  assert.equal(connection.metadata.secret_version_id,'mbsec-browser123');
  await screenshot('connections');
  // Delete the deliberately invalid local token; no provider validation or calls.
  await page.evaluate(async id=>{const csrf=document.cookie.split('; ').find(s=>s.startsWith('csrftoken='))?.split('=')[1];await fetch('/app/api/connections/'+id,{method:'DELETE',headers:{'X-CSRFToken':csrf}})},connection.id);
  const prepared=device==='desktop'?await seed():ownerPrepared;
  if(device==='desktop')ownerPrepared=prepared;
  await page.goto(origin+'/app/approval/'+prepared);
  const namedReview=await api(page,'prepared/'+prepared);
  await page.getByRole('heading',{name:namedReview.data.name,exact:true}).waitFor();
  if(device==='desktop'){
   const approveButton=page.getByRole('button',{name:'Approve dry run',exact:true});assert.ok(await approveButton.isDisabled());await screenshot('approval');
   await page.getByRole('checkbox',{name:/I reviewed the pinned model/}).check();await approveButton.click();
   await page.waitForURL(/\/app\/studies\//);ownerStudy=page.url().split('/').at(-1);
   await page.getByText('Controller dry run completed. No GPU resources were allocated and no inference measurements were produced.',{exact:true}).waitFor({timeout:60000});
  }else{await screenshot('approval');await page.getByRole('button',{name:'Open study',exact:true}).click()}
  await page.getByRole('heading',{name:'Latency & throughput',exact:true}).waitFor();assert.ok(await page.getByRole('heading',{name:'No comparable measurements yet'}).isVisible());await screenshot('study');
  const studyName='Comparison <script>window.nameExecuted=true</script> '+device;
  await page.getByLabel('Study name',{exact:true}).fill('  '+studyName+'  ');
  await page.getByRole('button',{name:'Save name',exact:true}).click();
  await page.getByRole('heading',{name:studyName,exact:true}).waitFor();
  assert.equal(await page.evaluate(()=>window.nameExecuted),undefined);
  const renamed=await api(page,'prepared/'+prepared);
  assert.equal(renamed.data.name,studyName);
  assert.equal(renamed.data.digest,namedReview.data.digest);
  await page.goto(origin+'/app/overview');
  assert.ok(await page.getByText(studyName,{exact:true}).isVisible());
  await page.goto(origin+'/app/studies');
  assert.ok(await page.getByText(studyName,{exact:true}).isVisible());
  await page.goto(origin+'/app/studies/'+ownerStudy);
  await page.getByRole('heading',{name:studyName,exact:true}).waitFor();
  const download=page.getByRole('link',{name:'results.json',exact:false});const response=await page.request.get(origin+await download.getAttribute('href'));assert.equal(response.status(),200);assert.equal((await response.json()).measurement_status,'unmeasured_dry_run');
  if(device==='desktop'){
   const cancelPrepared=await seed();const review=await api(page,'prepared/'+cancelPrepared);
   const first=await api(page,'prepared/'+cancelPrepared,{digest:review.data.digest,approved:true});
   const duplicate=await api(page,'prepared/'+cancelPrepared,{digest:review.data.digest,approved:true});assert.equal(first.data.id,duplicate.data.id,'Repeated approval must return the same durable job');
   const cancel=await api(page,'jobs/'+first.data.id,{});assert.equal(cancel.status,200);
   await page.goto(origin+'/app/studies/'+first.data.id);
   await page.waitForFunction(async id=>(await(await fetch('/app/api/jobs/'+id)).json()).state==='cancelled',first.data.id,{timeout:30000});
   await screenshot('cancelled-study');
  }
  await page.goto(origin+'/app/datasets');await page.getByRole('heading',{name:'Datasets',exact:true}).waitFor();await screenshot('datasets');
  await page.goto(origin+'/app/recommendations');await page.getByRole('heading',{name:'Recommendations',exact:true}).waitFor();await screenshot('recommendations');
  await page.getByRole('button',{name:'Configure workload',exact:true}).first().click();await page.getByRole('heading',{name:'Configure your study',exact:true}).waitFor();await screenshot('configure');
  assert.equal(await page.evaluate(()=>localStorage.length),0,'No credentials or private progress belongs in localStorage');
  assert.equal(await page.evaluate(()=>sessionStorage.length),0,'No credentials or private progress belongs in sessionStorage');
  await page.keyboard.press('Tab');assert.ok(await page.evaluate(()=>['A','BUTTON','INPUT','SELECT','SUMMARY','TEXTAREA'].includes(document.activeElement.tagName)),'Keyboard focus must reach a usable control');
  assert.deepEqual(errors,[],`${device}: runtime errors`);assert.deepEqual(consoleErrors,[],`${device}: browser console errors`);
  await context.close();
 }
 // A second real session cannot access any of the first user's resources.
 const other=await browser.newPage();await other.goto(origin+'/app/register');await other.getByLabel('Username',{exact:true}).fill(username+'-other');await other.getByLabel('Password',{exact:true}).fill(password);await other.getByRole('button',{name:'Create workspace'}).click();await other.getByRole('heading',{name:'Your inference workspace'}).waitFor();
 assert.equal((await api(other,'jobs/'+ownerStudy)).status,404);assert.equal((await api(other,'prepared/'+ownerPrepared)).status,404);assert.equal((await api(other,'bootstrap')).data.jobs.length,0);
 await other.close();
 await writeFile(`${shots}/platform-browser-results.json`,JSON.stringify({results,authentication:true,csrf:true,maskedConnections:true,explicitApproval:true,duplicateSubmission:true,cancellation:true,realWorkerDryRun:true,privateDownload:true,twoUserIsolation:true},null,2));
 console.log(`Verified ${results.length} desktop/mobile screens, real session authentication, masked connections, explicit approval, isolated worker dry run, downloads and two-user API isolation.`);
}finally{await browser.close()}
