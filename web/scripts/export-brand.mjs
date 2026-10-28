// Export the current Confiron vector identity and social card.
import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {chromium} from '@playwright/test';
const root=new URL('../',import.meta.url),output=new URL('public/logos/',root);
await mkdir(output,{recursive:true});
const [layout,prototype,site,font]=await Promise.all([
 readFile(new URL('src/layouts/Site.astro',root),'utf8'),
 readFile(new URL('src/styles/prototype.css',root),'utf8'),
 readFile(new URL('src/styles/site.css',root),'utf8'),
 readFile(new URL('node_modules/@fontsource/inter/files/inter-latin-600-normal.woff2',root))
]);
const wordmark=layout.match(/<a class="wordmark"[^>]*>[\s\S]*?<\/a>/)?.[0];
if(!wordmark)throw Error('Website wordmark not found');
const icon=await readFile(new URL('public/logos/confiron-icon.svg',root),'utf8');
const embeddedWordmark=wordmark.replace('src="/logos/confiron-icon.svg"',`src="data:image/svg+xml;base64,${Buffer.from(icon).toString('base64')}"`);
const browser=await chromium.launch();
try{
 const page=await browser.newPage({viewport:{width:1000,height:500},deviceScaleFactor:4});
 for(const [name,markup] of [['confiron-logo',embeddedWordmark],['confiron-icon',icon]]){
  await page.setContent(`<style>@font-face{font-family:Inter;src:url(data:font/woff2;base64,${font.toString('base64')}) format('woff2');font-weight:600}${prototype}${site}html,body{margin:0;background:transparent}.export{display:inline-flex;align-items:center;padding:24px}.export>a{color:#edf0ed}.export>svg{width:160px;height:160px}</style><div class="export">${markup}</div>`);
  await page.evaluate(()=>document.fonts.ready);
  await page.locator('.export').screenshot({path:fileURLToPath(new URL(name+'.png',output)),omitBackground:true});
  await page.evaluate(()=>document.body.style.background='#101316');
  await page.locator('.export').screenshot({path:fileURLToPath(new URL(name+'.jpg',output)),type:'jpeg',quality:100});
 }
 const social=await browser.newPage({viewport:{width:1200,height:630},deviceScaleFactor:1});
 await social.setContent(`<style>@font-face{font-family:Inter;src:url(data:font/woff2;base64,${font.toString('base64')})}*{box-sizing:border-box}body{margin:0;padding:70px 80px;background:#101316;color:#edf0ed;font-family:Inter,sans-serif}.brand{display:flex;align-items:center;gap:15px;font-size:36px;color:#a7f3cf}.brand img{width:45px;height:45px}h1{font-size:65px;line-height:1.2;letter-spacing:-2px;margin:80px 0 35px;font-weight:600}em{font-style:normal;color:#a7f3cf}p{font-size:23px;color:#a9b4bf}</style><div class="brand"><img src="data:image/svg+xml;base64,${Buffer.from(icon).toString('base64')}"/>confiron.</div><h1>Configuration is boring.<br/><em>Let Confiron do it for you.</em></h1><p>Discover models · Optimize inference · Benchmark performance</p>`);
 await social.evaluate(()=>document.fonts.ready);
 await social.screenshot({path:fileURLToPath(new URL('public/social.png',root))});
}finally{await browser.close()}
// Keep the current local production artifact in sync without rebuilding JS.
const built=new URL('dist/logos/',root);await mkdir(built,{recursive:true});
for(const name of ['confiron-icon.svg','confiron-logo.png','confiron-logo.jpg','confiron-icon.png','confiron-icon.jpg'])await copyFile(new URL(name,output),new URL(name,built));
console.log('Exported Confiron icon and wordmark: public/logos/confiron-{icon,logo}.{png,jpg}');
