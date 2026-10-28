// Export existing website artwork; no generated or redesigned logo assets.
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
const icon=wordmark.match(/<svg[\s\S]*?<\/svg>/)[0].replace('width="32" height="32"','xmlns="http://www.w3.org/2000/svg" width="256" height="256"').replace('fill="currentColor"','fill="#edf0ed"').replace('fill="var(--accent)"','fill="#a7f3cf"');
await writeFile(new URL('infiro-icon.svg',output),icon+'\n');
const browser=await chromium.launch();
try{
 const page=await browser.newPage({viewport:{width:1000,height:500},deviceScaleFactor:4});
 for(const [name,markup] of [['infiro-logo',wordmark],['infiro-icon',icon]]){
  await page.setContent(`<style>@font-face{font-family:Inter;src:url(data:font/woff2;base64,${font.toString('base64')}) format('woff2');font-weight:600}${prototype}${site}html,body{margin:0;background:transparent}.export{display:inline-flex;align-items:center;padding:24px}.export>a{color:#edf0ed}.export>svg{width:160px;height:160px}</style><div class="export">${markup}</div>`);
  await page.evaluate(()=>document.fonts.ready);
  await page.locator('.export').screenshot({path:fileURLToPath(new URL(name+'.png',output)),omitBackground:true});
  await page.evaluate(()=>document.body.style.background='#101316');
  await page.locator('.export').screenshot({path:fileURLToPath(new URL(name+'.jpg',output)),type:'jpeg',quality:100});
 }
}finally{await browser.close()}
// Keep the current local production artifact in sync without rebuilding JS.
const built=new URL('dist/logos/',root);await mkdir(built,{recursive:true});
for(const name of ['infiro-icon.svg','infiro-logo.png','infiro-logo.jpg','infiro-icon.png','infiro-icon.jpg'])await copyFile(new URL(name,output),new URL(name,built));
console.log('Exported white-and-green Infiro icon and wordmark: public/logos/infiro-{icon,logo}.{png,jpg}');
