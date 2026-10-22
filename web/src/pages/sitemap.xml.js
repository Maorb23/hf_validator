export function GET({ site }) {
  const routes = ['/', '/technology/', '/architecture/', '/benchmarks/', '/methodology/', '/evaluation-guide/', '/demo/', '/demo/search/', '/demo/pareto/', '/demo/journey/', '/demo/studies/'];
  const escape = s => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
  const urls = site ? routes.map(path => `<url><loc>${escape(new URL(path, site).href)}</loc></url>`).join('') : '';
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls}</urlset>`, { headers: { 'Content-Type': 'application/xml' } });
}
