import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

export default defineConfig({
  site: process.env.SITE_URL || undefined,
  output: 'static',
  devToolbar: { enabled: false },
  integrations: [react()],
  vite: {
    // Production prebundling must not overwrite React's development JSX runtime.
    cacheDir: `node_modules/.vite-${process.env.NODE_ENV || 'development'}`,
    server: { watch: { usePolling: true, interval: 400, ignored: ['**/screenshots/**'] } },
  },
});
