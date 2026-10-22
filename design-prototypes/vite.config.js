import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    watch: {
      // File notifications on a Windows-mounted WSL checkout can be unreliable.
      usePolling: true,
      interval: 300,
      ignored: ['**/screenshots/**'],
    },
  },
});
