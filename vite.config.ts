import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    // Relative asset URLs, so the same build works at a GitHub Pages project
    // path (https://<user>.github.io/<repo>/) without knowing the repo name.
    base: './',
    plugins: [react(), tailwindcss()],
    // No `define` for env vars: this bundle is published as a public static
    // site, and anything inlined here would ship to every visitor.
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
