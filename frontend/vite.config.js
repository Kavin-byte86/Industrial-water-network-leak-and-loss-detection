import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  // Two separate pages from one build: the operator dashboard (index.html) and
  // the leak-injection test bench (testbench.html). They share the API client
  // and stylesheet but are otherwise independent apps.
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        testbench: resolve(__dirname, 'testbench.html'),
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
