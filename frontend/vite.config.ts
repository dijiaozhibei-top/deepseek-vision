import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../app/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/v1': 'http://localhost:8000',
      '/status': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
