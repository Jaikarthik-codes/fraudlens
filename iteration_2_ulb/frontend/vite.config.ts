import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development, forward every API route to the local backend (port 5000)
const backend = { target: 'http://127.0.0.1:5000', changeOrigin: true }

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': backend,
      '/compare': backend,
      '/health': backend,
    },
  },
})
