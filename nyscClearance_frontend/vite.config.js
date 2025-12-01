import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = (env.VITE_API_BASE || 'http://localhost:8000').replace(/\/$/, '')
  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': { target: apiTarget, changeOrigin: true },
        '/static': { target: apiTarget, changeOrigin: true },
        '/media': { target: apiTarget, changeOrigin: true },
      },
    },
    preview: {
      port: 5173,
    },
  }
})
