// Centralized API base + URL helpers
// Use `VITE_API_BASE` for production (e.g., https://api.yourdomain.com)
// Leave empty in local dev to rely on Vite proxy
export const API_BASE = (import.meta.env?.VITE_API_BASE || '').replace(/\/$/, '')

export function apiUrl(path = '/'){
  const p = String(path || '/')
  return API_BASE ? `${API_BASE}${p.startsWith('/') ? p : `/${p}`}` : p
}

// For <a href> usage where a full absolute URL is preferred in production
export const apiHref = apiUrl

