// Centralized API/frontend base + URL helpers
import { defaultApiBase } from '../utils/domain'

// API base: use `VITE_API_BASE` when set. On nyscclearance.com/subdomains, default to api.nyscclearance.com.
export const API_BASE = (import.meta.env?.VITE_API_BASE || defaultApiBase()).replace(/\/$/, '')
// Frontend base: explicit `VITE_FRONTEND_URL`, else fall back to current origin
export const FRONTEND_BASE = (import.meta.env?.VITE_FRONTEND_URL || window.location.origin || '').replace(/\/$/, '')

export function apiUrl(path = '/') {
  const p = String(path || '/')
  return API_BASE ? `${API_BASE}${p.startsWith('/') ? p : `/${p}`}` : p
}

export function feUrl(path = '/') {
  const p = String(path || '/')
  return FRONTEND_BASE ? `${FRONTEND_BASE}${p.startsWith('/') ? p : `/${p}`}` : p
}

// For <a href> usage where a full absolute URL is preferred in production
export const apiHref = apiUrl
export const feHref = feUrl
