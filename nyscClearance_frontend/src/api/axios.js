import axios from 'axios'
import { apiUrl, API_BASE } from './urls'

// Configure base URL from env, default to Django dev server
const api = axios.create({
  // Base URL centralized via urls.js (uses VITE_API_BASE in prod, empty in dev for proxy)
  baseURL: API_BASE || '',
  withCredentials: true,
  xsrfCookieName: 'csrftoken',
  xsrfHeaderName: 'X-CSRFToken',
})

// Simple token storage
const TOKEN_KEY = 'nysc.token'
const REMEMBER_KEY = 'nysc.remember'

function readToken(storage){
  try{ return storage.getItem(TOKEN_KEY) || '' }catch{ return '' }
}

export function getToken(){
  return readToken(localStorage) || readToken(sessionStorage)
}

export function setToken(t, remember = true){
  try{
    localStorage.removeItem(TOKEN_KEY)
    sessionStorage.removeItem(TOKEN_KEY)
    localStorage.setItem(REMEMBER_KEY, remember ? '1' : '0')
    if(t){
      const target = remember ? localStorage : sessionStorage
      target.setItem(TOKEN_KEY, t)
      api.defaults.headers.common['Authorization'] = `Bearer ${t}`
    }
  }catch{
    try{
      if(t){
        sessionStorage.setItem(TOKEN_KEY, t)
        api.defaults.headers.common['Authorization'] = `Bearer ${t}`
      }
    }catch{}
  }
}

export function clearToken(){
  try{ localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(REMEMBER_KEY) }catch{}
  try{ sessionStorage.removeItem(TOKEN_KEY) }catch{}
  try{ delete api.defaults.headers.common['Authorization'] }catch{}
}

// Initialize Authorization header from any existing token
{ const t = getToken(); if(t){ api.defaults.headers.common['Authorization'] = `Bearer ${t}` } }

// Ensure CSRF cookie using fetch to avoid axios interceptor loops
export async function ensureCsrf(){
  try{
    const url = apiUrl('/api/auth/csrf/')
    await fetch(url, { credentials: 'include' })
  }catch(e){ /* ignore */ }
}

// Interceptor: guarantee CSRF on unsafe methods and set header explicitly
api.interceptors.request.use(async (config) => {
  const method = (config.method || 'get').toLowerCase()
  const hasBearer = !!getToken()
  if(['post','put','patch','delete'].includes(method) && !hasBearer){
    await ensureCsrf()
    try{
      const m = document.cookie.match(/(?:^|; )csrftoken=([^;]+)/)
      if(m){
        config.headers = config.headers || {}
        config.headers['X-CSRFToken'] = decodeURIComponent(m[1])
      }
    }catch(e){ /* ignore */ }
  }
  return config
})

function isStaleAuthError(error){
  const status = error?.response?.status
  if(status === 401) return true
  if(status !== 403) return false
  const detail = String(error?.response?.data?.detail || '').toLowerCase()
  return (
    detail.includes('token') ||
    detail.includes('credential') ||
    detail.includes('authentication') ||
    detail.includes('not authenticated') ||
    detail.includes('csrf')
  )
}

function isPublicAuthRoute(){
  try{
    return ['/login', '/signup', '/forgot-password', '/reset-password', '/verify-success'].some((path) => window.location.pathname.startsWith(path))
  }catch(e){
    return false
  }
}

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if(isStaleAuthError(error) && !isPublicAuthRoute()){
      clearToken()
      try{
        const next = `${window.location.pathname}${window.location.search || ''}`
        const loginUrl = next && next !== '/' ? `/login?next=${encodeURIComponent(next)}` : '/login'
        window.location.assign(loginUrl)
      }catch(e){}
    }
    return Promise.reject(error)
  }
)

export default api
