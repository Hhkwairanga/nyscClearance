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
export function getToken(){ try{ return localStorage.getItem(TOKEN_KEY) || '' }catch{ return '' } }
export function setToken(t){ try{ if(t){ localStorage.setItem(TOKEN_KEY, t); api.defaults.headers.common['Authorization'] = `Bearer ${t}` } }catch{} }
export function clearToken(){ try{ localStorage.removeItem(TOKEN_KEY); delete api.defaults.headers.common['Authorization'] }catch{} }

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

export default api
