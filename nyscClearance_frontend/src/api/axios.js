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
  if(['post','put','patch','delete'].includes(method)){
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
