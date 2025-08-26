import axios from 'axios'

// Configure base URL from env, default to Django dev server
const api = axios.create({
  // Prefer same-origin + Vite proxy by default; override with VITE_API_BASE if needed
  baseURL: (import.meta.env.VITE_API_BASE || ''),
  withCredentials: true,
  xsrfCookieName: 'csrftoken',
  xsrfHeaderName: 'X-CSRFToken',
})

// Ensure CSRF cookie using fetch to avoid axios interceptor loops
export async function ensureCsrf(){
  try{
    const url = (api.defaults.baseURL || '').replace(/\/$/, '') + '/api/auth/csrf/'
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
