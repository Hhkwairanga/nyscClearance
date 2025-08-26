import axios from 'axios'

const api = axios.create({
  withCredentials: true,
  xsrfCookieName: 'csrftoken',
  xsrfHeaderName: 'X-CSRFToken',
})

export async function ensureCsrf(){
  try{ await api.get('/api/auth/csrf/') }catch(e){ /* ignore */ }
}

export default api
