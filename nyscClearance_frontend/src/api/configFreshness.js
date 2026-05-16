import api from './axios'

export const CONFIG_REFRESH_MS = 10000

export async function fetchAdminConfigVersion(){
  const res = await api.get(`/api/auth/config/version/?_=${Date.now()}`, {
    headers: {
      'Cache-Control': 'no-cache',
      Pragma: 'no-cache',
    },
  })
  return String(res?.data?.version || '')
}
