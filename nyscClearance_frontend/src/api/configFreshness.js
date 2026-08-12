import api from './axios'

export const CONFIG_REFRESH_MS = 10000

export async function fetchAdminConfigVersion(){
  try{
    const res = await api.get(`/api/auth/config/version/?_=${Date.now()}`, { showNetworkErrorPage: false })
    return String(res?.data?.version || '')
  }catch(e){
    return ''
  }
}
