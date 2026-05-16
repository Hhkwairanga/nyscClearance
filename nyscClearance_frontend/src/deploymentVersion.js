import { apiUrl } from './api/urls'

const BUILD_TIMESTAMP = typeof __APP_BUILD_TIMESTAMP__ !== 'undefined'
  ? __APP_BUILD_TIMESTAMP__
  : 'development'

export const APP_VERSION = String(import.meta.env.VITE_APP_VERSION || BUILD_TIMESTAMP || 'development').trim()

const VERSION_KEY = 'app_version'
const RELOAD_GUARD_KEY = 'app_version_reload_guard'
const LOGOUT_PENDING_KEY = 'app_version_logout_pending'

async function logoutServerSession(){
  let timeout
  try{
    const controller = new AbortController()
    timeout = setTimeout(() => controller.abort(), 4000)
    await fetch(apiUrl('/api/auth/logout/'), {
      method: 'POST',
      credentials: 'include',
      cache: 'no-store',
      signal: controller.signal,
    })
    return true
  }catch(e){
    return false
  }finally{
    if(timeout) clearTimeout(timeout)
  }
}

function expireReadableAuthCookies(){
  try{
    document.cookie = 'csrftoken=; Max-Age=0; path=/'
  }catch(e){}
}

async function unregisterServiceWorkers(){
  try{
    if(!('serviceWorker' in navigator)) return
    const registrations = await navigator.serviceWorker.getRegistrations()
    await Promise.all(registrations.map((registration) => registration.unregister().catch(() => false)))
  }catch(e){}
}

export async function refreshForDeploymentVersion(){
  if(typeof window === 'undefined' || !APP_VERSION) return false

  let previousVersion = ''
  let guardedVersion = ''
  let logoutPendingVersion = ''
  try{
    previousVersion = localStorage.getItem(VERSION_KEY) || ''
    guardedVersion = sessionStorage.getItem(RELOAD_GUARD_KEY) || ''
    logoutPendingVersion = sessionStorage.getItem(LOGOUT_PENDING_KEY) || ''
  }catch(e){
    return false
  }

  if(previousVersion === APP_VERSION){
    if(logoutPendingVersion === APP_VERSION){
      const loggedOut = await logoutServerSession()
      if(loggedOut){
        expireReadableAuthCookies()
        try{ sessionStorage.removeItem(LOGOUT_PENDING_KEY) }catch(e){}
      }
    }
    try{ sessionStorage.removeItem(RELOAD_GUARD_KEY) }catch(e){}
    return false
  }

  if(guardedVersion === APP_VERSION){
    if(logoutPendingVersion === APP_VERSION){
      const loggedOut = await logoutServerSession()
      if(loggedOut){
        expireReadableAuthCookies()
        try{ sessionStorage.removeItem(LOGOUT_PENDING_KEY) }catch(e){}
      }
    }
    try{
      localStorage.setItem(VERSION_KEY, APP_VERSION)
      sessionStorage.removeItem(RELOAD_GUARD_KEY)
    }catch(e){}
    return false
  }

  try{ localStorage.clear() }catch(e){}
  try{ sessionStorage.clear() }catch(e){}
  await unregisterServiceWorkers()

  let canReload = false
  try{
    localStorage.setItem(VERSION_KEY, APP_VERSION)
    sessionStorage.setItem(RELOAD_GUARD_KEY, APP_VERSION)
    sessionStorage.setItem(LOGOUT_PENDING_KEY, APP_VERSION)
    canReload = localStorage.getItem(VERSION_KEY) === APP_VERSION
  }catch(e){
    canReload = false
  }

  const loggedOut = await logoutServerSession()
  if(loggedOut){
    expireReadableAuthCookies()
    try{ sessionStorage.removeItem(LOGOUT_PENDING_KEY) }catch(e){}
  }

  if(canReload){
    window.location.reload()
    return true
  }

  return false
}
