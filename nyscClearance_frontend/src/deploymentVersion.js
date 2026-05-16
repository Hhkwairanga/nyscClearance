const BUILD_TIMESTAMP = typeof __APP_BUILD_TIMESTAMP__ !== 'undefined'
  ? __APP_BUILD_TIMESTAMP__
  : 'development'

export const APP_VERSION = String(import.meta.env.VITE_APP_VERSION || BUILD_TIMESTAMP || 'development').trim()

const VERSION_KEY = 'app_version'
const RELOAD_GUARD_KEY = 'app_version_reload_guard'

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
  try{
    previousVersion = localStorage.getItem(VERSION_KEY) || ''
    guardedVersion = sessionStorage.getItem(RELOAD_GUARD_KEY) || ''
  }catch(e){
    return false
  }

  if(previousVersion === APP_VERSION){
    try{ sessionStorage.removeItem(RELOAD_GUARD_KEY) }catch(e){}
    return false
  }

  if(guardedVersion === APP_VERSION){
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
    canReload = localStorage.getItem(VERSION_KEY) === APP_VERSION
  }catch(e){
    canReload = false
  }

  if(canReload){
    window.location.reload()
    return true
  }

  return false
}
