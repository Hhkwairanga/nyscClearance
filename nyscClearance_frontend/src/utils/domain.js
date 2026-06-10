const ROOT_DOMAIN = (import.meta.env?.VITE_ROOT_DOMAIN || 'nyscclearance.com').replace(/^\.+|\.+$/g, '')
const API_SUBDOMAIN = import.meta.env?.VITE_API_SUBDOMAIN || 'api'
const LEGACY_FRONTEND_HOST = (import.meta.env?.VITE_LEGACY_FRONTEND_HOST || 'nyscclearance.sahabs.tech').toLowerCase()
const LEGACY_API_BASE = (import.meta.env?.VITE_LEGACY_API_BASE || 'https://api.sahabs.tech').replace(/\/$/, '')

function currentHost(){
  try{ return window.location.hostname.toLowerCase() }catch{ return '' }
}

export function isProductionDomain(host = currentHost()){
  return host === ROOT_DOMAIN || host.endsWith(`.${ROOT_DOMAIN}`)
}

export function isLegacyProductionDomain(host = currentHost()){
  return Boolean(LEGACY_FRONTEND_HOST) && host === LEGACY_FRONTEND_HOST
}

export function isReservedSubdomain(subdomain){
  return ['www', API_SUBDOMAIN, 'api', 'admin', 'home'].includes(String(subdomain || '').toLowerCase())
}

export function currentEnterpriseSubdomain(host = currentHost()){
  if(!isProductionDomain(host) || host === ROOT_DOMAIN) return ''
  const suffix = `.${ROOT_DOMAIN}`
  const subdomain = host.endsWith(suffix) ? host.slice(0, -suffix.length) : ''
  if(!subdomain || subdomain.includes('.') || isReservedSubdomain(subdomain)) return ''
  return subdomain
}

export function enterpriseUrl(subdomain, path = '/'){
  const clean = String(subdomain || '').trim().toLowerCase()
  if(!clean) return ''
  const normalizedPath = String(path || '/').startsWith('/') ? path : `/${path}`
  return `https://${clean}.${ROOT_DOMAIN}${normalizedPath}`
}

export function defaultApiBase(){
  if(isLegacyProductionDomain()) return LEGACY_API_BASE
  if(!isProductionDomain()) return ''
  return `https://${API_SUBDOMAIN}.${ROOT_DOMAIN}`
}

export const APP_ROOT_DOMAIN = ROOT_DOMAIN
