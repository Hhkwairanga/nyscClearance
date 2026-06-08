import api from '../api/axios'

let configPromise = null
let loaderPromise = null

export async function getGoogleMapsConfig(){
  if(!configPromise){
    configPromise = api.get('/api/auth/maps/config/').then((res) => res.data)
  }
  return configPromise
}

export async function loadGoogleMaps(){
  if(window.google?.maps) return window.google.maps
  if(loaderPromise) return loaderPromise

  loaderPromise = (async () => {
    const config = await getGoogleMapsConfig()
    if(!config?.configured || !config?.browser_api_key){
      throw new Error('Google Maps is not configured.')
    }

    const callbackName = `__nyscGoogleMapsReady_${Date.now()}`
    const params = new URLSearchParams({
      key: config.browser_api_key,
      callback: callbackName,
      v: 'weekly',
    })
    if(config.map_id) params.set('map_ids', config.map_id)

    await new Promise((resolve, reject) => {
      window[callbackName] = () => {
        delete window[callbackName]
        resolve()
      }

      const script = document.createElement('script')
      script.src = `https://maps.googleapis.com/maps/api/js?${params.toString()}`
      script.async = true
      script.defer = true
      script.onerror = () => {
        delete window[callbackName]
        reject(new Error('Failed to load Google Maps.'))
      }
      document.head.appendChild(script)
    })

    return window.google.maps
  })()

  return loaderPromise
}

