import React, { useEffect, useRef, useState } from 'react'
import { Crosshair } from 'lucide-react'
import { loadGoogleMaps } from '../utils/googleMaps'

const defaultCenter = { lat: 9.082, lng: 8.6753 }

function normalizePosition(value){
  const lat = Number(value?.lat)
  const lng = Number(value?.lng)
  if(Number.isFinite(lat) && Number.isFinite(lng) && (lat !== 0 || lng !== 0)){
    return { lat, lng }
  }
  return null
}

export default function MapPicker({ value, onChange, height=280, zoom=6 }){
  const mapEl = useRef(null)
  const mapRef = useRef(null)
  const markerRef = useRef(null)
  const [error, setError] = useState('')
  const [locating, setLocating] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try{
        const maps = await loadGoogleMaps()
        if(cancelled || !mapEl.current || mapRef.current) return
        const center = normalizePosition(value) || defaultCenter
        const map = new maps.Map(mapEl.current, {
          center,
          zoom: normalizePosition(value) ? Math.max(zoom, 14) : zoom,
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: false,
        })
        const marker = new maps.Marker({
          map,
          position: center,
          draggable: true,
        })
        marker.addListener('dragend', () => {
          const pos = marker.getPosition()
          if(pos) onChange?.({ lat: pos.lat(), lng: pos.lng() })
        })
        map.addListener('click', (event) => {
          const next = { lat: event.latLng.lat(), lng: event.latLng.lng() }
          marker.setPosition(next)
          onChange?.(next)
        })
        mapRef.current = map
        markerRef.current = marker
      }catch(e){
        if(!cancelled) setError(e?.message || 'Google Maps could not be loaded.')
      }
    })()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    const next = normalizePosition(value)
    if(!next || !mapRef.current || !markerRef.current) return
    markerRef.current.setPosition(next)
    mapRef.current.setCenter(next)
  }, [value?.lat, value?.lng])

  function useCurrentLocation(){
    if(!navigator.geolocation) return
    setLocating(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const next = { lat: pos.coords.latitude, lng: pos.coords.longitude }
        markerRef.current?.setPosition(next)
        mapRef.current?.setCenter(next)
        mapRef.current?.setZoom(16)
        onChange?.(next)
        setLocating(false)
      },
      () => setLocating(false),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    )
  }

  return (
    <div>
      <div className="google-map-shell" style={{height}}>
        {error ? (
          <div className="google-map-empty">{error}</div>
        ) : (
          <>
            <div ref={mapEl} className="google-map-canvas" />
            <button className="google-map-locate" type="button" onClick={useCurrentLocation} disabled={locating} title="Use my current location">
              <Crosshair size={17} />
            </button>
          </>
        )}
      </div>
      <div className="form-text mt-1">
        Use the location button and grant browser permission to select your current location.
      </div>
    </div>
  )
}

