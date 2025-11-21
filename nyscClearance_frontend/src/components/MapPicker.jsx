import React, { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import pinUrl from '../assets/location-pin.svg'

export default function MapPicker({ value, onChange, height=280, zoom=6 }){
  const mapRef = useRef(null)
  const mapEl = useRef(null)

  useEffect(() => {
    if(!mapEl.current) return
    if(!mapRef.current){
      const locIcon = L.icon({
        iconUrl: pinUrl,
        iconSize: [32, 48],
        iconAnchor: [16, 48],
        popupAnchor: [0, -48],
      })
      const center = value?.lat && value?.lng ? [value.lat, value.lng] : [9.0820, 8.6753] // Nigeria center
      const map = L.map(mapEl.current).setView(center, zoom)
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
      }).addTo(map)
      const marker = L.marker(center, { draggable: true, icon: locIcon }).addTo(map)
      marker.on('dragend', () => {
        const pos = marker.getLatLng()
        onChange && onChange({ lat: pos.lat, lng: pos.lng })
      })
      map.on('click', (e) => {
        marker.setLatLng(e.latlng)
        onChange && onChange({ lat: e.latlng.lat, lng: e.latlng.lng })
      })

      // Add a "Use my location" control
      const locateControl = L.control({ position: 'topleft' })
      locateControl.onAdd = () => {
        const container = L.DomUtil.create('div', 'leaflet-bar')
        const btn = L.DomUtil.create('a', '', container)
        btn.href = '#'
        btn.title = 'Use my current location'
        btn.innerHTML = '📍'
        btn.style.width = '34px'
        btn.style.height = '34px'
        btn.style.lineHeight = '34px'
        btn.style.textAlign = 'center'
        btn.style.fontSize = '18px'
        L.DomEvent.on(btn, 'click', (e) => {
          L.DomEvent.stopPropagation(e)
          L.DomEvent.preventDefault(e)
          if(!navigator.geolocation){
            console.warn('Geolocation not supported')
            return
          }
          navigator.geolocation.getCurrentPosition((pos) => {
            const { latitude, longitude } = pos.coords
            const latlng = [latitude, longitude]
            marker.setLatLng(latlng)
            map.setView(latlng, 16)
            onChange && onChange({ lat: latitude, lng: longitude })
          }, (err) => {
            console.warn('Geolocation error:', err?.message || err)
          }, { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 })
        })
        return container
      }
      locateControl.addTo(map)
      mapRef.current = { map, marker }
    } else if(value?.lat && value?.lng){
      mapRef.current.marker.setLatLng([value.lat, value.lng])
      mapRef.current.map.setView([value.lat, value.lng], mapRef.current.map.getZoom())
    }
  }, [value])

  return (
    <div>
      <div ref={mapEl} style={{height}} />
      <div className="form-text mt-1">
        Click the 📍 button on the map and grant browser permission to use your current location.
      </div>
    </div>
  )
}
