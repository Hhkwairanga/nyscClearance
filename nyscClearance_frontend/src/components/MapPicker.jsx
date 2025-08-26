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
      mapRef.current = { map, marker }
    } else if(value?.lat && value?.lng){
      mapRef.current.marker.setLatLng([value.lat, value.lng])
      mapRef.current.map.setView([value.lat, value.lng], mapRef.current.map.getZoom())
    }
  }, [value])

  return (
    <div ref={mapEl} style={{height}} />
  )
}
