import React, { useEffect, useMemo, useRef, useState } from 'react'
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const defaultCenter = { lat: 9.082, lng: 8.6753 } // Nigeria

function Recenter({ center, zoom }) {
  const map = useMap()
  useEffect(() => {
    if (!center?.lat || !center?.lng) return
    map.setView([center.lat, center.lng], zoom)
  }, [center?.lat, center?.lng, zoom, map])
  return null
}

function MapClick({ onPick }) {
  useMapEvents({
    click: (e) => {
      onPick?.({ lat: e.latlng.lat.toFixed(6), lng: e.latlng.lng.toFixed(6) })
    },
  })
  return null
}

function LocateControl({ onPick }) {
  const map = useMap()
  async function reverseGeocode(lat, lng) {
    try {
      const url = new URL('https://nominatim.openstreetmap.org/reverse')
      url.searchParams.set('lat', String(lat))
      url.searchParams.set('lon', String(lng))
      url.searchParams.set('format', 'json')
      url.searchParams.set('addressdetails', '1')
      const res = await fetch(url.toString(), { headers: { Accept: 'application/json' } })
      if (!res.ok) return null
      const data = await res.json()
      return data?.display_name || null
    } catch (e) {
      return null
    }
  }

  useEffect(() => {
    const control = L.control({ position: 'topleft' })
    control.onAdd = () => {
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
        if (!navigator.geolocation) return
        navigator.geolocation.getCurrentPosition(
          async (pos) => {
            const { latitude, longitude } = pos.coords
            const next = { lat: latitude.toFixed(6), lng: longitude.toFixed(6) }
            const addr = await reverseGeocode(latitude, longitude)
            onPick?.({ ...next, address: addr || null })
            map.setView([latitude, longitude], 16)
          },
          () => {},
          { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        )
      })
      return container
    }
    control.addTo(map)
    return () => {
      try {
        control.remove()
      } catch (e) {}
    }
  }, [map, onPick])
  return null
}

function useDebounced(value, delayMs) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs])
  return debounced
}

export default function GeofencePicker({
  address,
  onAddressChange,
  lat,
  lng,
  onLatLngChange,
  height = 260,
}) {
  const markerPos = useMemo(() => {
    const la = Number(lat)
    const lo = Number(lng)
    if (Number.isFinite(la) && Number.isFinite(lo) && (la !== 0 || lo !== 0)) return { lat: la, lng: lo }
    return null
  }, [lat, lng])

  const [open, setOpen] = useState(false)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const lastRequestRef = useRef(0)

  const debouncedAddress = useDebounced(address || '', 450)

  const markerIcon = useMemo(
    () =>
      new L.Icon({
        iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
        iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
        shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
      }),
    []
  )

  useEffect(() => {
    const q = debouncedAddress.trim()
    if (q.length < 3) {
      setResults([])
      setError(null)
      return
    }

    const reqId = Date.now()
    lastRequestRef.current = reqId
    setLoading(true)
    setError(null)

    ;(async () => {
      try {
        const url = new URL('https://nominatim.openstreetmap.org/search')
        url.searchParams.set('q', q)
        url.searchParams.set('format', 'json')
        url.searchParams.set('addressdetails', '1')
        url.searchParams.set('limit', '6')

        const res = await fetch(url.toString(), {
          headers: {
            Accept: 'application/json',
          },
        })
        if (!res.ok) throw new Error('Failed to fetch suggestions')
        const data = await res.json()
        if (lastRequestRef.current !== reqId) return
        setResults(
          (Array.isArray(data) ? data : []).map((r) => ({
            id: `${r.place_id}`,
            label: r.display_name,
            lat: Number(r.lat),
            lng: Number(r.lon),
          }))
        )
      } catch (e) {
        if (lastRequestRef.current !== reqId) return
        setError(e?.message || 'Failed to fetch suggestions')
      } finally {
        if (lastRequestRef.current === reqId) setLoading(false)
      }
    })()
  }, [debouncedAddress])

  const center = markerPos || defaultCenter

  return (
    <div className="geofence-picker">
      <label className="form-label">Branch Address</label>
      <div className="position-relative">
        <input
          className="form-control"
          value={address}
          onChange={(e) => {
            onAddressChange?.(e.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          placeholder="Start typing address…"
        />

        {open && (loading || results.length > 0 || error) && (
          <div className="geofence-suggest">
            {loading && <div className="geofence-suggest-item text-muted">Searching…</div>}
            {error && <div className="geofence-suggest-item text-danger">{error}</div>}
            {!loading && !error && results.map((r) => (
              <button
                type="button"
                key={r.id}
                className="geofence-suggest-item"
                onClick={() => {
                  onAddressChange?.(r.label)
                  onLatLngChange?.({ lat: r.lat, lng: r.lng })
                  setOpen(false)
                }}
              >
                {r.label}
              </button>
            ))}
            {!loading && !error && results.length === 0 && debouncedAddress.trim().length >= 3 && (
              <div className="geofence-suggest-item text-muted">No matches.</div>
            )}
          </div>
        )}
      </div>

      <div className="row g-2 mt-2">
        <div className="col-md-6">
          <label className="form-label">Latitude</label>
          <input
            className="form-control"
            value={lat}
            onChange={(e) => onLatLngChange?.({ lat: e.target.value, lng })}
            placeholder="Auto-filled"
          />
        </div>
        <div className="col-md-6">
          <label className="form-label">Longitude</label>
          <input
            className="form-control"
            value={lng}
            onChange={(e) => onLatLngChange?.({ lat, lng: e.target.value })}
            placeholder="Auto-filled"
          />
        </div>
      </div>

      <div className="geofence-map mt-2" style={{ height }}>
        <MapContainer center={[center.lat, center.lng]} zoom={markerPos ? 14 : 6} style={{ height: '100%', width: '100%' }}>
          <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <Recenter center={center} zoom={markerPos ? 14 : 6} />
          <LocateControl
            onPick={({ lat, lng, address: addr }) => {
              onLatLngChange?.({ lat, lng })
              if (addr) onAddressChange?.(addr)
            }}
          />
          <MapClick onPick={onLatLngChange} />
          <Marker
            position={[center.lat, center.lng]}
            draggable
            icon={markerIcon}
            eventHandlers={{
              dragend: (e) => {
                const pos = e.target.getLatLng()
                onLatLngChange?.({ lat: pos.lat.toFixed(6), lng: pos.lng.toFixed(6) })
              },
            }}
          />
        </MapContainer>
      </div>
      <div className="form-text">Tip: pick a suggestion, click the map, or use the 📍 button.</div>
    </div>
  )
}
