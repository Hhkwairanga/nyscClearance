import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Crosshair } from 'lucide-react'
import api from '../api/axios'
import { loadGoogleMaps } from '../utils/googleMaps'

const defaultCenter = { lat: 9.082, lng: 8.6753 }
const minSearchLength = 4

function normalizePosition(lat, lng){
  const la = Number(lat)
  const lo = Number(lng)
  if(Number.isFinite(la) && Number.isFinite(lo) && (la !== 0 || lo !== 0)) return { lat: la, lng: lo }
  return null
}

export default function GeofencePicker({
  address,
  onAddressChange,
  lat,
  lng,
  onLatLngChange,
  height = 260,
}) {
  const markerPos = useMemo(() => normalizePosition(lat, lng), [lat, lng])
  const mapEl = useRef(null)
  const mapRef = useRef(null)
  const markerRef = useRef(null)
  const lastRequestRef = useRef(0)
  const searchCacheRef = useRef(new Map())
  const reverseTimeoutRef = useRef(null)

  const [open, setOpen] = useState(false)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [reverseLoading, setReverseLoading] = useState(false)
  const [error, setError] = useState(null)
  const [mapError, setMapError] = useState('')
  const [locating, setLocating] = useState(false)

  const center = markerPos || defaultCenter
  const searchQuery = String(address || '').trim()

  async function reverseGeocode(next){
    setReverseLoading(true)
    try{
      const res = await api.get('/api/auth/maps/reverse/', { params: next })
      const label = String(res.data?.address || '').trim()
      if(label) onAddressChange?.(label)
    }catch(e){
      setError(e?.response?.data?.detail || e?.message || 'Could not resolve address for this location.')
    }finally{
      setReverseLoading(false)
    }
  }

  function queueReverseGeocode(next){
    window.clearTimeout(reverseTimeoutRef.current)
    reverseTimeoutRef.current = window.setTimeout(() => reverseGeocode(next), 700)
  }

  function pickPosition(next, shouldReverse = true){
    const clean = { lat: Number(next.lat).toFixed(6), lng: Number(next.lng).toFixed(6) }
    onLatLngChange?.(clean)
    markerRef.current?.setPosition({ lat: Number(clean.lat), lng: Number(clean.lng) })
    mapRef.current?.setCenter({ lat: Number(clean.lat), lng: Number(clean.lng) })
    if(shouldReverse) queueReverseGeocode(clean)
  }

  async function searchAddress(){
    const q = searchQuery
    setOpen(true)
    setError(null)

    if (q.length < minSearchLength) {
      setResults([])
      setError(`Enter at least ${minSearchLength} characters before searching.`)
      return
    }

    const cacheKey = q.toLowerCase()
    if(searchCacheRef.current.has(cacheKey)){
      setResults(searchCacheRef.current.get(cacheKey))
      return
    }

    const reqId = Date.now()
    lastRequestRef.current = reqId
    setLoading(true)

    try {
      const res = await api.get('/api/auth/maps/geocode/', { params: { q } })
      if (lastRequestRef.current !== reqId) return
      const nextResults = Array.isArray(res.data?.results) ? res.data.results : []
      searchCacheRef.current.set(cacheKey, nextResults)
      setResults(nextResults)
      if(nextResults.length === 0) setError('No matching address found. Try adding city, state, or landmark.')
    } catch (e) {
      if (lastRequestRef.current !== reqId) return
      setResults([])
      setError(e?.response?.data?.detail || e?.message || 'Failed to search address.')
    } finally {
      if (lastRequestRef.current === reqId) setLoading(false)
    }
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try{
        const maps = await loadGoogleMaps()
        if(cancelled || !mapEl.current || mapRef.current) return
        const map = new maps.Map(mapEl.current, {
          center,
          zoom: markerPos ? 14 : 6,
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
          if(pos) pickPosition({ lat: pos.lat(), lng: pos.lng() })
        })
        map.addListener('click', (event) => {
          pickPosition({ lat: event.latLng.lat(), lng: event.latLng.lng() })
        })
        mapRef.current = map
        markerRef.current = marker
      }catch(e){
        if(!cancelled) setMapError(e?.message || 'Google Maps could not be loaded.')
      }
    })()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if(!markerPos || !mapRef.current || !markerRef.current) return
    markerRef.current.setPosition(markerPos)
    mapRef.current.setCenter(markerPos)
    if(mapRef.current.getZoom() < 14) mapRef.current.setZoom(14)
  }, [markerPos?.lat, markerPos?.lng])

  useEffect(() => {
    return () => window.clearTimeout(reverseTimeoutRef.current)
  }, [])

  function useCurrentLocation(){
    if(!navigator.geolocation) return
    setLocating(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const next = { lat: pos.coords.latitude, lng: pos.coords.longitude }
        pickPosition(next)
        mapRef.current?.setZoom(16)
        setLocating(false)
      },
      () => setLocating(false),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    )
  }

  return (
    <div className="geofence-picker">
      <label className="form-label">Branch Address</label>
      <div className="position-relative">
        <div className="geofence-search-row">
          <input
            className="form-control"
            value={address}
            onChange={(e) => {
              onAddressChange?.(e.target.value)
              setResults([])
              setError(null)
              setOpen(false)
            }}
            onFocus={() => {
              if(results.length > 0 || error) setOpen(true)
            }}
            onKeyDown={(e) => {
              if(e.key === 'Enter'){
                e.preventDefault()
                searchAddress()
              }
            }}
            placeholder="Enter branch address, city, or landmark"
          />
          <button type="button" className="btn btn-primary" onClick={searchAddress} disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {open && (loading || results.length > 0 || error) && (
          <div className="geofence-suggest">
            {loading && <div className="geofence-suggest-item text-muted">Searching...</div>}
            {error && <div className="geofence-suggest-item text-danger">{error}</div>}
            {!loading && !error && results.map((r) => (
              <button
                type="button"
                key={r.id}
                className="geofence-suggest-item"
                onClick={() => {
                  onAddressChange?.(r.label)
                  pickPosition({ lat: r.lat, lng: r.lng }, false)
                  setOpen(false)
                }}
              >
                {r.label}
              </button>
            ))}
            {!loading && !error && results.length === 0 && searchQuery.length >= minSearchLength && (
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

      <div className="google-map-shell mt-2" style={{ height }}>
        {mapError ? (
          <div className="google-map-empty">{mapError}</div>
        ) : (
          <>
            <div ref={mapEl} className="google-map-canvas" />
            <button className="google-map-locate" type="button" onClick={useCurrentLocation} disabled={locating} title="Use my current location">
              <Crosshair size={17} />
            </button>
          </>
        )}
      </div>
      <div className="form-text">
        {reverseLoading ? 'Resolving address from selected coordinates...' : 'Tip: search once, pick a result, then drag or click the map only if you need to fine-tune.'}
      </div>
    </div>
  )
}
