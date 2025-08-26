import React, { useEffect, useState } from 'react'
import api, { ensureCsrf } from '../api/axios'
import MapPicker from '../components/MapPicker'
import axios from 'axios'

export default function Signup(){
  const [form, setForm] = useState({
    email: '', name: '', address: '', number_of_corpers: '', password: '', password_confirm: '',
    location_lat: '', location_lng: ''
  })
  const [status, setStatus] = useState(null)

  useEffect(() => { ensureCsrf() }, [])

  const onChange = (e) => {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: value }))
  }

  const submit = async (e) => {
    e.preventDefault()
    setStatus('pending')
    try {
      await ensureCsrf()
      await api.post('/api/auth/register/', {
        ...form,
        number_of_corpers: parseInt(form.number_of_corpers || '0', 10)
      })
      setStatus('success')
    } catch(err){
      const msg = err?.response?.data?.detail
        || Object.values(err?.response?.data || {})?.[0]?.[0]
        || err.message
      setStatus(`error:${msg}`)
    }
  }

  return (
    <div className="row justify-content-center">
      <div className="col-md-8 col-lg-6">
        <div className="card shadow-sm">
          <div className="card-body p-4">
            <h1 className="h4 mb-3 text-olive">Organization Sign Up</h1>
            <form onSubmit={submit}>
              <div className="mb-3">
                <label className="form-label">Email</label>
                <input name="email" type="email" className="form-control" value={form.email} onChange={onChange} required />
              </div>
              <div className="mb-3">
                <label className="form-label">Organization Name</label>
                <input name="name" className="form-control" value={form.name} onChange={onChange} required />
              </div>
              <div className="mb-3">
                <label className="form-label">Address</label>
                <textarea name="address" className="form-control" value={form.address} onChange={onChange} rows="3" />
              </div>
              <div className="mb-3">
                <label className="form-label">Organization Location</label>
                <div className="row g-2">
                  <div className="col-md-4">
                    <input className="form-control" name="location_lat" placeholder="Latitude" value={form.location_lat} onChange={onChange} />
                  </div>
                  <div className="col-md-4">
                    <input className="form-control" name="location_lng" placeholder="Longitude" value={form.location_lng} onChange={onChange} />
                  </div>
                </div>
                <div className="mt-2">
                  <MapPicker
                    value={(form.location_lat && form.location_lng) ? { lat: parseFloat(form.location_lat), lng: parseFloat(form.location_lng)} : null}
                    onChange={(pos) => setForm(prev => ({...prev, location_lat: pos.lat.toFixed(6), location_lng: pos.lng.toFixed(6)}))}
                  />
                </div>
              </div>
              <div className="mb-3">
                <label className="form-label">Number of Corpers</label>
                <input name="number_of_corpers" type="number" min="0" className="form-control" value={form.number_of_corpers} onChange={onChange} />
              </div>
              <div className="mb-3">
                <label className="form-label">Password</label>
                <input name="password" type="password" className="form-control" value={form.password} onChange={onChange} required minLength={8} />
              </div>
              <div className="mb-3">
                <label className="form-label">Confirm Password</label>
                <input name="password_confirm" type="password" className="form-control" value={form.password_confirm} onChange={onChange} required minLength={8} />
                {form.password && form.password_confirm && form.password !== form.password_confirm && (
                  <div className="form-text text-danger">Passwords do not match</div>
                )}
              </div>
              <div className="d-grid">
                <button className="btn btn-olive" disabled={status==='pending' || (form.password !== form.password_confirm)}>Create account</button>
              </div>
            </form>
            {status==='success' && (
              <div className="alert alert-success mt-3">Registration successful. Check your email to verify.</div>
            )}
            {status?.startsWith('error') && (
              <div className="alert alert-danger mt-3">{status.split(':')[1]}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
