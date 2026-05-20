import React, { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import api, { ensureCsrf, setToken } from '../api/axios'

export default function ChangePassword(){
  const navigate = useNavigate()
  const location = useLocation()
  const [form, setForm] = useState({ old_password: '', new_password: '', new_password_confirm: '' })
  const [status, setStatus] = useState(null)

  useEffect(() => { ensureCsrf() }, [])

  const onChange = (e) => setForm((p) => ({ ...p, [e.target.name]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    setStatus('pending')
    try{
      await ensureCsrf()
      await api.post('/api/auth/password/change/', form)
      setStatus('saved')
      const params = new URLSearchParams(location.search)
      const next = params.get('next') || '/dashboard/branch'
      navigate(next, { replace: true })
    }catch(err){
      setStatus('error:' + (err?.response?.data?.detail || 'Failed to update password'))
    }
  }

  return (
    <div className="auth-page">
      <div className="container py-4 py-lg-5">
        <div className="row justify-content-center">
          <div className="col-lg-6">
            <div className="auth-card p-4 p-lg-5">
              <h1 className="h4 mb-2 text-olive">Update your password</h1>
              <p className="text-muted small mb-4">
                For security, please change the default password before continuing.
              </p>

              <form onSubmit={submit}>
                <div className="row g-3">
                  <div className="col-12">
                    <label className="form-label">Current password</label>
                    <input className="form-control" name="old_password" type="password" value={form.old_password} onChange={onChange} required />
                    <div className="form-text">If this is your first login, your current password is <strong>Password123</strong>.</div>
                  </div>
                  <div className="col-12">
                    <label className="form-label">New password</label>
                    <input className="form-control" name="new_password" type="password" value={form.new_password} onChange={onChange} required minLength={8} />
                  </div>
                  <div className="col-12">
                    <label className="form-label">Confirm new password</label>
                    <input className="form-control" name="new_password_confirm" type="password" value={form.new_password_confirm} onChange={onChange} required minLength={8} />
                  </div>
                </div>

                <div className="d-grid mt-4">
                  <button className="btn btn-olive" disabled={status === 'pending'}>
                    {status === 'pending' ? 'Updating…' : 'Update password'}
                  </button>
                </div>
              </form>

              {status?.startsWith('error') && <div className="alert alert-danger mt-3">{status.split(':')[1]}</div>}
              {status === 'saved' && <div className="alert alert-success mt-3">Password updated successfully.</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

