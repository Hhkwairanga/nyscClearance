import React, { useMemo, useState } from 'react'
import api, { ensureCsrf } from '../api/axios'
import { useNavigate } from 'react-router-dom'

export default function VerifySuccess(){
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const token = params.get('token')
  const role = params.get('role')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [status, setStatus] = useState(null)
  const navigate = useNavigate()

  async function setPwd(e){
    e.preventDefault()
    setStatus('pending')
    if(password !== confirm){ setStatus('error:Passwords do not match'); return }
    try{
      await ensureCsrf()
      await api.post('/api/auth/password/set/', { token, password })
      setStatus('success')
      const path = role==='BRANCH' ? '/login/branch' : role==='CORPER' ? '/login/corper' : '/login/org'
      setTimeout(()=> navigate(path), 1200)
    }catch(err){
      const msg = err?.response?.data?.detail || 'Failed to set password'
      setStatus(`error:${msg}`)
    }
  }

  if(token){
    return (
      <div className="row justify-content-center py-5">
        <div className="col-md-6 col-lg-5">
          <div className="card shadow-sm">
            <div className="card-body p-4">
              <h1 className="h4 mb-3 text-olive">Verify Success — Set Password</h1>
              <form onSubmit={setPwd}>
                <label className="form-label">New Password</label>
                <input className="form-control mb-3" type="password" value={password} onChange={e=>setPassword(e.target.value)} required minLength={8} />
                <label className="form-label">Confirm Password</label>
                <input className="form-control mb-3" type="password" value={confirm} onChange={e=>setConfirm(e.target.value)} required minLength={8} />
                <div className="d-grid">
                  <button className="btn btn-olive" disabled={status==='pending'}>Save Password</button>
                </div>
              </form>
              {status==='success' && <div className="alert alert-success mt-3 fade show">Password saved. Redirecting to login…</div>}
              {status?.startsWith('error') && <div className="alert alert-danger mt-3 fade show">{status.split(':')[1]}</div>}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="text-center py-5">
      <h1 className="display-6 text-olive">Email Verified</h1>
      <p className="lead">Your account is now active. You may close this window.</p>
    </div>
  )
}
