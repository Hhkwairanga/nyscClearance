import React, { useMemo, useState } from 'react'
import api, { ensureCsrf } from '../api/axios'

export default function ResetPassword(){
  const params = useMemo(()=> new URLSearchParams(window.location.search), [])
  const token = params.get('token')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [status, setStatus] = useState(null)

  async function submit(e){
    e.preventDefault(); setStatus('pending')
    if(password !== confirm){ setStatus('error:mismatch'); return }
    try{
      await ensureCsrf()
      await api.post('/api/auth/password/reset/confirm/', { token, password })
      setStatus('success')
    }catch(err){ setStatus('error') }
  }

  if(!token){
    return <div className="text-center py-5">Missing or invalid reset token.</div>
  }

  return (
    <div className="row justify-content-center py-5">
      <div className="col-md-6 col-lg-5">
        <div className="card shadow-sm">
          <div className="card-body p-4">
            <h1 className="h4 mb-3 text-olive">Reset Password</h1>
            <form onSubmit={submit}>
              <label className="form-label">New Password</label>
              <input className="form-control mb-3" type="password" value={password} onChange={e=>setPassword(e.target.value)} required minLength={8} />
              <label className="form-label">Confirm Password</label>
              <input className="form-control mb-3" type="password" value={confirm} onChange={e=>setConfirm(e.target.value)} required minLength={8} />
              <div className="d-grid"><button className="btn btn-olive" disabled={status==='pending'}>Save New Password</button></div>
            </form>
            {status==='success' && <div className="alert alert-success mt-3">Password updated. You can now log in.</div>}
            {status==='error:mismatch' && <div className="alert alert-danger mt-3">Passwords do not match.</div>}
            {status==='error' && <div className="alert alert-danger mt-3">Could not reset password. Try requesting a new link.</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
