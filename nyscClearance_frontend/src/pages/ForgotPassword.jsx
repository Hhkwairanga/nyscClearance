import React, { useState } from 'react'
import api, { ensureCsrf } from '../api/axios'

export default function ForgotPassword(){
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState(null)

  async function submit(e){
    e.preventDefault(); setStatus('pending')
    try{
      await ensureCsrf()
      await api.post('/api/auth/password/reset/', { email })
      setStatus('sent')
    }catch(err){ setStatus('sent') }
  }

  return (
    <div className="row justify-content-center py-5">
      <div className="col-md-6 col-lg-5">
        <div className="card shadow-sm">
          <div className="card-body p-4">
            <h1 className="h4 mb-3 text-olive">Forgot Password</h1>
            <form onSubmit={submit}>
              <label className="form-label">Email</label>
              <input className="form-control mb-3" type="email" value={email} onChange={e=>setEmail(e.target.value)} required />
              <div className="d-grid"><button className="btn btn-olive" disabled={status==='pending'}>Send Reset Link</button></div>
            </form>
            {status==='sent' && <div className="alert alert-info mt-3">If the email exists, a reset link was sent.</div>}
          </div>
        </div>
      </div>
    </div>
  )
}

