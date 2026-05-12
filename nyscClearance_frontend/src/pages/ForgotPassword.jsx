import React, { useState } from 'react'
import api, { ensureCsrf } from '../api/axios'
import { Link } from 'react-router-dom'
import { Mail, ShieldCheck } from 'lucide-react'

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
    <div className="auth-page">
      <div className="container py-4 py-lg-5">
        <div className="row align-items-stretch g-4 justify-content-center">
          <div className="col-lg-5">
            <div className="auth-side p-4 p-lg-5 h-100">
              <span className="auth-eyebrow">Password reset</span>
              <h1 className="auth-title mt-2">Reset your password.</h1>
              <p className="text-muted mt-3">
                Enter the email address associated with your account and we’ll send a secure reset link.
              </p>

              <div className="mt-4 d-grid gap-3">
                <div className="auth-perk">
                  <span className="auth-perk-icon" aria-hidden>
                    <ShieldCheck size={18} />
                  </span>
                  <div>
                    <div className="fw-semibold">Secure by default</div>
                    <div className="small text-muted">Reset links are time-limited for safety.</div>
                  </div>
                </div>
              </div>

              <div className="auth-side-footer mt-4">
                <span>Remembered your password?</span>
                <Link to="/login" className="auth-link">
                  Back to login
                </Link>
              </div>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="auth-card p-4 p-lg-5 h-100">
              <h2 className="h4 mb-1 text-olive">Forgot Password</h2>
              <div className="text-muted small mb-3">We’ll email you a reset link.</div>

              <form onSubmit={submit}>
                <label className="form-label">Email</label>
                <div className="input-group mb-3">
                  <span className="input-group-text" aria-hidden>
                    <Mail size={18} />
                  </span>
                  <input className="form-control" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
                </div>
                <div className="d-grid">
                  <button className="btn btn-olive" disabled={status === 'pending'}>
                    {status === 'pending' ? 'Sending…' : 'Send Reset Link'}
                  </button>
                </div>
              </form>

              {status === 'sent' && (
                <div className="alert alert-info mt-3 mb-0">If the email exists, a reset link was sent.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
