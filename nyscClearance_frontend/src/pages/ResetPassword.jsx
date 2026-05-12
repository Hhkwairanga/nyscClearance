import React, { useMemo, useState } from 'react'
import api, { ensureCsrf } from '../api/axios'
import { Link, useNavigate } from 'react-router-dom'
import { KeyRound, LockKeyhole } from 'lucide-react'

export default function ResetPassword(){
  const params = useMemo(()=> new URLSearchParams(window.location.search), [])
  const token = params.get('token')
  const role = params.get('role')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [status, setStatus] = useState(null)
  const navigate = useNavigate()

  async function submit(e){
    e.preventDefault(); setStatus('pending')
    if(password !== confirm){ setStatus('error:mismatch'); return }
    try{
      await ensureCsrf()
      await api.post('/api/auth/password/reset/confirm/', { token, password })
      setStatus('success')
      const r = (role || '').toUpperCase()
      const to = r === 'BRANCH' ? '/login?role=BRANCH' : r === 'CORPER' ? '/login?role=CORPER' : '/login?role=ORG'
      setTimeout(() => navigate(to), 1200)
    }catch(err){ setStatus('error') }
  }

  if(!token){
    return <div className="text-center py-5">Missing or invalid reset token.</div>
  }

  return (
    <div className="auth-page">
      <div className="container py-4 py-lg-5">
        <div className="row align-items-stretch g-4 justify-content-center">
          <div className="col-lg-5">
            <div className="auth-side p-4 p-lg-5 h-100">
              <span className="auth-eyebrow">Password reset</span>
              <h1 className="auth-title mt-2">Choose a new password.</h1>
              <p className="text-muted mt-3">
                Your new password must be at least 8 characters. After saving, you’ll be redirected to the login page.
              </p>

              <div className="mt-4 d-grid gap-3">
                <div className="auth-perk">
                  <span className="auth-perk-icon" aria-hidden>
                    <LockKeyhole size={18} />
                  </span>
                  <div>
                    <div className="fw-semibold">Keep it secure</div>
                    <div className="small text-muted">Use a unique password you don’t reuse elsewhere.</div>
                  </div>
                </div>
              </div>

              <div className="auth-side-footer mt-4">
                <span>Need a new reset link?</span>
                <Link to="/forgot-password" className="auth-link">
                  Request again
                </Link>
              </div>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="auth-card p-4 p-lg-5 h-100">
              <h2 className="h4 mb-1 text-olive">Reset Password</h2>
              <div className="text-muted small mb-3">Set a new password for your account.</div>

              <form onSubmit={submit}>
                <label className="form-label">New Password</label>
                <div className="input-group mb-3">
                  <span className="input-group-text" aria-hidden>
                    <KeyRound size={18} />
                  </span>
                  <input
                    className="form-control"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    minLength={8}
                  />
                </div>

                <label className="form-label">Confirm Password</label>
                <input
                  className="form-control mb-3"
                  type="password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                  minLength={8}
                />

                <div className="d-grid">
                  <button className="btn btn-olive" disabled={status === 'pending'}>
                    {status === 'pending' ? 'Saving…' : 'Save New Password'}
                  </button>
                </div>
              </form>

              {status === 'success' && (
                <div className="alert alert-success mt-3 mb-0">Password updated. Redirecting to login…</div>
              )}
              {status === 'error:mismatch' && (
                <div className="alert alert-danger mt-3 mb-0">Passwords do not match.</div>
              )}
              {status === 'error' && (
                <div className="alert alert-danger mt-3 mb-0">Could not reset password. Try requesting a new link.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
