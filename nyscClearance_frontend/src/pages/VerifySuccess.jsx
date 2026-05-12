import React, { useEffect, useMemo, useState } from 'react'
import api, { ensureCsrf } from '../api/axios'
import { useNavigate } from 'react-router-dom'
import { Link } from 'react-router-dom'
import { BadgeCheck, KeyRound, ShieldCheck } from 'lucide-react'

export default function VerifySuccess(){
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const token = params.get('token')
  const roleParam = params.get('role')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [status, setStatus] = useState(null)
  const [verifiedRole, setVerifiedRole] = useState(roleParam)
  const navigate = useNavigate()

  async function setPwd(e){
    e.preventDefault()
    setStatus('pending')
    if(password !== confirm){ setStatus('error:Passwords do not match'); return }
    try{
      await ensureCsrf()
      await api.post('/api/auth/password/set/', { token, password })
      setStatus('success')
      const r = (verifiedRole || roleParam || '').toUpperCase()
      const path = r==='BRANCH' ? '/login?role=BRANCH' : r==='CORPER' ? '/login?role=CORPER' : '/login?role=ORG'
      setTimeout(()=> navigate(path), 1200)
    }catch(err){
      const msg = err?.response?.data?.detail || 'Failed to set password'
      setStatus(`error:${msg}`)
    }
  }

  useEffect(() => {
    if(!token) return
    // Call backend verify endpoint in SPA mode so account is marked verified without redirect
    (async () => {
      try{
        const r = await api.get(
          `/api/auth/verify/?token=${encodeURIComponent(token)}&front=1${roleParam ? `&role=${encodeURIComponent(roleParam)}` : ''}`
        )
        const fromApi = r?.data?.role
        if(fromApi) setVerifiedRole(fromApi)
      }catch(e){ /* ignore: the password set flow can still proceed */ }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  if(token){
    return (
      <div className="auth-page">
        <div className="container py-4 py-lg-5">
          <div className="row align-items-stretch g-4 justify-content-center">
            <div className="col-lg-5">
              <div className="auth-side p-4 p-lg-5 h-100">
                <span className="auth-eyebrow">Account activation</span>
                <h1 className="auth-title mt-2">Email verified.</h1>
                <p className="text-muted mt-3">Set a password to finish activating your account.</p>

                <div className="mt-4 d-grid gap-3">
                  <div className="auth-perk">
                    <span className="auth-perk-icon" aria-hidden>
                      <ShieldCheck size={18} />
                    </span>
                    <div>
                      <div className="fw-semibold">Secure setup</div>
                      <div className="small text-muted">Choose a strong password you don’t reuse elsewhere.</div>
                    </div>
                  </div>
                </div>

                <div className="auth-side-footer mt-4">
                  <span>Prefer to login later?</span>
                  <Link to="/login" className="auth-link">
                    Go to login
                  </Link>
                </div>
              </div>
            </div>

            <div className="col-lg-6">
              <div className="auth-card p-4 p-lg-5 h-100">
                <h2 className="h4 mb-1 text-olive">Set Password</h2>
                <div className="text-muted small mb-3">Your account is verified. Create a password to continue.</div>

                <form onSubmit={setPwd}>
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
                      {status === 'pending' ? 'Saving…' : 'Save Password'}
                    </button>
                  </div>
                </form>

                {status === 'success' && (
                  <div className="alert alert-success mt-3 mb-0 fade show">Password saved. Redirecting to login…</div>
                )}
                {status?.startsWith('error') && (
                  <div className="alert alert-danger mt-3 mb-0 fade show">{status.split(':')[1]}</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  useEffect(() => {
    const r = (verifiedRole || roleParam || '').toUpperCase()
    const to = r==='BRANCH' ? '/login?role=BRANCH' : r==='CORPER' ? '/login?role=CORPER' : '/login?role=ORG'
    const id = setTimeout(() => navigate(to), 1800)
    return () => clearTimeout(id)
  }, [navigate, roleParam, verifiedRole])

  return (
    <div className="auth-page">
      <div className="container py-4 py-lg-5">
        <div className="row align-items-stretch g-4 justify-content-center">
          <div className="col-lg-5">
            <div className="auth-side p-4 p-lg-5 h-100">
              <span className="auth-eyebrow">Account activation</span>
              <h1 className="auth-title mt-2">You’re all set.</h1>
              <p className="text-muted mt-3">Your email has been verified and your account is now active.</p>
              <div className="mt-4 d-grid gap-3">
                <div className="auth-perk">
                  <span className="auth-perk-icon" aria-hidden>
                    <BadgeCheck size={18} />
                  </span>
                  <div>
                    <div className="fw-semibold">Verification complete</div>
                    <div className="small text-muted">Redirecting you to the correct login page…</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="auth-card p-4 p-lg-5 h-100 d-flex flex-column justify-content-center">
              <h2 className="h4 mb-2 text-olive">Email Verified</h2>
              <p className="text-muted mb-4">Continue to login.</p>
              <a
                href={
                  (verifiedRole || roleParam || '').toUpperCase() === 'BRANCH'
                    ? '/login?role=BRANCH'
                    : (verifiedRole || roleParam || '').toUpperCase() === 'CORPER'
                      ? '/login?role=CORPER'
                      : '/login?role=ORG'
                }
                className="btn btn-olive"
              >
                Go to Login
              </a>
              <div className="small text-muted mt-2">Redirecting…</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
