import React, { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useLocation } from 'react-router-dom'
import api, { ensureCsrf, setToken } from '../api/axios'
import { Building2, ShieldCheck, User, Users } from 'lucide-react'

export default function Login() {
  const navigate = useNavigate()
  const location = useLocation()
  const [form, setForm] = useState({ email: '', password: '' })
  const [role, setRole] = useState('ORG')
  const [remember, setRemember] = useState(true)
  const [status, setStatus] = useState(null)

  useEffect(() => {
    ensureCsrf()
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    const fromQuery = (params.get('role') || '').toUpperCase()
    if (fromQuery === 'ORG' || fromQuery === 'BRANCH' || fromQuery === 'CORPER') {
      setRole(fromQuery)
    }
  }, [location.search])

  const roles = useMemo(
    () => [
      { value: 'ORG', label: 'Organisation', Icon: Building2 },
      { value: 'BRANCH', label: 'Admin', Icon: ShieldCheck },
      { value: 'CORPER', label: 'Corps Member', Icon: User },
    ],
    []
  )

  const roleMeta = roles.find((r) => r.value === role) || roles[0]

  const onChange = (e) => setForm((f) => ({ ...f, [e.target.name]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    setStatus('pending')
    try {
      await ensureCsrf()
      const { data } = await api.post('/api/auth/login/', { ...form, role })
      if (data?.token) setToken(data.token, remember)
      const params = new URLSearchParams(location.search)
      const next = params.get('next') || ''
      const fallbackPath = role === 'ORG' ? '/dashboard/org' : role === 'BRANCH' ? '/dashboard/branch' : '/dashboard/corper'
      const path = next.startsWith('/') && !next.startsWith('//') ? next : fallbackPath
      navigate(path)
    } catch (err) {
      setStatus('error:' + (err?.response?.data?.detail || 'Login failed'))
    }
  }

  return (
    <div className="auth-page">
      <div className="container py-4 py-lg-5">
        <div className="row align-items-stretch g-4 justify-content-center">
          <div className="col-lg-5">
            <div className="auth-side p-4 p-lg-5 h-100">
              <span className="auth-eyebrow">Login</span>
              <h1 className="auth-title mt-2">Welcome back.</h1>
              <p className="text-muted mt-3">
                Sign in to manage attendance, approvals, and clearance letters.
              </p>

              <div className="auth-perk mt-4">
                <span className="auth-perk-icon" aria-hidden>
                  <Users size={18} />
                </span>
                <div>
                  <div className="fw-semibold">Role-based access</div>
                  <div className="small text-muted">Choose your login type to land in the right dashboard.</div>
                </div>
              </div>

              <div className="auth-side-footer mt-4">
                <span>New organisation?</span>
                <Link to="/signup" className="auth-link">
                  Create an account
                </Link>
              </div>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="auth-card p-4 p-lg-5 h-100">
              <div className="d-flex justify-content-between align-items-start gap-3 mb-3">
                <div>
                  <h2 className="h4 mb-1 text-olive">Sign in</h2>
                  <div className="text-muted small">Choose your login type and continue.</div>
                </div>
              </div>

              <form onSubmit={submit}>
                <div className="row g-3">
                  <div className="col-12">
                    <label className="form-label">Login Type</label>
                    <div className="input-group">
                      <span className="input-group-text" aria-hidden>
                        <roleMeta.Icon size={18} />
                      </span>
                      <select className="form-select" value={role} onChange={(e) => setRole(e.target.value)}>
                        {roles.map((r) => (
                          <option key={r.value} value={r.value}>
                            {r.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="col-12">
                    <label className="form-label">Email</label>
                    <input className="form-control" name="email" type="email" value={form.email} onChange={onChange} required />
                  </div>

                  <div className="col-12">
                    <label className="form-label">Password</label>
                    <input className="form-control" name="password" type="password" value={form.password} onChange={onChange} required />
                  </div>
                </div>

                <div className="d-flex justify-content-between align-items-center mt-3">
                  <div className="form-check">
                    <input
                      className="form-check-input"
                      id="remember-login"
                      type="checkbox"
                      checked={remember}
                      onChange={(e) => setRemember(e.target.checked)}
                    />
                    <label className="form-check-label small text-muted" htmlFor="remember-login">
                      Remember me
                    </label>
                  </div>
                  <Link to="/forgot-password" className="small auth-link">
                    Forgot password?
                  </Link>
                </div>

                <div className="d-grid mt-4">
                  <button className="btn btn-olive" disabled={status === 'pending'}>
                    {status === 'pending' ? 'Signing in…' : 'Login'}
                  </button>
                </div>
              </form>

              {status?.startsWith('error') && <div className="alert alert-danger mt-3">{status.split(':')[1]}</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
