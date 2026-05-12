import React, { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import api, { ensureCsrf } from '../api/axios'
import { Building2, CheckCircle2, Mail, ShieldCheck } from 'lucide-react'

export default function Signup(){
  const [form, setForm] = useState({
    email: '', name: '', address: '', phone_number: ''
  })
  const [status, setStatus] = useState(null)

  const perks = useMemo(
    () => [
      {
        Icon: ShieldCheck,
        title: 'Secure onboarding',
        body: 'Verified accounts and role-based dashboards keep access controlled.',
      },
      {
        Icon: Mail,
        title: 'Email verification',
        body: 'Verify your email, then set your password securely.',
      },
      {
        Icon: Building2,
        title: 'Built for organisations',
        body: 'Branch admins, corpers, leaves and holidays all in one place.',
      },
    ],
    []
  )

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
      await api.post('/api/auth/register/', form)
      setStatus('success')
    } catch(err){
      const msg = err?.response?.data?.detail
        || Object.values(err?.response?.data || {})?.[0]?.[0]
        || err.message
      setStatus(`error:${msg}`)
    }
  }

  return (
    <div className="auth-page">
      <div className="container py-4 py-lg-5">
        <div className="row align-items-stretch g-4 justify-content-center">
          <div className="col-lg-5">
            <div className="auth-side p-4 p-lg-5 h-100">
              <span className="auth-eyebrow">Organisation signup</span>
              <h1 className="auth-title mt-2">Start automating NYSC monthly clearance.</h1>
              <p className="text-muted mt-3">
                Create your organisation account, set your location, and onboard branches and corps members with verified access.
              </p>

              <div className="mt-4 d-grid gap-3">
                {perks.map(({ Icon, title, body }) => (
                  <div className="auth-perk" key={title}>
                    <span className="auth-perk-icon" aria-hidden>
                      <Icon size={18} />
                    </span>
                    <div>
                      <div className="fw-semibold">{title}</div>
                      <div className="small text-muted">{body}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="auth-side-footer mt-4">
                <CheckCircle2 size={16} aria-hidden />
                <span>Already have an account?</span>
                <Link to="/login" className="auth-link">Log in</Link>
              </div>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="auth-card p-4 p-lg-5 h-100">
              <div className="d-flex justify-content-between align-items-start gap-3 mb-3">
                <div>
                  <h2 className="h4 mb-1 text-olive">Create your organisation account</h2>
                  <div className="text-muted small">Takes about 2 minutes.</div>
                </div>
              </div>

              <form onSubmit={submit}>
                <div className="row g-3">
                  <div className="col-12">
                    <label className="form-label">Email</label>
                    <input name="email" type="email" className="form-control" value={form.email} onChange={onChange} required />
                  </div>

                  <div className="col-12">
                    <label className="form-label">Organization Name</label>
                    <input name="name" className="form-control" value={form.name} onChange={onChange} required />
                  </div>

                  <div className="col-12">
                    <label className="form-label">Address</label>
                    <textarea name="address" className="form-control" value={form.address} onChange={onChange} rows="3" />
                  </div>

                  <div className="col-md-6">
                    <label className="form-label">Phone Number</label>
                    <input
                      name="phone_number"
                      type="tel"
                      className="form-control"
                      value={form.phone_number}
                      onChange={onChange}
                      placeholder="e.g., +2348012345678"
                    />
                  </div>
                </div>

                <div className="d-grid mt-4">
                  <button className="btn btn-olive" disabled={status === 'pending'}>
                    {status === 'pending' ? 'Creating…' : 'Create account'}
                  </button>
                </div>

                <div className="auth-meta mt-3">
                  <span className="text-muted small">By creating an account, you agree to our Terms.</span>
                  <Link to="/login" className="small auth-link">
                    Already have an account? Log in
                  </Link>
                </div>
              </form>

              {status === 'success' && (
                <div className="alert alert-success mt-3">Registration successful. Check your email to verify and set your password.</div>
              )}
              {status?.startsWith('error') && <div className="alert alert-danger mt-3">{status.split(':')[1]}</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
