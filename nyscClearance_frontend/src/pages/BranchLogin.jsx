import React, { useEffect, useState } from 'react'
import api, { ensureCsrf, setToken } from '../api/axios'
import { useNavigate } from 'react-router-dom'

export default function BranchLogin(){
  const navigate = useNavigate()
  const [form, setForm] = useState({ email:'', password:'' })
  const [status, setStatus] = useState(null)
  useEffect(() => { ensureCsrf() }, [])
  const onChange = e => setForm(f => ({...f, [e.target.name]: e.target.value}))
  const submit = async (e) => {
    e.preventDefault(); setStatus('pending')
    try{
      await ensureCsrf();
      const { data } = await api.post('/api/auth/login/', { ...form, role:'BRANCH' });
      if(data?.token){ setToken(data.token) }
      navigate('/dashboard/branch')
    }catch(err){ setStatus('error:'+(err?.response?.data?.detail||'Login failed')) }
  }
  return (
    <div className="row justify-content-center">
      <div className="col-md-6 col-lg-5">
        <div className="card shadow-sm"><div className="card-body p-4">
          <h1 className="h4 mb-3 text-olive">Branch Admin Login</h1>
          <form onSubmit={submit}>
            <div className="mb-3"><label className="form-label">Email</label><input className="form-control" name="email" type="email" value={form.email} onChange={onChange} required/></div>
            <div className="mb-3"><label className="form-label">Password</label><input className="form-control" name="password" type="password" value={form.password} onChange={onChange} required/></div>
            <div className="mb-3 text-end"><a href="/forgot-password" className="small">Forgot password?</a></div>
            <div className="d-grid"><button className="btn btn-olive" disabled={status==='pending'}>Login</button></div>
          </form>
          {status?.startsWith('error') && <div className="alert alert-danger mt-3">{status.split(':')[1]}</div>}
        </div></div>
      </div>
    </div>
  )
}
