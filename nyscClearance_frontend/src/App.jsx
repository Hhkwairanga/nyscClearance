import React, { useEffect, useState } from 'react'
import { Outlet, Link, useNavigate } from 'react-router-dom'
import api, { ensureCsrf } from './api/axios'

export default function App(){
  const navigate = useNavigate()
  const [me, setMe] = useState(null)
  useEffect(() => { (async()=>{ await ensureCsrf(); try{ const r = await api.get('/api/auth/me/'); setMe(r.data) }catch(e){} })() }, [])
  const logout = async () => { try{ await api.post('/api/auth/logout/'); setMe({authenticated:false}); navigate('/login') }catch(e){} }
  const isAuthed = !!me?.authenticated
  return (
    <>
      <nav className="navbar navbar-expand-lg navbar-dark bg-olive">
        <div className="container">
          <Link to="/" className="navbar-brand d-flex align-items-center">
            <img src="/nyscclearance_logo.png" alt="NYSC Clearance" height="32" className="me-2"/>
            <span>NYSC Clearance</span>
          </Link>
          <div className="ms-auto d-flex gap-2">
            {!isAuthed && <Link to="/signup" className="btn btn-khaki">Sign Up</Link>}
            {!isAuthed && <Link to="/login" className="btn btn-outline-light">Login</Link>}
            {isAuthed && <Link to="/dashboard" className="btn btn-khaki">Dashboard</Link>}
            {isAuthed && <button className="btn btn-outline-light" onClick={logout}>Logout</button>}
          </div>
        </div>
      </nav>
      <main className="container py-4">
        <Outlet />
      </main>
      <div className="app-footer">&copy; Sahab Technology 2025</div>
    </>
  )
}
