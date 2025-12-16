import React, { useEffect, useState } from 'react'
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom'
import api, { ensureCsrf } from './api/axios'

export default function App(){
  const navigate = useNavigate()
  const location = useLocation()
  const [me, setMe] = useState(null)
  useEffect(() => {
    (async()=>{
      try{
        await ensureCsrf()
        const r = await api.get('/api/auth/me/')
        setMe(r.data)
      }catch(e){
        setMe({ authenticated: false })
      }
    })()
  }, [location.pathname])
  const logout = async () => { try{ await api.post('/api/auth/logout/'); setMe({authenticated:false}); navigate('/login') }catch(e){} }
  const isAuthed = !!me?.authenticated
  const dashPath = !isAuthed ? '/' : (me?.role==='ORG' ? '/dashboard/org' : me?.role==='BRANCH' ? '/dashboard/branch' : '/dashboard/corper')
  return (
    <>
      <nav className="navbar navbar-expand-lg navbar-dark bg-olive">
        <div className="container">
          <Link to={dashPath} className="navbar-brand d-flex align-items-center">
            <img src="/nyscclearance_logo.png" alt="NYSC Clearance" height="32" className="me-2"/>
            <span>NYSC Clearance</span>
          </Link>
          <div className="ms-auto d-flex gap-2">
            {!isAuthed && <Link to="/signup" className="btn btn-khaki">Sign Up</Link>}
            {!isAuthed && <Link to="/login" className="btn btn-outline-light">Login</Link>}
            {isAuthed && <button className="btn btn-outline-light" onClick={logout}>Logout</button>}
          </div>
        </div>
      </nav>
      <main className="container py-4">
        <Outlet />
      </main>
      <a
        className="contact-fab"
        href="https://home.sahab.tech"
        target="_blank"
        rel="noreferrer"
        aria-label="Contact Us"
        title="Contact Us"
      >
        <span role="img" aria-hidden="true">📞</span>
        <span className="d-none d-sm-inline">Contact Us</span>
      </a>
      <div className="app-footer">&copy; Sahab Technology 2025</div>
    </>
  )
}
