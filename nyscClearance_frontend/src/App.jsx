import React, { useEffect, useState } from 'react'
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom'
import api, { ensureCsrf, clearToken } from './api/axios'

export default function App(){
  const navigate = useNavigate()
  const location = useLocation()
  const isHome = location.pathname === '/'
  const isDashboard = location.pathname === '/dashboard' || location.pathname.startsWith('/dashboard/')
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
  const logout = async () => {
    try{ await api.post('/api/auth/logout/') }catch(e){}
    try{ clearToken() }catch(e){}
    setMe({authenticated:false}); navigate('/login')
  }
  const isAuthed = !!me?.authenticated
  const dashPath = !isAuthed ? '/' : (me?.role==='ORG' ? '/dashboard/org' : me?.role==='BRANCH' ? '/dashboard/branch' : '/dashboard/corper')

  useEffect(() => {
    if(isHome && isAuthed){
      navigate(dashPath, { replace: true })
    }
  }, [isHome, isAuthed, dashPath, navigate])

  const holdLandingForAuthCheck = isHome && me === null

  return (
    <div className="app-shell">
      <nav className="navbar navbar-expand-lg navbar-dark bg-olive">
        <div className="container">
          <Link to={dashPath} className="navbar-brand d-flex align-items-center">
            <img src="/nyscclearance_logo.svg" alt="NYSC Clearance" className="brand-logo"/>
          </Link>
          <div className="ms-auto d-flex align-items-center gap-2">
            {isAuthed && isDashboard && (
            <div className="d-none d-lg-flex align-items-center gap-3 me-3">
              <a className="nav-link px-0 text-white-50" href="https://home.sahabs.tech" target="_blank" rel="noreferrer">About</a>
              <a className="nav-link px-0 text-white-50" href="https://home.sahabs.tech" target="_blank" rel="noreferrer">Services</a>
              <a className="nav-link px-0 text-white-50" href="https://home.sahabs.tech" target="_blank" rel="noreferrer">Contact</a>
              <a className="nav-link px-0 text-white-50" href="https://home.sahabs.tech" target="_blank" rel="noreferrer">Help</a>
            </div>
            )}
            {!isAuthed && <Link to="/signup" className="btn btn-khaki">Sign Up</Link>}
            {!isAuthed && <Link to="/login" className="btn btn-outline-light">Login</Link>}
            {isAuthed && <button className="btn btn-outline-light" onClick={logout}>Logout</button>}
          </div>
        </div>
      </nav>
      <main className={isHome || isDashboard ? '' : 'container py-4'}>
        {holdLandingForAuthCheck ? null : <Outlet />}
      </main>
      {/* Footer intentionally only exists on the landing page */}
    </div>
  )
}
