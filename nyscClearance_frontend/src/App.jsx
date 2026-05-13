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
  return (
    <div className="app-shell">
      <nav className="navbar navbar-expand-lg navbar-dark bg-olive">
        <div className="container">
          <Link to={dashPath} className="navbar-brand d-flex align-items-center">
            <img src="/nyscclearance_logo.svg" alt="NYSC Clearance" className="brand-logo"/>
          </Link>
          <div className="ms-auto d-flex gap-2">
            {!isAuthed && <Link to="/signup" className="btn btn-khaki">Sign Up</Link>}
            {!isAuthed && <Link to="/login" className="btn btn-outline-light">Login</Link>}
            {isAuthed && <button className="btn btn-outline-light" onClick={logout}>Logout</button>}
          </div>
        </div>
      </nav>
      <main className={isHome || isDashboard ? '' : 'container py-4'}>
        <Outlet />
      </main>
      {/* Footer intentionally only exists on the landing page */}
    </div>
  )
}
