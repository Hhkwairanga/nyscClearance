import React, { useEffect, useState } from 'react'
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom'
import api, { ensureCsrf, clearToken } from './api/axios'
import { RefreshCw } from 'lucide-react'
import PublicFooter from './components/PublicFooter'

function AppLoadingScreen(){
  return (
    <div className="app-loading-screen">
      <div className="app-loading-card text-center">
        <div className="dashboard-loading-icon mx-auto" aria-hidden>
          <RefreshCw size={24} className="spin-icon" />
        </div>
        <div className="app-loading-title mt-3">Preparing your workspace</div>
        <p className="text-muted small mb-3">Checking your session and loading the latest app version.</p>
        <div className="loading-progress" aria-hidden>
          <div className="loading-progress-bar" />
        </div>
      </div>
    </div>
  )
}

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

  if(me === null){
    return <AppLoadingScreen />
  }

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
	              <Link className="nav-link px-0 text-white-50" to="/about">About</Link>
	              <a className="nav-link px-0 text-white-50" href="https://home.sahabs.tech" target="_blank" rel="noreferrer">Services</a>
	              <Link className="nav-link px-0 text-white-50" to="/contact">Help</Link>
	            </div>
	            )}
	            {!isAuthed && <Link to="/signup" className="btn btn-khaki">Sign Up</Link>}
	            {!isAuthed && <Link to="/login" className="btn btn-outline-light">Login</Link>}
	            {isAuthed && <button className="btn btn-outline-light" onClick={logout}>Logout</button>}
	          </div>
        </div>
      </nav>
      <main className={isHome || isDashboard ? '' : 'container py-4'}>
        <Outlet />
      </main>
      {!isDashboard && <PublicFooter />}
    </div>
  )
}
