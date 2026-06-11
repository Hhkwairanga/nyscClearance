import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { AlertTriangle, Home, RefreshCw, WifiOff } from 'lucide-react'

function SupportActions({ showRetry = true }){
  return (
    <div className="d-flex flex-wrap justify-content-center gap-2 mt-4">
      {showRetry && (
        <button className="btn btn-olive" onClick={() => window.location.reload()}>
          <RefreshCw size={16} className="me-2" />
          Try again
        </button>
      )}
      <Link className="btn btn-outline-secondary" to="/">
        <Home size={16} className="me-2" />
        Go home
      </Link>
      <Link className="btn btn-khaki" to="/contact">
        Contact support
      </Link>
    </div>
  )
}

export function AppErrorPage({
  code = 'Something went wrong',
  title = 'We could not load this page',
  message = 'NYSC Clearance ran into an unexpected issue. Please refresh the page or contact support if it continues.',
  icon = 'error',
  showRetry = true,
}){
  const Icon = icon === 'network' ? WifiOff : AlertTriangle
  return (
    <section className="app-error-page">
      <div className="app-error-card text-center">
        <div className="app-error-icon mx-auto" aria-hidden>
          <Icon size={30} />
        </div>
        <p className="app-error-code mb-2">{code}</p>
        <h1>{title}</h1>
        <p className="text-muted mb-0">{message}</p>
        <SupportActions showRetry={showRetry} />
        <div className="app-error-help mt-4">
          <strong>Need help?</strong> Email <a href="mailto:admin@sahabs.tech">admin@sahabs.tech</a> or call{' '}
          <a href="tel:+2347082505053">+2347082505053</a>.
        </div>
      </div>
    </section>
  )
}

export function NotFoundPage(){
  return (
    <AppErrorPage
      code="404"
      title="This NYSC Clearance page was not found"
      message="The page may have moved, the link may be incomplete, or your organization workspace may use a different route."
      showRetry={false}
    />
  )
}

export function NetworkFailurePage(){
  const location = useLocation()
  const reason = location.state?.reason || ''
  return (
    <AppErrorPage
      code="Network connection"
      title="We could not reach NYSC Clearance"
      message={reason || 'Please check your internet connection and try again. If your network is fine, the API may be temporarily unavailable.'}
      icon="network"
    />
  )
}

export class AppErrorBoundary extends React.Component{
  constructor(props){
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(){
    return { hasError: true }
  }

  componentDidCatch(error, info){
    try{ console.error('NYSC Clearance UI error', error, info) }catch(e){}
  }

  render(){
    if(this.state.hasError){
      return (
        <AppErrorPage
          code="Application error"
          title="NYSC Clearance needs a quick refresh"
          message="A page component failed to load correctly. Refresh the app to continue from a clean state."
        />
      )
    }
    return this.props.children
  }
}
