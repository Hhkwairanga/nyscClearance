import React from 'react'
import { Link } from 'react-router-dom'

export default function Landing(){
  return (
    <div className="landing-hero">
      <div className="container py-5">
        <div className="row align-items-center">
          <div className="col-lg-6">
            <h1 className="display-5 fw-bold text-olive">Automate Monthly Performance Clearance</h1>
            <p className="lead mt-3">Streamline corpers’ clearance with secure onboarding, tracked performance, and instant clearance letters — all in one place.</p>
            <div className="mt-4 d-flex gap-3">
              <Link className="btn btn-olive btn-lg" to="/signup">Get Started</Link>
              <a className="btn btn-outline-secondary btn-lg" href="#features">Learn More</a>
            </div>
          </div>
          <div className="col-lg-6 mt-4 mt-lg-0">
            <div className="animation-card">
              <div className="bubble b1"></div>
              <div className="bubble b2"></div>
              <div className="bubble b3"></div>
              <div className="doc shadow-sm">
                <div className="line w-75"></div>
                <div className="line w-50"></div>
                <div className="line w-100"></div>
                <div className="status">Clearance Approved ✓</div>
              </div>
            </div>
          </div>
        </div>

        <section id="features" className="mt-5 pt-4">
          <div className="row g-4">
            <div className="col-md-4">
              <div className="card h-100 shadow-sm feature-card">
                <div className="card-body">
                  <h5 className="card-title">Email Verification</h5>
                  <p className="card-text">Secure organization onboarding with verified accounts and controlled access.</p>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100 shadow-sm feature-card">
                <div className="card-body">
                  <h5 className="card-title">Automated Workflow</h5>
                  <p className="card-text">Monthly performance capture to generate official clearance letters automatically.</p>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100 shadow-sm feature-card">
                <div className="card-body">
                  <h5 className="card-title">Bootstrap Design</h5>
                  <p className="card-text">Clean olive/khaki theme aligned with NYSC style for consistency.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

