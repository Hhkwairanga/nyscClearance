import React from 'react'
import { Link } from 'react-router-dom'

export default function Landing(){
  return (
    <div className="landing-hero">
      <div className="container py-5">
        <div className="row align-items-center">
          <div className="col-lg-6">
            <h1 className="display-5 fw-bold text-olive">Effortless NYSC Monthly Clearance</h1>
            <p className="lead mt-3">Onboard corpers, manage branches, approve leaves and publish holidays, then generate clearance in minutes. Simple, secure, and made for speed.</p>
            <div className="mt-4 d-flex gap-3">
              <Link className="btn btn-olive btn-lg" to="/signup">Sign Up</Link>
              <Link className="btn btn-outline-secondary btn-lg" to="/login">Log In</Link>
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
                <div className="status">Clearance Approved <span className="blink-check" aria-hidden>✓</span></div>
                <div className="scroll-window mt-2" aria-hidden>
                  <img
                    src="/clearance_letter.png"
                    alt="Clearance Letter"
                    className="clearance-letter-img scroll-content"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <section id="features" className="mt-5 pt-4">
          <div className="row g-4">
            <div className="col-md-4">
              <div className="card h-100 shadow-sm feature-card">
                <div className="card-body">
                  <h5 className="card-title">Verified Onboarding</h5>
                  <p className="card-text">Invite organizations, branch admins, and corpers with email verification to keep access locked down and accounts authentic.</p>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100 shadow-sm feature-card">
                <div className="card-body">
                  <h5 className="card-title">Leaves & Holidays</h5>
                  <p className="card-text">Approve or reject leave requests, set organization-wide public holidays, and keep your calendar coordinated across branches.</p>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100 shadow-sm feature-card">
                <div className="card-body">
                  <h5 className="card-title">Role-Based Dashboards</h5>
                  <p className="card-text">Clear views for organizations, branch admins, and corpers. See exactly what matters, act fast, and stay audit-ready.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
