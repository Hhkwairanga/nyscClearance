import React from 'react'
import { Link } from 'react-router-dom'

export default function About(){
  return (
    <div className="container py-4">
      <div className="row justify-content-center">
        <div className="col-12 col-lg-10 col-xxl-9">
          <div className="card shadow-sm">
            <div className="card-body p-4 p-md-5">
              <div className="d-flex align-items-start justify-content-between gap-3 flex-wrap">
                <div>
                  <div className="text-muted small">About</div>
                  <h1 className="h3 mb-2">NYSC Clearance</h1>
                  <p className="text-muted mb-0">
                    A modern clearance and attendance platform that helps organisations manage corps members with less
                    paperwork and more confidence.
                  </p>
                </div>
                <div className="d-flex gap-2">
                  <Link to="/contact" className="btn btn-outline-secondary btn-sm">Contact support</Link>
                  <a href="https://home.sahabs.tech" target="_blank" rel="noreferrer" className="btn btn-olive btn-sm">Explore Sahabs</a>
                </div>
              </div>

              <hr className="my-4" />

              <h2 className="h5 mb-2">What it does</h2>
              <p className="mb-3">
                NYSC Clearance brings attendance, performance tracking, communication, and clearance letter workflows
                into one place for organisations, admins, and corps members.
              </p>

              <div className="row g-3">
                <div className="col-12 col-md-6">
                  <div className="border rounded-3 p-3 h-100">
                    <div className="fw-semibold mb-1">Attendance & compliance</div>
                    <div className="text-muted small">
                      Track clock-ins, apply attendance rules, and keep auditable records for reporting.
                    </div>
                  </div>
                </div>
                <div className="col-12 col-md-6">
                  <div className="border rounded-3 p-3 h-100">
                    <div className="fw-semibold mb-1">Clearance letters</div>
                    <div className="text-muted small">
                      Generate professional clearance letters with organisation branding and sign-off details.
                    </div>
                  </div>
                </div>
                <div className="col-12 col-md-6">
                  <div className="border rounded-3 p-3 h-100">
                    <div className="fw-semibold mb-1">Wallet & subscription</div>
                    <div className="text-muted small">
                      Keep services uninterrupted using wallet funding and subscription plans (where enabled).
                    </div>
                  </div>
                </div>
                <div className="col-12 col-md-6">
                  <div className="border rounded-3 p-3 h-100">
                    <div className="fw-semibold mb-1">Structure management</div>
                    <div className="text-muted small">
                      Organise branches, departments, and units so every corps member has a clear placement.
                    </div>
                  </div>
                </div>
              </div>

              <hr className="my-4" />

              <h2 className="h5 mb-2">Support</h2>
              <p className="mb-0">
                Need help or want a tailored setup? Reach us at <a href="mailto:admin@sahabs.tech">admin@sahabs.tech</a> or call{' '}
                <a href="tel:+2347082505053">+2347082505053</a>.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

