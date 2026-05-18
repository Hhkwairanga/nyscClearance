import React from 'react'
import { Link } from 'react-router-dom'
import { CalendarDays, Mail, MessageCircle, Phone } from 'lucide-react'

const CONTACT_EMAIL = 'admin@sahabs.tech'
const CONTACT_PHONE = '+2347082505053'

export default function Contact(){
  const demoSubject = 'NYSC Clearance demo request'
  const demoBody = 'Hello Sahab Technology, I would like to book a demo for NYSC Clearance.'
  const mailto = `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(demoSubject)}&body=${encodeURIComponent(demoBody)}`
  const tel = `tel:${CONTACT_PHONE.replace(/\s+/g, '')}`

  return (
    <div className="contact-page">
      <div className="container py-4 py-lg-5">
        <div className="row align-items-stretch g-4 justify-content-center">
          <div className="col-lg-5">
            <div className="auth-side p-4 p-lg-5 h-100">
              <span className="auth-eyebrow">Contact us</span>
              <h1 className="auth-title mt-2">Book a guided demo.</h1>
              <p className="text-muted mt-3">
                Tell us about your organisation, number of corps members, and preferred setup flow. We’ll help you choose the right plan.
              </p>
              <div className="mt-4 d-grid gap-3">
                <div className="auth-perk">
                  <span className="auth-perk-icon" aria-hidden>
                    <CalendarDays size={18} />
                  </span>
                  <div>
                    <div className="fw-semibold">Quick walkthrough</div>
                    <div className="small text-muted">See attendance, wallet, subscription, and clearance workflows in action.</div>
                  </div>
                </div>
                <div className="auth-perk">
                  <span className="auth-perk-icon" aria-hidden>
                    <MessageCircle size={18} />
                  </span>
                  <div>
                    <div className="fw-semibold">Implementation guidance</div>
                    <div className="small text-muted">We’ll explain the best rollout path for head office, branches, admins, and corps members.</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="auth-card p-4 p-lg-5 h-100">
              <h2 className="h4 mb-1 text-olive">Reach the team</h2>
              <div className="text-muted small mb-4">Choose the fastest channel for your demo request.</div>

              <div className="d-grid gap-3">
                <a className="contact-option" href={mailto}>
                  <span className="auth-perk-icon" aria-hidden>
                    <Mail size={18} />
                  </span>
                  <span>
                    <strong>Email us</strong>
                    <small>{CONTACT_EMAIL}</small>
                  </span>
                </a>
                <a className="contact-option" href={tel}>
                  <span className="auth-perk-icon" aria-hidden>
                    <Phone size={18} />
                  </span>
                  <span>
                    <strong>Call us</strong>
                    <small>{CONTACT_PHONE}</small>
                  </span>
                </a>
              </div>

              <div className="alert alert-success mt-4 mb-0">
                Prefer to explore first? <Link to="/signup" className="auth-link">Start with a free organisation account.</Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
