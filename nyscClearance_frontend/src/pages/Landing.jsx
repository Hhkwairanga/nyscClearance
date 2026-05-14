import React, { useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  BadgeCheck,
  Building2,
  CheckCircle2,
  FileCheck2,
  LockKeyhole,
  Mail,
  MapPin,
  QrCode,
  ShieldCheck,
  Smartphone,
  Star,
  Users,
} from 'lucide-react'

function PricingCard({ title, price, items, featured }) {
  return (
    <div className="col-md-6 col-lg-3">
      <div className={`nl-pricing-card p-4 h-100 ${featured ? 'featured' : ''}`}>
        {featured && <span className="nl-save-badge">Most Recommended</span>}
        <span className="nl-eyebrow">{title}</span>
        <h2 className="nl-fw-black my-2">{price}</h2>
        <p className="small opacity-75">Best for growing organisations.</p>
        <hr />
        {items.map((item) => (
          <p className="small mb-2" key={item}>
            <CheckCircle2 className="nl-check" size={15} aria-hidden /> {''}
            {item}
          </p>
        ))}
        <Link
          to="/signup"
          className={`btn w-100 mt-3 rounded-pill ${featured ? 'btn-olive' : 'btn-outline-secondary'}`}
        >
          Start Free Trial
        </Link>
      </div>
    </div>
  )
}

export default function Landing() {
  const stats = useMemo(
    () => [
      [Building2, '120+', 'Organizations Onboarded'],
      [Users, '8,400+', 'Corps Members Tracked'],
      [FileCheck2, '52,000+', 'Clearance Letters Generated'],
      [ShieldCheck, '0', 'Fraud Incidents Reported'],
    ],
    []
  )

  return (
    <div className="nl-page">
      <section className="nl-hero-section py-5">
        <div className="container">
          <div className="nl-hero-card p-4 p-lg-5 position-relative overflow-hidden">
            <div className="row align-items-center g-5">
              <div className="col-lg-6">
                <h1 className="display-5 nl-fw-black mb-3">
                  NYSC Clearance,
                  <br />
                  Finally Automated.
                </h1>
                <p className="lead text-muted mb-4">
                  Face Recognition Check In, Geofenced Attendance and QR Verified Clearance Letters.
                  Manage Corps Members from One Platform, Set Up in Minutes.
                </p>
                <div className="d-flex flex-wrap gap-2 mb-5">
                  <Link className="btn btn-olive px-4" to="/signup">
                    Start free trial
                  </Link>
                  <Link className="btn btn-outline-secondary px-4" to="/login">
                    Book a demo
                  </Link>
                </div>
                <div className="d-flex align-items-center gap-3 small text-muted">
                  <div className="nl-avatar-stack" aria-hidden>
                    <span />
                    <span />
                    <span />
                  </div>
                  <span>Trusted by organization managers across Nigeria</span>
                </div>
              </div>
              <div className="col-lg-6">
                <div className="nl-dashboard-preview mx-auto" aria-hidden>
                  <div className="nl-preview-top" />
                  <div className="nl-preview-grid">
                    <div />
                    <div />
                    <div />
                    <div />
                    <div />
                    <div />
                  </div>
                  <div className="nl-preview-line w-75" />
                  <div className="nl-preview-line w-50" />
                  <div className="nl-preview-btn">Approved</div>
                </div>
              </div>
            </div>
            <div className="nl-circle one" aria-hidden />
            <div className="nl-circle two" aria-hidden />
          </div>
        </div>
      </section>

      <section className="nl-stats py-4">
        <div className="container">
          <div className="row text-center text-white g-4">
            {stats.map(([Icon, value, label]) => (
              <div className="col-6 col-lg-3" key={label}>
                <div className="nl-stat-icon" aria-hidden>
                  <Icon size={22} />
                </div>
                <h2 className="nl-fw-black mb-0">{value}</h2>
                <p className="small opacity-75 mb-0">{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-5 nl-bg-soft">
        <div className="container">
          <div className="text-center mb-4">
            <span className="nl-eyebrow">The problem</span>
            <h2 className="nl-section-title">Managing Corper Attendance<br />is Broken.</h2>
            <p className="text-muted small">Here’s what most organisations deal with every month.</p>
          </div>
          <div className="row justify-content-center g-4">
            <div className="col-lg-5">
              <div className="nl-problem-card old p-4">
                <h6 className="fw-bold">The Old Way</h6>
                {[
                  'Paper attendance registers are easy to fake, hard to store',
                  'WhatsApp messages create inconsistent records',
                  'Manually typed clearance letters for each corps member',
                  'No way to verify if a letter is genuine',
                ].map((t) => (
                  <p className="small mb-2 text-muted" key={t}>
                    ✖ {t}
                  </p>
                ))}
              </div>
            </div>
            <div className="col-lg-5">
              <div className="nl-problem-card new p-4">
                <h6 className="fw-bold">The New Way</h6>
                {[
                  'Face recognition check-in via any smartphone',
                  'Geofenced attendance, they must be present',
                  'Clearance letters auto-generated in one click',
                  'QR-code on every letter for instant verification',
                ].map((t) => (
                  <p className="small mb-2 text-muted" key={t}>
                    ● {t}
                  </p>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="py-5 nl-bg-soft">
        <div className="container">
          <div className="text-center mb-4">
            <span className="nl-eyebrow">Features</span>
            <h2 className="nl-section-title">Why NYSC Clearance<br />is Different</h2>
          </div>
          <div className="row g-4 justify-content-center">
            {[
              [Smartphone, 'Face ID Check-In via Any Smartphone', 'Corps members check in using face recognition on their personal phone or any available device.'],
              [MapPin, 'Geofenced Attendance, They Must Be Present', 'Attendance is only valid within the GPS boundary of your organisation.'],
              [QrCode, 'Tamper-Proof Clearance Letters', 'Every letter carries a unique QR code for instant verification.'],
              [ShieldCheck, 'Leaves & Holidays', 'Approve or reject leave requests, and set organization-wide public holidays.'],
            ].map(([Icon, title, body]) => (
              <div className="col-md-6 col-lg-3" key={title}>
                <div className="nl-feature-card p-4 h-100">
                  <span className="nl-feature-icon" aria-hidden>
                    <Icon size={20} />
                  </span>
                  <h6 className="fw-bold mt-4">{title}</h6>
                  <p className="small text-muted">{body}</p>
                  <a href="#pricing" className="small text-decoration-none nl-link">
                    Learn more
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="how" className="nl-how-section py-5 text-white">
        <div className="container text-center">
          <span className="nl-eyebrow light">How it works</span>
          <h2 className="nl-fw-black mb-2">Up and Running in Minutes</h2>
          <p className="small opacity-75 mb-5">
            No technical expertise needed. If you can use a smartphone, you can set up NYSC Clearance.
          </p>
          <div className="row g-4">
            {[
              'Onboard Your Organisation',
              'Enroll Your Corps Members',
              'Track Attendance',
              'Generate Clearance Letters',
            ].map((t, i) => (
              <div className="col-md-6 col-lg-3" key={t}>
                <div className="nl-step-number mx-auto mb-3">{i + 1}</div>
                <h6 className="fw-bold">{t}</h6>
                <p className="small opacity-75">Complete this step from your dashboard in a few guided clicks.</p>
              </div>
            ))}
          </div>
          <Link className="btn btn-light mt-4 rounded-pill px-4" to="/signup">
            See it in action <ArrowRight size={16} />
          </Link>
        </div>
      </section>

      <section className="py-5 nl-bg-soft">
        <div className="container">
          <div className="text-center mb-4">
            <span className="nl-eyebrow">Social proof</span>
            <h2 className="nl-section-title">Loved by HR Officers.<br />Appreciated by Corps Members.</h2>
          </div>
          <div className="row g-4 justify-content-center">
            {['Amina Abdulkareem', 'Anike Ogunlana', 'Fatima Bello'].map((name) => (
              <div className="col-md-6 col-lg-3" key={name}>
                <div className="nl-testimonial-card p-4 h-100">
                  <div className="nl-stars mb-3" aria-hidden>
                    <Star size={14} />
                    <Star size={14} />
                    <Star size={14} />
                    <Star size={14} />
                    <Star size={14} />
                  </div>
                  <p className="small text-muted fst-italic">
                    NYSC Clearance made monthly clearance faster, cleaner, and easier for our organisation.
                  </p>
                  <h6 className="fw-bold mb-0">{name}</h6>
                  <span className="small text-muted">HR Manager</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="pricing" className="py-5 nl-bg-soft">
        <div className="container">
          <div className="text-center mb-4">
            <span className="nl-eyebrow">Pricing</span>
            <h2 className="nl-section-title">Simple Pricing. No Surprises.</h2>
            <p className="small text-muted">Choose the plan that fits your organisation.</p>
          </div>
          <div className="row g-4 justify-content-center align-items-stretch">
            <PricingCard
              title="Starter"
              price="₦15,000"
              items={['Face enrollment', 'Geofenced attendance', 'Clearance letter generation', 'Email support']}
            />
            <PricingCard
              featured
              title="Growth"
              price="₦42,000"
              items={['Everything in Starter', 'Branch management', 'Leave & holiday management', 'Role-based dashboard', 'Priority support']}
            />
            <PricingCard
              title="Enterprise"
              price="Custom"
              items={['Everything in Growth', 'Custom onboarding', 'Dedicated support manager', 'SLA & API access']}
            />
          </div>
        </div>
      </section>

      <section className="nl-security-section py-5 nl-bg-soft">
        <div className="container">
          <div className="row align-items-center g-5 justify-content-center">
            <div className="col-lg-5">
              <span className="nl-eyebrow">Security & trust</span>
              <h2 className="nl-section-title text-start">Every Letter is Verifiable.<br />Every Record is Tamper-Proof.</h2>
              <p className="text-muted">
                NYSC clearance fraud is a known risk. NYSC Clearance reduces it with verification, audit-friendly records, and controlled access.
              </p>
              <div className="d-flex flex-wrap gap-2 mb-4">
                {['QR Verified', 'Geo-restricted', 'Immutable Records'].map((x) => (
                  <span className="nl-pill" key={x}>
                    {x}
                  </span>
                ))}
              </div>
              <Link className="btn btn-olive rounded-pill px-4" to="/signup">
                See How It Works
              </Link>
            </div>
            <div className="col-lg-4">
              <div className="nl-letter-card p-4">
                <h6 className="fw-bold">
                  Clearance Approved <CheckCircle2 size={16} aria-hidden />
                </h6>
                <div className="nl-letter-line w-100" />
                <div className="nl-letter-line w-75" />
                <div className="nl-letter-line w-50" />
                <BadgeCheck className="mt-4" color="var(--olive)" aria-hidden />
              </div>
              <div className="nl-secure-alert mt-3">
                <LockKeyhole size={18} aria-hidden /> 0 fraud incidents reported
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-5 nl-bg-soft">
        <div className="container">
          <div className="nl-corper-box mx-auto text-center p-4 p-lg-5">
            <span className="nl-eyebrow">For corps members</span>
            <h2 className="nl-section-title">Are You a Corps Member?</h2>
            <p className="text-muted small">You shouldn’t have to chase your supervisor for a clearance letter every month.</p>
            <Link className="btn btn-olive rounded-pill px-4 mb-4" to="/signup">
              Share With Your Employer
            </Link>
            <div className="nl-email-box text-start mx-auto">
              <Mail size={18} aria-hidden />
              <span>
                Hi Sir/Ma, I’d like to introduce you to NYSC Clearance, a platform that automates attendance and clearance letters.
              </span>
              <button
                className="btn btn-dark btn-sm rounded-pill ms-auto"
                type="button"
                onClick={() => {
                  const text =
                    "Hi Sir/Ma, I’d like to introduce you to NYSC Clearance, a platform that automates attendance and clearance letters.";
                  navigator?.clipboard?.writeText?.(text)
                }}
              >
                Copy
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="py-5 nl-bg-soft text-center">
        <div className="container">
          <h2 className="nl-section-title">Ready to Make Monthly<br />Clearance Effortless?</h2>
          <p className="small text-muted">Join organisations already using NYSC Clearance to save time and protect records.</p>
          <div className="d-flex justify-content-center flex-wrap gap-3">
            <Link className="btn btn-olive rounded-pill px-4" to="/signup">
              Start Your Free Trial
            </Link>
            <Link className="btn btn-outline-secondary rounded-pill px-4" to="/login">
              Book a 15-Minute Demo
            </Link>
          </div>
        </div>
      </section>

      <footer className="nl-footer py-5 text-white">
        <div className="container">
          <div className="row g-4">
            <div className="col-lg-4">
              <h5 className="fw-bold">NYSC Clearance</h5>
              <p className="small opacity-75">
                A smarter way to manage attendance and clearance for corps members.
              </p>
            </div>
            {[
              ['Product', 'Features', 'How it Works', 'Pricing'],
              ['Company', 'About', 'FAQ', 'Contact us'],
              ['Legal', 'Privacy Policy', 'Terms of Service'],
            ].map(([title, ...links]) => (
              <div className="col-6 col-lg-2" key={title}>
                <h6>{title}</h6>
                {links.map((l) => (
                  <p className="small opacity-75 mb-2" key={l}>
                    {title === 'Legal' && l === 'Privacy Policy' ? (
                      <Link className="nl-footer-link" to="/privacy">
                        {l}
                      </Link>
                    ) : title === 'Legal' && l === 'Terms of Service' ? (
                      <Link className="nl-footer-link" to="/terms">
                        {l}
                      </Link>
                    ) : (
                      l
                    )}
                  </p>
                ))}
              </div>
            ))}
          </div>
          <hr className="border-light opacity-25" />
          <p className="small opacity-75 mb-0">© {new Date().getFullYear()} Sahab Technology. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}
