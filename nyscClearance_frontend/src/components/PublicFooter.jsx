import React from 'react'
import { Link } from 'react-router-dom'

const productLinks = [
  ['Features', '/#features'],
  ['How it Works', '/#how'],
  ['Pricing', '/#pricing'],
]

const companyLinks = [
  ['About', '/#about'],
  ['FAQ', '/#faq'],
]

const CONTACT_EMAIL = 'admin@sahabs.tech'
const CONTACT_PHONE = '+2347082505053'

export default function PublicFooter(){
  return (
    <footer className="nl-footer py-5 text-white">
      <div className="container">
        <div className="row g-4">
          <div className="col-lg-4">
            <h5 className="fw-bold">NYSC Clearance</h5>
            <p className="small opacity-75">
              A smarter way to manage attendance and clearance for corps members.
            </p>
            <p className="small mb-1">
              <a className="nl-footer-link" href={`mailto:${CONTACT_EMAIL}`}>
                {CONTACT_EMAIL}
              </a>
            </p>
            <p className="small mb-0">
              <a className="nl-footer-link" href={`tel:${CONTACT_PHONE}`}>
                {CONTACT_PHONE}
              </a>
            </p>
          </div>

          <div className="col-6 col-lg-2">
            <h6>Product</h6>
            {productLinks.map(([label, href]) => (
              <p className="small mb-2" key={label}>
                <a className="nl-footer-link" href={href}>
                  {label}
                </a>
              </p>
            ))}
          </div>

          <div className="col-6 col-lg-2">
            <h6>Company</h6>
            {companyLinks.map(([label, href]) => (
              <p className="small mb-2" key={label}>
                <a className="nl-footer-link" href={href}>
                  {label}
                </a>
              </p>
            ))}
            <p className="small mb-2">
              <Link className="nl-footer-link" to="/contact">
                Contact us
              </Link>
            </p>
          </div>

          <div className="col-6 col-lg-2">
            <h6>Legal</h6>
            <p className="small mb-2">
              <Link className="nl-footer-link" to="/privacy">
                Privacy Policy
              </Link>
            </p>
            <p className="small mb-2">
              <Link className="nl-footer-link" to="/terms">
                Terms of Service
              </Link>
            </p>
          </div>
        </div>
        <hr className="border-light opacity-25" />
        <p className="small opacity-75 mb-0">© {new Date().getFullYear()} Sahab Technology. All rights reserved.</p>
      </div>
    </footer>
  )
}
