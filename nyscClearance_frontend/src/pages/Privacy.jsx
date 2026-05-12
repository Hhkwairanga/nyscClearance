import React from 'react'

export default function Privacy() {
  return (
    <div className="legal-page">
      <div className="container py-5">
        <div className="legal-card p-4 p-lg-5">
          <h1 className="mb-1 text-olive">Privacy Policy</h1>
          <p className="text-muted mb-4">Last updated: {new Date().getFullYear()}</p>

          <p className="mb-4">
            This Privacy Policy explains how Sahab Technology (“we”, “us”, “our”) collects, uses, shares, and protects information
            when you use NYSC Clearance (the “Service”).
          </p>

          <h2 className="h5 mt-4">1. Information We Collect</h2>
          <ul className="text-muted">
            <li><span className="fw-semibold">Account details</span>: name, email, phone number, organisation profile data.</li>
            <li><span className="fw-semibold">Operational data</span>: attendance logs, clearance status, branch/corper records created within the Service.</li>
            <li><span className="fw-semibold">Technical data</span>: device/browser information, IP address, and basic usage logs for security and performance.</li>
          </ul>

          <h2 className="h5 mt-4">2. How We Use Information</h2>
          <ul className="text-muted">
            <li>Provide, maintain, and improve the Service.</li>
            <li>Authenticate users, enforce security controls, and prevent fraud.</li>
            <li>Send service emails such as verification, password setup, and important account notifications.</li>
            <li>Comply with legal obligations and enforce our Terms.</li>
          </ul>

          <h2 className="h5 mt-4">3. How We Share Information</h2>
          <p className="text-muted">
            We do not sell your personal data. We may share information with trusted service providers who help us operate the Service
            (for example, email delivery or payment processing). We may also share information when required by law or to protect our rights.
          </p>

          <h2 className="h5 mt-4">4. Data Retention</h2>
          <p className="text-muted">
            We retain information for as long as necessary to provide the Service and meet legal, security, and operational requirements.
          </p>

          <h2 className="h5 mt-4">5. Security</h2>
          <p className="text-muted">
            We use appropriate technical and organisational measures to protect information. No system is 100% secure,
            so we cannot guarantee absolute security.
          </p>

          <h2 className="h5 mt-4">6. Your Choices</h2>
          <ul className="text-muted">
            <li>You may request corrections to inaccurate account information.</li>
            <li>You may request account deletion subject to legal/operational retention requirements.</li>
          </ul>

          <h2 className="h5 mt-4">7. Changes to This Policy</h2>
          <p className="text-muted">
            We may update this Policy from time to time. Continued use of the Service after changes become effective constitutes acceptance.
          </p>

          <h2 className="h5 mt-4">8. Contact</h2>
          <p className="text-muted mb-0">
            If you have questions about this Privacy Policy, contact Sahab Technology via the official channels provided in the application.
          </p>
        </div>
      </div>
    </div>
  )
}

