import React from 'react'

export default function Terms() {
  return (
    <div className="legal-page">
      <div className="container py-5">
        <div className="legal-card p-4 p-lg-5">
          <h1 className="mb-1 text-olive">Terms & Conditions</h1>
          <p className="text-muted mb-4">Last updated: {new Date().getFullYear()}</p>

          <p className="mb-4">
            These Terms & Conditions (“Terms”) govern your access to and use of NYSC Clearance (the “Service”),
            operated by Sahab Technology (“we”, “us”, “our”). By creating an account, accessing, or using the Service,
            you agree to be bound by these Terms.
          </p>

          <h2 className="h5 mt-4">1. Eligibility</h2>
          <p className="text-muted">
            You represent that you have the legal authority to register an organisation and that the information you provide is accurate.
            If you are using the Service on behalf of an organisation, you represent that you are authorised to bind that organisation.
          </p>

          <h2 className="h5 mt-4">2. Account Registration & Security</h2>
          <ul className="text-muted">
            <li>You are responsible for maintaining the confidentiality of your login credentials.</li>
            <li>You must promptly notify us of any unauthorised access or suspicious activity.</li>
            <li>You are responsible for all actions performed through your account.</li>
          </ul>

          <h2 className="h5 mt-4">3. Acceptable Use</h2>
          <ul className="text-muted">
            <li>Do not misuse the Service, attempt to bypass access controls, or interfere with system integrity.</li>
            <li>Do not upload content you do not have rights to, or content that violates applicable laws.</li>
            <li>Do not use the Service to generate or distribute fraudulent clearance documents.</li>
          </ul>

          <h2 className="h5 mt-4">4. Data & Privacy</h2>
          <p className="text-muted">
            Our processing of personal data is described in our Privacy Policy. By using the Service you acknowledge that
            we may process information such as account details, attendance records, and organisation data to provide and improve the Service.
          </p>

          <h2 className="h5 mt-4">5. Service Availability</h2>
          <p className="text-muted">
            We aim to provide a reliable Service but do not guarantee uninterrupted availability. We may modify or discontinue
            parts of the Service at any time, including for maintenance, security, or legal compliance.
          </p>

          <h2 className="h5 mt-4">6. Fees & Payments (if applicable)</h2>
          <p className="text-muted">
            If you purchase a paid plan, you agree to pay the stated fees and applicable taxes. Fees are non-refundable unless otherwise required by law.
          </p>

          <h2 className="h5 mt-4">7. Intellectual Property</h2>
          <p className="text-muted">
            The Service, including its design, code, and branding, is owned by Sahab Technology. You receive a limited,
            non-exclusive right to use the Service in accordance with these Terms.
          </p>

          <h2 className="h5 mt-4">8. Termination</h2>
          <p className="text-muted">
            We may suspend or terminate access to the Service if we reasonably believe you have violated these Terms or applicable laws.
            You may stop using the Service at any time.
          </p>

          <h2 className="h5 mt-4">9. Disclaimer</h2>
          <p className="text-muted">
            The Service is provided “as is” and “as available”. We disclaim all warranties to the maximum extent permitted by law.
          </p>

          <h2 className="h5 mt-4">10. Limitation of Liability</h2>
          <p className="text-muted">
            To the maximum extent permitted by law, Sahab Technology shall not be liable for indirect, incidental, special,
            consequential, or punitive damages, or any loss of data, profits, or revenues.
          </p>

          <h2 className="h5 mt-4">11. Changes to These Terms</h2>
          <p className="text-muted">
            We may update these Terms from time to time. Continued use of the Service after changes become effective constitutes acceptance.
          </p>

          <h2 className="h5 mt-4">12. Contact</h2>
          <p className="text-muted mb-0">
            For questions about these Terms, contact Sahab Technology via the official channels provided in the application.
          </p>
        </div>
      </div>
    </div>
  )
}

