import React from 'react'
import { Link } from 'react-router-dom'

export default function ChooseLoginType(){
  return (
    <div className="container py-5">
      <div className="row justify-content-center">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-body text-center">
              <h3 className="mb-4">Choose Login Type</h3>
              <div className="d-grid gap-3">
                <Link className="btn btn-dark" to="/login/organization">Organization Login</Link>
                <Link className="btn btn-dark" to="/login/branch">Branch Admin Login</Link>
                <Link className="btn btn-dark" to="/login/corper">Corper Login</Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
