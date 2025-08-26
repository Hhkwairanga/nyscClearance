import React from 'react'
import { Link } from 'react-router-dom'

export default function Login(){
  return (
    <div className="row justify-content-center py-4">
      <div className="col-md-8 col-lg-6">
        <div className="card shadow-sm"><div className="card-body p-4">
          <h1 className="h4 mb-3 text-olive">Choose Login Type</h1>
          <div className="list-group">
            <Link className="list-group-item list-group-item-action" to="/login/org">Organization Login</Link>
            <Link className="list-group-item list-group-item-action" to="/login/branch">Branch Admin Login</Link>
            <Link className="list-group-item list-group-item-action" to="/login/corper">Corper Login</Link>
          </div>
        </div></div>
      </div>
    </div>
  )
}
