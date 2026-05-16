import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import 'bootstrap/dist/css/bootstrap.min.css'
import './styles/theme.css'
import App from './App'
import Landing from './pages/Landing'
import Signup from './pages/Signup'
import Login from './pages/Login'
import VerifySuccess from './pages/VerifySuccess'
import Dashboard from './pages/Dashboard'
import ForgotPassword from './pages/ForgotPassword'
import ResetPassword from './pages/ResetPassword'
import Terms from './pages/Terms'
import Privacy from './pages/Privacy'
import { refreshForDeploymentVersion } from './deploymentVersion'

async function bootstrap(){
  const reloading = await refreshForDeploymentVersion()
  if(reloading) return

  createRoot(document.getElementById('root')).render(
    <React.StrictMode>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />}>
            <Route index element={<Landing />} />
            <Route path="signup" element={<Signup />} />
            <Route path="login" element={<Login />} />
            <Route path="verify-success" element={<VerifySuccess />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="dashboard/org" element={<Dashboard />} />
            <Route path="dashboard/branch" element={<Dashboard />} />
            <Route path="dashboard/corper" element={<Dashboard />} />
            <Route path="forgot-password" element={<ForgotPassword />} />
            <Route path="reset-password" element={<ResetPassword />} />
            <Route path="terms" element={<Terms />} />
            <Route path="privacy" element={<Privacy />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </React.StrictMode>
  )
}

bootstrap()
