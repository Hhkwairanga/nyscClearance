import React, { useEffect, useRef, useState } from 'react'
import api, { ensureCsrf } from '../api/axios'
import MapPicker from '../components/MapPicker'
import { Bar, Doughnut } from 'react-chartjs-2'
import AutoFadeAlert from '../components/AutoFadeAlert'
import { Chart as ChartJS, CategoryScale, LinearScale, ArcElement, BarElement, Tooltip, Legend } from 'chart.js'
ChartJS.register(CategoryScale, LinearScale, ArcElement, BarElement, Tooltip, Legend)

export default function Dashboard(){
  const [profile, setProfile] = useState(null)
  const [me, setMe] = useState(null)
  const [branches, setBranches] = useState([])
  const [deps, setDeps] = useState([])
  const [units, setUnits] = useState([])
  const [corpers, setCorpers] = useState([])
  const [stats, setStats] = useState(null)
  const [holidays, setHolidays] = useState([])
  const [leaves, setLeaves] = useState([])
  const [notifications, setNotifications] = useState([])
  const [status, setStatus] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    (async () => {
      await ensureCsrf()
      await refreshAll()
    })()
  }, [])

  async function refreshAll(){
    try{
      const [m,p,b,d,u,c,s,h,l,n] = await Promise.all([
        api.get('/api/auth/me/'),
        api.get('/api/auth/profile/'),
        api.get('/api/auth/branches/'),
        api.get('/api/auth/departments/'),
        api.get('/api/auth/units/'),
        api.get('/api/auth/corpers/'),
        api.get('/api/auth/stats/'),
        api.get('/api/auth/holidays/'),
        api.get('/api/auth/leaves/'),
        api.get('/api/auth/notifications/'),
      ])
      setMe(m.data)
      setProfile(p.data)
      setBranches(b.data)
      setDeps(d.data)
      setUnits(u.data)
      setCorpers(c.data)
      setStats(s.data)
      setHolidays(h.data)
      setLeaves(l.data)
      setNotifications(n.data)
    }catch(e){ setStatus('error:failed to load') }
  }

  async function saveProfile(e){
    e.preventDefault()
    setStatus('pending')
    const form = new FormData(e.target)
    try{
      const res = await api.put('/api/auth/profile/', form, { headers: { 'Content-Type':'multipart/form-data' } })
      setProfile(res.data)
      setStatus('saved:profile')
    }catch(err){ setStatus('error:profile') }
  }

  const latRef = useRef(null)
  const lngRef = useRef(null)

  async function createBranch(e){
    e.preventDefault(); setStatus('pending')
    const form = new FormData(e.target)
    try{ await api.post('/api/auth/branches/', Object.fromEntries(form)); await refreshAll(); setStatus('saved:branch') }catch(e){ setStatus('error:branch') }
  }

  async function createDepartment(e){
    e.preventDefault(); setStatus('pending')
    const form = new FormData(e.target)
    try{ await api.post('/api/auth/departments/', Object.fromEntries(form)); await refreshAll(); setStatus('saved:department') }catch(e){ setStatus('error:department') }
  }

  async function createUnit(e){
    e.preventDefault(); setStatus('pending')
    const form = new FormData(e.target)
    try{ await api.post('/api/auth/units/', Object.fromEntries(form)); await refreshAll(); setStatus('saved:unit') }catch(e){ setStatus('error:unit') }
  }

  async function createCorper(e){
    e.preventDefault(); setStatus('pending')
    const fd = new FormData(e.target)
    const data = Object.fromEntries(fd)
    try{
      await api.post('/api/auth/corpers/', data)
      await refreshAll(); setStatus('saved:corper')
      e.target.reset()
    }catch(e){ setStatus('error:corper') }
  }

  async function createLeave(e){
    e.preventDefault(); setStatus('pending')
    const data = Object.fromEntries(new FormData(e.target))
    try{ await api.post('/api/auth/leaves/', data); await refreshAll(); setStatus('saved:leave'); e.target.reset() }catch(err){ setStatus('error:leave') }
  }

  async function createHoliday(e){
    e.preventDefault(); setStatus('pending')
    const data = Object.fromEntries(new FormData(e.target))
    try{ await api.post('/api/auth/holidays/', data); await refreshAll(); setStatus('saved:holiday'); e.target.reset() }catch(e){ setStatus('error:holiday') }
  }

  async function approveLeave(id){ try{ await api.post(`/api/auth/leaves/${id}/approve/`); await refreshAll() }catch(e){} }
  async function rejectLeave(id){ try{ await api.post(`/api/auth/leaves/${id}/reject/`); await refreshAll() }catch(e){} }

  const logoUrl = profile?.logo || ''

  return (
    <div className="container-fluid p-0">
      <nav className="navbar navbar-light bg-white border-bottom px-3 sticky-top d-flex justify-content-between topnav">
        <div className="d-flex align-items-center">
          {logoUrl ? (
            <img src={logoUrl} alt="Org Logo" style={{height:36, width:36, objectFit:'cover', borderRadius:6}}/>
          ) : (
            <div style={{height:36, width:36, borderRadius:6, background:'#eef2ea'}}></div>
          )}
          <div className="ms-2">
            <div className="fw-semibold small">{me?.name || 'Organization'}</div>
            <div className="text-muted small">{me?.email}</div>
          </div>
        </div>
        <div className="">
          <div className="btn-group" role="group" aria-label="Navigation">
            {me?.role!=='CORPER' && (
              <button className={`btn btn-sm ${activeTab==='overview'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('overview')}>Home</button>
            )}
            {(me?.role==='ORG') && (
              <button className={`btn btn-sm ${activeTab==='structure'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('structure')}>Structure</button>
            )}
            {(me?.role==='ORG' || me?.role==='BRANCH') && (
              <button className={`btn btn-sm ${activeTab==='corpers'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('corpers')}>Corpers Management</button>
            )}
            {(me?.role==='BRANCH') && (
              <>
                <button className={`btn btn-sm ${activeTab==='leave'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('leave')}>Leave Management {leaves.filter(l=>l.status==='PENDING').length ? <span className="badge bg-danger ms-1">{leaves.filter(l=>l.status==='PENDING').length}</span> : null}</button>
                <button className={`btn btn-sm ${activeTab==='query'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('query')}>Query Management</button>
                <button className={`btn btn-sm ${activeTab==='report'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('report')}>Report</button>
              </>
            )}
            {(me?.role==='CORPER') && (
              <>
                <button className={`btn btn-sm ${activeTab==='attendance'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('attendance')}>Attendance Overview</button>
                <button className={`btn btn-sm ${activeTab==='leave'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('leave')}>Leave Management</button>
                <button className={`btn btn-sm ${activeTab==='performance'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('performance')}>Performance Clearance</button>
              </>
            )}
          </div>
        </div>
      </nav>
      <main className="p-3 p-md-4">
          {activeTab==='overview' && (
            <>
              <h2 className="mb-3 text-olive">Overview</h2>
              {me?.role==='ORG' && (
                <div className="card shadow-sm mb-3"><div className="card-body">
                  <h5 className="card-title">Send Notification (All or Branch)</h5>
                  <form onSubmit={(e)=>{ e.preventDefault(); const data = Object.fromEntries(new FormData(e.target)); (async()=>{ try{ await api.post('/api/auth/notifications/', data); await refreshAll() }catch(err){} })(); e.target.reset() }}>
                    <div className="row g-2">
                      <div className="col-md-4"><input className="form-control" name="title" placeholder="Title" required/></div>
                      <div className="col-md-4"><select className="form-select" name="branch"><option value="">All branches</option>{branches.map(b=> <option key={b.id} value={b.id}>{b.name}</option>)}</select></div>
                      <div className="col-12"><textarea className="form-control" name="message" rows="2" placeholder="Message" required/></div>
                      <div className="col-12 col-md-2 d-grid"><button className="btn btn-olive">Send</button></div>
                    </div>
                  </form>
                </div></div>
              )}
              {me?.role==='BRANCH' && (
                <div className="card shadow-sm mb-3"><div className="card-body">
                  <h5 className="card-title">Notify My Branch Corpers</h5>
                  <form onSubmit={(e)=>{ e.preventDefault(); const data = Object.fromEntries(new FormData(e.target)); (async()=>{ try{ await api.post('/api/auth/notifications/', data); await refreshAll() }catch(err){} })(); e.target.reset() }}>
                    <div className="row g-2">
                      <div className="col-md-4"><input className="form-control" name="title" placeholder="Title" required/></div>
                      <div className="col-12"><textarea className="form-control" name="message" rows="2" placeholder="Message" required/></div>
                      <div className="col-12 col-md-2 d-grid"><button className="btn btn-olive">Send</button></div>
                    </div>
                  </form>
                </div></div>
              )}
              {me?.role==='CORPER' && (
                <div className="card shadow-sm mb-3"><div className="card-body">
                  <h5 className="card-title">Notifications</h5>
                  <div className="list-group">
                    {notifications.map(n => (
                      <div key={n.id} className="list-group-item">
                        <div className="fw-semibold">{n.title}</div>
                        <div className="small text-muted">{new Date(n.created_at).toLocaleString()}</div>
                        <div className="mt-1">{n.message}</div>
                      </div>
                    ))}
                    {notifications.length===0 && <div className="text-muted">No notifications yet.</div>}
                  </div>
                </div></div>
              )}
              {me?.role!=='CORPER' && (
              <div className="row g-3">
                <div className="col-6 col-md-3">
                  <div className="card text-center shadow-sm"><div className="card-body">
                    <div className="text-muted small">Corpers</div>
                    <div className="h4 mb-0">{stats?.totals?.corpers ?? 0}</div>
                  </div></div>
                </div>
                <div className="col-6 col-md-3">
                  <div className="card text-center shadow-sm"><div className="card-body">
                    <div className="text-muted small">Branches</div>
                    <div className="h4 mb-0">{stats?.totals?.branches ?? 0}</div>
                  </div></div>
                </div>
                <div className="col-6 col-md-3">
                  <div className="card text-center shadow-sm"><div className="card-body">
                    <div className="text-muted small">Departments</div>
                    <div className="h4 mb-0">{stats?.totals?.departments ?? 0}</div>
                  </div></div>
                </div>
                <div className="col-6 col-md-3">
                  <div className="card text-center shadow-sm"><div className="card-body">
                    <div className="text-muted small">Units</div>
                    <div className="h4 mb-0">{stats?.totals?.units ?? 0}</div>
                  </div></div>
                </div>
              </div>
              )}
              {me?.role!=='CORPER' && (
              <div className="row g-3 mt-1">
                <div className="col-12 col-lg-6">
                  <div className="card shadow-sm"><div className="card-body" style={{height:300}}>
                    <h6 className="card-title">Corpers by Branch</h6>
                    <Bar data={{
                      labels: (stats?.corpers_by_branch||[]).map(r=>r.branch),
                      datasets: [{
                        label: 'Corpers',
                        data: (stats?.corpers_by_branch||[]).map(r=>r.count),
                        backgroundColor: '#556B2F'
                      }]
                    }} options={{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}} }} />
                  </div></div>
                </div>
                <div className="col-12 col-lg-6">
                  <div className="card shadow-sm"><div className="card-body" style={{height:300}}>
                    <h6 className="card-title">Attendance (Placeholder)</h6>
                    <Doughnut data={{
                      labels:['Today','This Month'],
                      datasets:[{ data:[stats?.attendance?.today||0, stats?.attendance?.this_month||0], backgroundColor:['#BDB76B','#556B2F'] }]
                    }} options={{ responsive:true, maintainAspectRatio:false }} />
                  </div></div>
                </div>
              </div>
              )}
            </>
          )}

          {activeTab==='structure' && me?.role==='ORG' && (
            <>
              <h2 className="mb-3 text-olive">Structure</h2>
              <div className="row g-4">
                <div className="col-12">
                  <div className="card shadow-sm"><div className="card-body">
                    <h5 className="card-title">Organization Profile</h5>
                    <form onSubmit={saveProfile} encType="multipart/form-data">
                      <div className="row g-2">
                        <div className="col-6 col-md-3">
                          <label className="form-label">Late Time</label>
                          <input className="form-control" type="time" name="late_time" defaultValue={profile?.late_time || ''} />
                        </div>
                        <div className="col-6 col-md-3">
                          <label className="form-label">Closing Time</label>
                          <input className="form-control" type="time" name="closing_time" defaultValue={profile?.closing_time || ''} />
                        </div>
                        <div className="col-12 col-md-3">
                          <label className="form-label">Logo</label>
                          <input className="form-control" type="file" name="logo" accept="image/*" />
                        {/* no preview under choose file as requested */}
                        </div>
                        <div className="col-6 col-md-3">
                          <label className="form-label">Org Latitude</label>
                          <input className="form-control" name="location_lat" defaultValue={profile?.location_lat ?? ''} placeholder="Latitude" />
                        </div>
                        <div className="col-6 col-md-3">
                          <label className="form-label">Org Longitude</label>
                          <input className="form-control" name="location_lng" defaultValue={profile?.location_lng ?? ''} placeholder="Longitude" />
                        </div>
                        <div className="col-12 col-md-3 align-self-end">
                          <button className="btn btn-olive w-100">Save</button>
                        </div>
                      </div>
                    </form>
                    {status==='saved:profile' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Profile updated.</AutoFadeAlert>}
                    {status==='error:profile' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Failed to update profile.</AutoFadeAlert>}
                  </div></div>
                </div>
                <div className="col-lg-6">
                  <div className="card shadow-sm">
                    <div className="card-body">
                      <h5 className="card-title">Public Holidays</h5>
                      <form className="mb-3" onSubmit={createHoliday}>
                        <div className="row g-2">
                          <div className="col-md-4"><input className="form-control" name="title" placeholder="Holiday title" required/></div>
                          <div className="col-md-3"><input className="form-control" type="date" name="start_date" required/></div>
                          <div className="col-md-3"><input className="form-control" type="date" name="end_date" required/></div>
                          <div className="col-md-2 d-grid"><button className="btn btn-olive">Add</button></div>
                        </div>
                      </form>
                      <ul className="list-group">
                        {holidays.map(h => (
                          <li key={h.id} className="list-group-item d-flex justify-content-between align-items-center">
                            <span>{h.title} — {h.start_date}{h.end_date!==h.start_date?` → ${h.end_date}`:''}</span>
                            <button className="btn btn-sm btn-outline-danger" onClick={async()=>{ try{ await api.delete(`/api/auth/holidays/${h.id}/`); await refreshAll() }catch(e){} }}>Delete</button>
                          </li>
                        ))}
                        {holidays.length===0 && <li className="list-group-item text-muted">No holidays configured.</li>}
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="col-lg-6">
                  <div className="card shadow-sm">
                    <div className="card-body">
                      <h5 className="card-title">Create Branch Office</h5>
                      <form id="branch-form" onSubmit={createBranch}>
                        <label className="form-label">Branch name</label>
                        <input className="form-control mb-2" name="name" placeholder="Branch name" required/>
                        <label className="form-label">Address</label>
                        <textarea className="form-control mb-2" name="address" placeholder="Address"/>
                        <div className="row g-2">
                          <div className="col-md-4">
                            <label className="form-label">Admin Name</label>
                            <input className="form-control" name="admin_name" placeholder="Branch admin name" />
                          </div>
                          <div className="col-md-5">
                            <label className="form-label">Admin Email</label>
                            <input className="form-control" type="email" name="admin_email" placeholder="admin@example.com" />
                          </div>
                          <div className="col-md-3">
                            <label className="form-label">Staff ID</label>
                            <input className="form-control" name="admin_staff_id" placeholder="ID" />
                          </div>
                        </div>
                        <div className="row g-2 mb-2">
                          <div className="col-md-6">
                            <label className="form-label">Latitude</label>
                            <input className="form-control" name="latitude" placeholder="Latitude" ref={latRef}/>
                          </div>
                          <div className="col-md-6">
                            <label className="form-label">Longitude</label>
                            <input className="form-control" name="longitude" placeholder="Longitude" ref={lngRef}/>
                          </div>
                        </div>
                        <MapPicker onChange={(pos) => {
                          if(latRef.current) latRef.current.value = pos.lat.toFixed(6)
                          if(lngRef.current) lngRef.current.value = pos.lng.toFixed(6)
                        }}/>
                        <div className="mt-3 d-grid">
                          <button className="btn btn-olive">Add Branch</button>
                        </div>
                      </form>
                      {status==='saved:branch' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Branch created. If admin email provided, an invite was sent.</AutoFadeAlert>}
                      {status==='error:branch' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not create branch.</AutoFadeAlert>}
                    </div>
                  </div>
                </div>

                <div className="col-lg-6">
                  <div className="card shadow-sm">
                    <div className="card-body">
                      <h5 className="card-title">Create Department</h5>
                      <form onSubmit={createDepartment}>
                        <select className="form-select mb-2" name="branch" required>
                          <option value="">Select Branch</option>
                          {branches.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                        </select>
                        <input className="form-control mb-3" name="name" placeholder="Department name" required/>
                        <button className="btn btn-olive">Add Department</button>
                      </form>
                      {status==='saved:department' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Department created.</AutoFadeAlert>}
                      {status==='error:department' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not create department.</AutoFadeAlert>}
                    </div>
                  </div>
                </div>

                <div className="col-lg-6">
                  <div className="card shadow-sm">
                    <div className="card-body">
                      <h5 className="card-title">Create Unit (optional)</h5>
                      <form onSubmit={createUnit}>
                        <select className="form-select mb-2" name="department" required>
                          <option value="">Select Department</option>
                          {deps.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                        </select>
                        <input className="form-control mb-3" name="name" placeholder="Unit name" required/>
                        <button className="btn btn-olive">Add Unit</button>
                      </form>
                      {status==='saved:unit' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Unit created.</AutoFadeAlert>}
                      {status==='error:unit' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not create unit.</AutoFadeAlert>}
                    </div>
                  </div>
                </div>
              <div className="col-12">
                <div className="card shadow-sm">
                  <div className="card-body">
                    <h5 className="card-title">Registered Branches</h5>
                    <div className="row g-3">
                      {branches.length === 0 && <div className="text-muted">No branches yet.</div>}
                      {branches.map(b => (
                        <div className="col-12 col-sm-6 col-lg-4" key={b.id}>
                          <div className="border rounded p-3 h-100">
                            <div className="d-flex justify-content-between align-items-start">
                              <div>
                                <div className="fw-semibold">{b.name}</div>
                                <div className="small text-muted">{b.address || '—'}</div>
                              </div>
                              <button className="btn btn-sm btn-outline-secondary" onClick={()=>{
                                const name = prompt('Branch name', b.name) || b.name;
                                const address = prompt('Address', b.address||'') || '';
                                const admin_name = prompt('Admin name', b.admin_info?.name||'') || '';
                                const admin_email = prompt('Admin email', b.admin_info?.email||'') || '';
                                const admin_staff_id = prompt('Admin staff ID', b.admin_info?.staff_id||'') || '';
                                (async()=>{ try{ await api.put(`/api/auth/branches/${b.id}/`, { name, address, latitude:b.latitude, longitude:b.longitude, admin_name, admin_email, admin_staff_id }); await refreshAll() }catch(e){} })();
                              }}>Edit</button>
                            </div>
                            {(b.latitude || b.longitude) && (
                              <div className="small mt-1">Lat: {b.latitude ?? '—'} · Lng: {b.longitude ?? '—'}</div>
                            )}
                            {b.admin_info && (
                              <div className="mt-2 small">
                                <div><span className="text-muted">Admin:</span> {b.admin_info.name}</div>
                                <div><span className="text-muted">Email:</span> {b.admin_info.email}</div>
                                <div><span className="text-muted">Staff ID:</span> {b.admin_info.staff_id || '—'}</div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              </div>
            </>
          )}

          {activeTab==='corpers' && (me?.role==='ORG' || me?.role==='BRANCH') && (
            <>
              <h2 className="mb-3 text-olive">Corpers</h2>
              <div className="card shadow-sm">
                <div className="card-body">
                  <h5 className="card-title">Enroll Corp Member</h5>
                  <form onSubmit={createCorper}>
                    <div className="row g-2">
                      <div className="col-md-4">
                        <label className="form-label">Email</label>
                        <input className="form-control" type="email" name="email" placeholder="corper@example.com" required/>
                      </div>
                      <div className="col-md-4">
                        <label className="form-label">Fullname</label>
                        <input className="form-control" name="full_name" placeholder="surname first name lastname" required/>
                      </div>
                      <div className="col-md-2">
                        <label className="form-label">Gender</label>
                        <select className="form-select" name="gender" required>
                          <option value="">Gender</option>
                          <option value="M">Male</option>
                          <option value="F">Female</option>
                          <option value="O">Other</option>
                        </select>
                      </div>
                      <div className="col-md-3">
                        <label className="form-label">StateCode</label>
                        <input className="form-control" name="state_code" placeholder="AA/00A/0000" required/>
                      </div>
                      <div className="col-md-3">
                        <label className="form-label">Passing Out Date</label>
                        <input className="form-control" type="date" name="passing_out_date" required/>
                      </div>
                      <div className="col-md-3">
                        <label className="form-label">Branch</label>
                        <select className="form-select" name="branch" required>
                          <option value="">Select branch</option>
                          {branches.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                        </select>
                      </div>
                      <div className="col-md-3">
                        <label className="form-label">Department (optional)</label>
                        <select className="form-select" name="department">
                          <option value="">Department (optional)</option>
                          {deps.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                        </select>
                      </div>
                      <div className="col-md-3">
                        <label className="form-label">Unit (optional)</label>
                        <select className="form-select" name="unit">
                          <option value="">Unit (optional)</option>
                          {units.map(u => <option key={u.id} value={u.id}>{u.name}</option>)}
                        </select>
                      </div>
                      <div className="col-md-3 d-grid">
                        <button className="btn btn-olive">Enroll</button>
                      </div>
                    </div>
                  </form>
                  {status==='saved:corper' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Enrollment successful. A verification email was sent to the corper. Use native app for face enrollment.</AutoFadeAlert>}
                  {status==='error:corper' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not enroll corper.</AutoFadeAlert>}
                </div>
              </div>
            <div className="card shadow-sm mt-3">
              <div className="card-body">
                <h5 className="card-title">Registered Corpers</h5>
                <div className="table-responsive">
                  <table className="table table-sm align-middle">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Full Name</th>
                        <th>Email</th>
                        <th>State Code</th>
                        <th>Branch</th>
                      </tr>
                    </thead>
                    <tbody>
                      {corpers.map((c, idx) => (
                        <tr key={c.id}>
                          <td>{idx+1}</td>
                          <td>{c.full_name}</td>
                          <td>{c.email}</td>
                          <td>{c.state_code}</td>
                          <td>{(branches.find(b=>b.id===c.branch)?.name) || '—'}</td>
                        </tr>
                      ))}
                      {corpers.length===0 && (
                        <tr><td colSpan="5" className="text-muted">No corpers enrolled yet.</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            </>
          )}

          {activeTab==='attendance' && (
            <>
              <h2 className="mb-3 text-olive">Attendance</h2>
              <div className="alert alert-info">Attendance module coming next: daily logs, late tracking using your configured times.</div>
            </>
          )}

        {activeTab==='leave' && me?.role==='CORPER' && (
          <>
            <h2 className="mb-3 text-olive">Leave Management</h2>
            <div className="card shadow-sm mb-3"><div className="card-body">
              <h5 className="card-title">Apply for Leave</h5>
              <form onSubmit={createLeave}>
                <div className="row g-2">
                  <div className="col-md-4"><label className="form-label">Start Date</label><input className="form-control" type="date" name="start_date" required/></div>
                  <div className="col-md-4"><label className="form-label">End Date</label><input className="form-control" type="date" name="end_date" required/></div>
                  <div className="col-12"><label className="form-label">Reason</label><textarea className="form-control" name="reason" rows="2"/></div>
                  <div className="col-12 col-md-3 d-grid"><button className="btn btn-olive">Submit</button></div>
                </div>
              </form>
              {status==='saved:leave' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Leave request submitted.</AutoFadeAlert>}
              {status==='error:leave' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not submit leave request.</AutoFadeAlert>}
            </div></div>
            <div className="card shadow-sm"><div className="card-body">
              <h5 className="card-title">My Leave Requests</h5>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead><tr><th>#</th><th>Start</th><th>End</th><th>Status</th></tr></thead>
                  <tbody>
                    {leaves.map((r,idx)=>(<tr key={r.id}><td>{idx+1}</td><td>{r.start_date}</td><td>{r.end_date}</td><td>{r.status}</td></tr>))}
                    {leaves.length===0 && <tr><td colSpan="4" className="text-muted">No leave requests yet.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div></div>
          </>
        )}

        {activeTab==='leave' && me?.role==='BRANCH' && (
          <>
            <h2 className="mb-3 text-olive">Leave Management</h2>
            <div className="card shadow-sm"><div className="card-body">
              <h5 className="card-title">Pending Approvals</h5>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead><tr><th>#</th><th>Corper</th><th>Start</th><th>End</th><th>Reason</th><th>Actions</th></tr></thead>
                  <tbody>
                    {leaves.filter(l=>l.status==='PENDING').map((r,idx)=>(
                      <tr key={r.id}>
                        <td>{idx+1}</td>
                        <td>{r.corper_name}</td>
                        <td>{r.start_date}</td>
                        <td>{r.end_date}</td>
                        <td className="small">{r.reason}</td>
                        <td>
                          <button className="btn btn-sm btn-success me-2" onClick={()=>approveLeave(r.id)}>Approve</button>
                          <button className="btn btn-sm btn-danger" onClick={()=>rejectLeave(r.id)}>Reject</button>
                        </td>
                      </tr>
                    ))}
                    {leaves.filter(l=>l.status==='PENDING').length===0 && <tr><td colSpan="6" className="text-muted">No pending leave requests.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div></div>
          </>
        )}

          {activeTab==='query' && me?.role==='BRANCH' && (
            <>
              <h2 className="mb-3 text-olive">Query Management</h2>
              <div className="alert alert-info">Manage queries and disciplinary records here (coming soon).</div>
            </>
          )}

          {activeTab==='report' && me?.role==='BRANCH' && (
            <>
              <h2 className="mb-3 text-olive">Reports</h2>
              <div className="alert alert-info">Download and view branch reports (coming soon).</div>
            </>
          )}

          {activeTab==='performance' && me?.role==='CORPER' && (
            <>
              <h2 className="mb-3 text-olive">Performance Clearance</h2>
              <div className="alert alert-info">Performance metrics and clearance status (coming soon).</div>
            </>
          )}

          {activeTab==='profile' && me?.role==='CORPER' && (
            <>
              <h2 className="mb-3 text-olive">Profile Settings</h2>
              <div className="alert alert-info">Edit your personal details (coming soon).</div>
            </>
          )}
        </main>
    </div>
  )
}
