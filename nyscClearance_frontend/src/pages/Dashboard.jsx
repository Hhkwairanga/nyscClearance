// Dashboard: app shell for ORG / BRANCH / CORPER
// - Loads profile, structure, stats, notifications
// - Wallet: funding via Paystack init/verify; modal accepts comma-separated amounts
// - Handles callback params: ?paystack=1&reference=..., ?fund=1
import React, { useEffect, useRef, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import api, { ensureCsrf } from '../api/axios'
import MapPicker from '../components/MapPicker'
import { Bar, Doughnut } from 'react-chartjs-2'
import AutoFadeAlert from '../components/AutoFadeAlert'
import thankYouAudio from '../assets/thank_you_message.mp3'
import { Chart as ChartJS, CategoryScale, LinearScale, ArcElement, BarElement, Tooltip, Legend } from 'chart.js'
ChartJS.register(CategoryScale, LinearScale, ArcElement, BarElement, Tooltip, Legend)

export default function Dashboard(){
  const navigate = useNavigate()
  const location = useLocation()
  const [profile, setProfile] = useState(null)
  const [me, setMe] = useState(null)
  const [branches, setBranches] = useState([])
  const [deps, setDeps] = useState([])
  const [units, setUnits] = useState([])
  const [corpers, setCorpers] = useState([])
  const [stats, setStats] = useState(null)
  const [clearance, setClearance] = useState([])
  const [clQuery, setClQuery] = useState('')
  const [holidays, setHolidays] = useState([])
  const [leaves, setLeaves] = useState([])
  const [notifications, setNotifications] = useState([])
  const [wallet, setWallet] = useState(null)
  const [announcement, setAnnouncement] = useState(null)
  const [showFund, setShowFund] = useState(false)
  const [fundAmount, setFundAmount] = useState('')
  const [clPage, setClPage] = useState(1)
  const [status, setStatus] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [perf, setPerf] = useState(null)
  // Local helpers for enroll form filtering
  const [enrollBranch, setEnrollBranch] = useState('')
  const [enrollDept, setEnrollDept] = useState('')
  const [corperQuery, setCorperQuery] = useState('')

  // First load: ensure CSRF and fetch all data
  useEffect(() => {
    (async () => {
      await ensureCsrf()
      await refreshAll()
    })()
  }, [])

  // Handle result banners and Paystack return
  useEffect(() => {
    const sp = new URLSearchParams(window.location.search)
    // Open wallet fund modal if asked
    if(sp.get('fund') === '1'){
      setActiveTab('wallet')
      setShowFund(true)
      const url = new URL(window.location.href)
      url.searchParams.delete('fund')
      window.history.replaceState({}, '', url)
    }
    if(sp.get('capture') === 'success'){
      setStatus('saved:face-capture')
      setActiveTab('corpers')
      // Clean the query param without reloading
      const url = new URL(window.location.href)
      url.searchParams.delete('capture')
      window.history.replaceState({}, '', url)
    }
    if(sp.get('attendance') === 'success'){
      setStatus('saved:attendance')
      setActiveTab('attendance')
      const url = new URL(window.location.href)
      url.searchParams.delete('attendance')
      window.history.replaceState({}, '', url)
      try{
        const audio = new Audio(thankYouAudio)
        audio.volume = 1.0
        audio.play().catch(()=>{})
      }catch(e){}
    }
    // Paystack callback handling after redirect
    const paystack = sp.get('paystack')
    const reference = sp.get('reference')
    if(paystack && reference){
      (async()=>{
        try{
          const vr = await api.post('/api/auth/wallet/paystack/verify/', { reference })
          if(vr.data?.status === 'success'){
            setStatus('saved:wallet-fund')
            await refreshAll()
            setActiveTab('wallet')
          }else{
            setStatus('error:wallet-fund')
          }
        }catch(err){ setStatus('error:wallet-fund') }
        // Clean params
        const url = new URL(window.location.href)
        url.searchParams.delete('paystack'); url.searchParams.delete('reference')
        window.history.replaceState({}, '', url)
      })()
    }
  }, [])

  useEffect(() => {
    if(me && me.authenticated === false){
      navigate('/login')
    }
    // Ensure role-specific dashboard path to avoid conflicts
    if(me?.authenticated){
      const target = me.role==='ORG' ? '/dashboard/org' : me.role==='BRANCH' ? '/dashboard/branch' : '/dashboard/corper'
      if(location.pathname === '/dashboard' || (location.pathname.startsWith('/dashboard') && location.pathname !== target)){
        navigate(target, { replace: true })
      }
    }
  }, [me, navigate, location.pathname])

  async function refreshAll(){
    try{
      const [m,p,b,d,u,c,s,h,l,n,w,a,cl] = await Promise.all([
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
        api.get('/api/auth/wallet/').catch(()=>({data:null})),
        api.get('/api/auth/announcement/').catch(()=>({data:null, status:204})),
        api.get('/api/auth/clearance/status/').catch(()=>({data:[]})),
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
      setWallet(w.data)
      setAnnouncement(a.data)
      setClearance(Array.isArray(cl.data)? cl.data : [])
      if(m.data?.role === 'CORPER'){
        try{ const r = await api.get('/api/auth/performance/summary/'); setPerf(r.data) }catch(e){}
      }
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
    // Drop empty values and branch for branch-admins (defaults server-side)
    Object.keys(data).forEach(k => { if(data[k] === '') delete data[k] })
    if(me?.role === 'BRANCH') delete data.branch
    try{
      await api.post('/api/auth/corpers/', data)
      await refreshAll(); setStatus('saved:corper')
      e.target.reset()
    }catch(e){ setStatus('error:corper') }
  }

  function enrollBranchOptions(){
    return branches
  }

  function enrollDeptOptions(){
    const bid = Number(enrollBranch || 0)
    if(!bid){
      if(me?.role === 'BRANCH'){
        const b = branches.find(x => x.admin_info && x.admin_info.email === me?.email) || branches[0]
        const targetId = b?.id
        return deps.filter(d => d.branch === targetId)
      }
      return deps
    }
    return deps.filter(d => d.branch === bid)
  }

  function enrollUnitOptions(){
    const did = Number(enrollDept || 0)
    if(did){ return units.filter(u => u.department === did) }
    const bid = Number(enrollBranch || 0)
    if(bid){
      const deptIds = deps.filter(d => d.branch === bid).map(d => d.id)
      return units.filter(u => deptIds.includes(u.department))
    }
    return units
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

  function fundWallet(){ setShowFund(true); setFundAmount('') }

  async function submitFund(){
    const raw = String(fundAmount || '').replace(/,/g, '')
    const amount = Number(raw)
    if(!Number.isFinite(amount) || amount <= 0){ alert('Enter a valid amount'); return }
    try{
      const callback = `${window.location.origin}/dashboard?paystack=1`
      const res = await api.post('/api/auth/wallet/paystack/initialize/', { email: me?.email, amount, callback_url: callback })
      const { authorization_url } = res.data || {}
      if(!authorization_url){ alert('Failed to initialize payment'); return }
      setShowFund(false)
      window.location.href = authorization_url
    }catch(e){ alert('Failed to start payment') }
  }

  // Wallet card + transactions; shared across roles
  function OrgWallet(){
    const bal = wallet?.balance || '0.00'
    const txs = wallet?.transactions || []
    const totals = txs.reduce((acc, t) => {
      const amt = Number(t.total_amount || 0)
      if (t.type === 'CREDIT') acc.credit += amt
      if (t.type === 'DEBIT') acc.debit += amt
      return acc
    }, { credit: 0, debit: 0 })
    const [txPage, setTxPage] = useState(1)
    const [txQuery, setTxQuery] = useState('')
    const filteredTxs = txs.filter(t => {
      const q = txQuery.trim().toLowerCase()
      if(!q) return true
      const hay = [t.description||'', t.reference||'', t.type||'', new Date(t.created_at).toLocaleString()].join(' ').toLowerCase()
      return hay.includes(q)
    })
    const txPageSize = 10
    const txTotalPages = Math.max(1, Math.ceil(filteredTxs.length / txPageSize))
    const txStart = (txPage - 1) * txPageSize
    const pageTxs = filteredTxs.slice(txStart, txStart + txPageSize)
    return (
      <div className="row g-3">
        <div className="col-12 col-lg-4">
          <div className="card shadow-sm"><div className="card-body">
            <div className="text-muted small">Current Balance</div>
            <div className="display-6">₦{Number(bal).toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
            <div className="mt-2 small">
              <div><span className="text-muted">Total Credit:</span> ₦{totals.credit.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
              <div><span className="text-muted">Total Debit:</span> ₦{totals.debit.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
            </div>
            <button className="btn btn-olive mt-3" onClick={fundWallet}>Fund Wallet</button>
          </div></div>
        </div>
        <div className="col-12 col-lg-8">
          <div className="card shadow-sm"><div className="card-body">
            <h6 className="card-title">Transactions</h6>
            <div className="mb-2">
              <input className="form-control form-control-sm" placeholder="Search transactions..." value={txQuery} onChange={(e)=>{ setTxQuery(e.target.value); setTxPage(1) }} />
            </div>
            <div className="table-responsive">
              <table className="table table-sm align-middle">
                <thead>
                  <tr>
                    <th>S/N</th>
                    <th>Date</th>
                    <th>Description</th>
                    <th>Type</th>
                    <th className="text-end">Amount</th>
                    <th className="text-end">VAT</th>
                    <th className="text-end">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {pageTxs.map((t, i) => (
                    <tr key={t.id}>
                      <td>{txStart + i + 1}</td>
                      <td>{new Date(t.created_at).toLocaleString()}</td>
                      <td>{t.description}{t.reference?` (${t.reference})`:''}</td>
                      <td>{t.type}</td>
                      <td className="text-end">₦{Number(t.amount).toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                      <td className="text-end">₦{Number(t.vat_amount).toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                      <td className="text-end">₦{Number(t.total_amount).toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                    </tr>
                  ))}
                  {filteredTxs.length===0 && (
                    <tr><td colSpan="7" className="text-muted">No transactions yet.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            {filteredTxs.length>10 && (
              <div className="d-flex justify-content-between align-items-center mt-2">
                <div className="small text-muted">Page {txPage} of {txTotalPages} · {filteredTxs.length} result(s)</div>
                <div className="btn-group">
                  <button className="btn btn-sm btn-outline-secondary" disabled={txPage===1} onClick={()=>setTxPage(p=>Math.max(1,p-1))}>Prev</button>
                  <button className="btn btn-sm btn-outline-secondary" disabled={txPage===txTotalPages} onClick={()=>setTxPage(p=>Math.min(txTotalPages,p+1))}>Next</button>
                </div>
              </div>
            )}
          </div></div>
        </div>
      </div>
    )
  }

  return (
    <div className="container-fluid p-0">
      {showFund && (
        <div style={{position:'fixed', inset:0, background:'rgba(0,0,0,0.35)', zIndex:1050}} onClick={()=>setShowFund(false)}>
          <div className="card shadow" style={{position:'absolute', top:'20%', left:'50%', transform:'translateX(-50%)', width:'min(420px, 92%)'}} onClick={e=>e.stopPropagation()}>
            <div className="card-header d-flex justify-content-between align-items-center">
              <strong>Fund Wallet</strong>
              <button className="btn btn-sm btn-outline-secondary" onClick={()=>setShowFund(false)}>Close</button>
            </div>
            <div className="card-body">
              <label className="form-label">Amount (NGN)</label>
              <input className="form-control" placeholder="e.g. 5,000" value={fundAmount} onChange={(e)=>setFundAmount(e.target.value)} />
              <div className="d-flex justify-content-end gap-2 mt-3">
                <button className="btn btn-outline-secondary" onClick={()=>setShowFund(false)}>Cancel</button>
                <button className="btn btn-olive" onClick={submitFund}>Continue</button>
              </div>
            </div>
          </div>
        </div>
      )}
      {me?.role==='ORG' && announcement && (
        <div style={{position:'fixed', top:0, left:0, right:0, bottom:0, background:'rgba(0,0,0,0.35)', zIndex:1050}} onClick={()=>setAnnouncement(null)}>
          <div className="card shadow" style={{position:'absolute', top:'15%', left:'50%', transform:'translateX(-50%)', width:'min(520px, 95%)'}} onClick={e=>e.stopPropagation()}>
            <div className="card-header d-flex justify-content-between align-items-center">
              <strong>{announcement.title || 'Notice'}</strong>
              <button className="btn btn-sm btn-outline-secondary" onClick={()=>setAnnouncement(null)}>Close</button>
            </div>
            <div className="card-body">
              <div style={{whiteSpace:'pre-wrap'}}>{announcement.message}</div>
            </div>
          </div>
        </div>
      )}
      <nav className="navbar navbar-light bg-white border-bottom px-3 sticky-top d-flex justify-content-between topnav">
        <div className="d-flex align-items-center">
          {logoUrl ? (
            <img src={logoUrl} alt="Organization Logo" className="org-logo"/>
          ) : (
            <div className="org-logo-placeholder"/>
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
            {(me?.role==='ORG' || me?.role==='BRANCH') && (
              <button className={`btn btn-sm ${activeTab==='structure'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('structure')}>Structure</button>
            )}
            {(me?.role==='ORG' || me?.role==='BRANCH') && (
              <button className={`btn btn-sm ${activeTab==='corpers'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('corpers')}>Corpers Management</button>
            )}
            {(me?.role==='ORG') && (
              <button className={`btn btn-sm ${activeTab==='wallet'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('wallet')}>Wallet</button>
            )}
            {(me?.role==='ORG' || me?.role==='BRANCH') && (
              <button className={`btn btn-sm ${activeTab==='clearance'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('clearance')}>Performance Clearance</button>
            )}
            {(me?.role==='BRANCH') && (
              <>
                <button className={`btn btn-sm ${activeTab==='wallet'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('wallet')}>Wallet</button>
                <button className={`btn btn-sm ${activeTab==='leave'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('leave')}>Leave Management {leaves.filter(l=>l.status==='PENDING').length ? <span className="badge bg-danger ms-1">{leaves.filter(l=>l.status==='PENDING').length}</span> : null}</button>
                <button className={`btn btn-sm ${activeTab==='query'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('query')}>Query Management</button>
                <button className={`btn btn-sm ${activeTab==='report'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('report')}>Report</button>
              </>
            )}
            {(me?.role==='CORPER') && (
              <>
                <button className={`btn btn-sm ${activeTab==='overview'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('overview')}>Home</button>
                <button className={`btn btn-sm ${activeTab==='attendance'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('attendance')}>Attendance</button>
                <button className={`btn btn-sm ${activeTab==='leave'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('leave')}>Leave Management</button>
                <button className={`btn btn-sm ${activeTab==='performance'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('performance')}>Performance Clearance</button>
                <button className={`btn btn-sm ${activeTab==='wallet'?'btn-olive':'btn-outline-secondary'}`} onClick={()=>setActiveTab('wallet')}>Wallet</button>
              </>
            )}
          </div>
        </div>
      </nav>
      <main className="p-3 p-md-4">
        {status==='saved:face-capture' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Face capture saved successfully.</AutoFadeAlert>
        )}
        {status==='saved:attendance' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Attendance marked successfully.</AutoFadeAlert>
        )}
        {status==='saved:wallet-fund' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Wallet funded successfully.</AutoFadeAlert>
        )}
        {status==='error:wallet-fund' && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Payment verification failed.</AutoFadeAlert>
        )}
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
                    <h6 className="card-title">Attendance</h6>
                    <Doughnut data={{
                      labels:['Today','This Month'],
                      datasets:[{ data:[stats?.attendance?.today||0, stats?.attendance?.this_month||0], backgroundColor:['#BDB76B','#556B2F'] }]
                    }} options={{ responsive:true, maintainAspectRatio:false }} />
                  </div></div>
                </div>
              </div>
              )}
              {me?.role!=='CORPER' && (
              <div className="row g-3 mt-1">
                <div className="col-12">
                  <div className="card shadow-sm"><div className="card-body" style={{height:300}}>
                    <h6 className="card-title">Attendance - Last 7 Days</h6>
                    <Bar data={{
                      labels: (stats?.attendance?.last7||[]).map(r=> new Date(r.date).toLocaleDateString()),
                      datasets: [{
                        label: 'Present',
                        data: (stats?.attendance?.last7||[]).map(r=> r.count),
                        backgroundColor: '#BDB76B'
                      }]
                    }} options={{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}} }} />
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
                        <div className="col-6 col-md-3">
                          <label className="form-label">Max Days Late</label>
                          <input className="form-control" type="number" min="0" step="1" name="max_days_late" defaultValue={profile?.max_days_late ?? ''} placeholder="e.g., 3" />
                        </div>
                        <div className="col-6 col-md-3">
                          <label className="form-label">Max Days Absent</label>
                          <input className="form-control" type="number" min="0" step="1" name="max_days_absent" defaultValue={profile?.max_days_absent ?? ''} placeholder="e.g., 2" />
                        </div>
                        <div className="col-12 col-md-3">
                          <label className="form-label">Logo</label>
                          <input className="form-control" type="file" name="logo" accept="image/*" />
                        {/* no preview under choose file as requested */}
                        </div>
                        <div className="col-12 col-md-6">
                          <label className="form-label">Director HR Name</label>
                          <input className="form-control" type="text" name="signatory_name" defaultValue={profile?.signatory_name || ''} placeholder="e.g., Jane Doe" />
                        </div>
                        <div className="col-12 col-md-3">
                          <label className="form-label">Director HR Signature</label>
                          <input className="form-control" type="file" name="signature" accept="image/*" />
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
                {status==='saved:face-capture' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Face capture saved successfully.</AutoFadeAlert>}
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
                            {/* Departments and Units for this branch */}
                            <div className="mt-2">
                              <div className="fw-semibold small text-olive">Departments & Units</div>
                              {deps.filter(d => d.branch === b.id).length === 0 && (
                                <div className="small text-muted">No departments yet.</div>
                              )}
                              {deps.filter(d => d.branch === b.id).map(d => (
                                <div key={d.id} className="small mt-2">
                                  <div className="d-flex align-items-center justify-content-between">
                                    <div className="fw-semibold">{d.name}</div>
                                    <button
                                      className="btn btn-sm btn-outline-secondary"
                                      onClick={() => {
                                        const newName = prompt('Edit department name (leave empty to delete)', d.name)
                                        if(newName === null) return; // cancelled
                                        const trimmed = (newName || '').trim()
                                        if(trimmed === ''){
                                          if(confirm('Delete this department and its units?')){
                                            (async()=>{ try{ await api.delete(`/api/auth/departments/${d.id}/`); await refreshAll() }catch(e){} })()
                                          }
                                        }else{
                                          (async()=>{ try{ await api.put(`/api/auth/departments/${d.id}/`, { name: trimmed, branch: d.branch }); await refreshAll() }catch(e){} })()
                                        }
                                      }}
                                    >Edit</button>
                                  </div>
                                  <div className="text-muted mt-1">
                                    {units.filter(u => u.department === d.id).length === 0 && 'No units'}
                                    {units.filter(u => u.department === d.id).map(u => (
                                      <span key={u.id} className="me-2 d-inline-flex align-items-center">
                                        {u.name}
                                        <button
                                          className="btn btn-sm btn-link text-decoration-none ms-1"
                                          onClick={() => {
                                            const newUnitName = prompt('Edit unit name (leave empty to delete)', u.name)
                                            if(newUnitName === null) return; // cancelled
                                            const trimmed = (newUnitName || '').trim()
                                            if(trimmed === ''){
                                              if(confirm('Delete this unit?')){
                                                (async()=>{ try{ await api.delete(`/api/auth/units/${u.id}/`); await refreshAll() }catch(e){} })()
                                              }
                                            }else{
                                              (async()=>{ try{ await api.put(`/api/auth/units/${u.id}/`, { name: trimmed, department: u.department }); await refreshAll() }catch(e){} })()
                                            }
                                          }}
                                        >Edit</button>
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              ))}
                              {me?.role==='ORG' && (
                                <div className="mt-3">
                                  <div className="small mb-1">Inherit from another branch</div>
                                  <div className="d-flex gap-2">
                                    <select className="form-select form-select-sm" defaultValue="" onChange={(e)=>{ b._copyFrom = e.target.value }} style={{maxWidth: '60%'}}>
                                      <option value="">Select source branch</option>
                                      {branches.filter(x => x.id !== b.id).map(x => (
                                        <option key={x.id} value={x.id}>{x.name}</option>
                                      ))}
                                    </select>
                                    <button className="btn btn-sm btn-outline-secondary" onClick={async()=>{
                                      if(!b._copyFrom){ alert('Select a source branch first'); return }
                                      try{ await api.post(`/api/auth/branches/${b.id}/clone_structure/`, { source: Number(b._copyFrom) }); await refreshAll() }catch(e){}
                                    }}>Copy</button>
                                  </div>
                                </div>
                              )}
                            </div>
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

          {activeTab==='wallet' && (me?.role==='ORG' || me?.role==='BRANCH' || me?.role==='CORPER') && (
            <>
              <h2 className="mb-3 text-olive">Wallet</h2>
              <OrgWallet />
            </>
          )}

          {activeTab==='clearance' && (me?.role==='ORG' || me?.role==='BRANCH') && (
            <>
              <h2 className="mb-3 text-olive">Performance Clearance (Prev. Month)</h2>
              <div className="card shadow-sm">
                <div className="card-body">
                  <div className="row g-2 mb-2">
                    <div className="col-md-4">
                      <input className="form-control form-control-sm" placeholder="Search corpers... (name, code, branch)" value={clQuery} onChange={(e)=>{ setClQuery(e.target.value); setClPage(1) }} />
                    </div>
                  </div>
                  <div className="table-responsive">
                    <table className="table table-sm align-middle">
                      <thead>
                        <tr>
                          <th>S/N</th>
                          <th>Corper</th>
                          <th>State Code</th>
                          <th>Branch</th>
                          <th className="text-end">Absent</th>
                          <th className="text-end">Late</th>
                          <th>Qualified</th>
                          <th>Downloaded</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {(() => {
                          const pageSize = 10
                          const q = clQuery.trim().toLowerCase()
                          const filtered = q ? clearance.filter(r => `${r.full_name} ${r.state_code} ${r.branch}`.toLowerCase().includes(q)) : clearance
                          const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
                          const current = Math.min(clPage, totalPages)
                          if(current !== clPage) setClPage(current)
                          const start = (current - 1) * pageSize
                          const rows = filtered.slice(start, start + pageSize)
                          return <>
                            {rows.map((row, idx) => (
                              <tr key={row.id}>
                                <td>{start + idx + 1}</td>
                                <td>{row.full_name}</td>
                                <td>{row.state_code}</td>
                                <td>{row.branch || '—'}</td>
                                <td className="text-end">{row.absent}</td>
                                <td className="text-end">{row.late}</td>
                                <td>{row.qualified ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span>}</td>
                                <td>{row.downloaded ? <span className="badge bg-primary">Yes</span> : 'No'}</td>
                                <td className="text-end">
                                  {!row.qualified && !row.override && !row.downloaded && (
                                    <button className="btn btn-sm btn-outline-secondary" onClick={async()=>{
                                      try{ await api.post('/api/auth/clearance/approve/', { corper: row.id }); await refreshAll() }catch(e){}
                                    }}>Approve</button>
                                  )}
                                </td>
                              </tr>
                            ))}
                            {filtered.length===0 && (
                              <tr><td colSpan="9" className="text-muted">No corpers found.</td></tr>
                            )}
                          </>
                        })()}
                      </tbody>
                    </table>
                  </div>
                  {(() => {
                    const pageSize = 10
                    const q = clQuery.trim().toLowerCase()
                    const filtered = q ? clearance.filter(r => `${r.full_name} ${r.state_code} ${r.branch}`.toLowerCase().includes(q)) : clearance
                    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
                    const current = Math.min(clPage, totalPages)
                    if(totalPages <= 1) return null
                    return (
                      <div className="d-flex justify-content-between align-items-center mt-2">
                        <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                        <div className="btn-group">
                          <button className="btn btn-sm btn-outline-secondary" disabled={current===1} onClick={()=>setClPage(p=>Math.max(1,p-1))}>Prev</button>
                          <button className="btn btn-sm btn-outline-secondary" disabled={current===totalPages} onClick={()=>setClPage(p=>Math.min(totalPages,p+1))}>Next</button>
                        </div>
                      </div>
                    )
                  })()}
                </div>
              </div>
            </>
          )}

          {activeTab==='structure' && me?.role==='BRANCH' && (
            <>
              <h2 className="mb-3 text-olive">Structure</h2>
              {/* Identify the admin's own branch */}
              {(() => {
                const myBranch = branches.find(x => x.admin_info && x.admin_info.email === me?.email) || branches[0]
                if(!myBranch){ return (<div className="text-muted">No branch assigned.</div>) }
                const myDeps = deps.filter(d => d.branch === myBranch.id)
                const myUnits = units.filter(u => myDeps.some(d => d.id === u.department))
                return (
                  <div className="row g-4">
                    <div className="col-lg-6">
                      <div className="card shadow-sm">
                        <div className="card-body">
                          <h5 className="card-title">My Branch</h5>
                          <div className="mb-2">
                            <div className="fw-semibold">{myBranch.name}</div>
                            <div className="small text-muted">{myBranch.address || '—'}</div>
                            {(myBranch.latitude || myBranch.longitude) && (
                              <div className="small mt-1">Lat: {myBranch.latitude ?? '—'} · Lng: {myBranch.longitude ?? '—'}</div>
                            )}
                          </div>
                          {/* Map to update branch location; includes "Use my current location" control */}
                          <div className="mb-2">
                            <MapPicker
                              value={(myBranch.latitude && myBranch.longitude) ? { lat: myBranch.latitude, lng: myBranch.longitude } : null}
                              onChange={(pos) => { myBranch._newPos = pos }}
                              height={240}
                              zoom={myBranch.latitude && myBranch.longitude ? 14 : 6}
                            />
                          </div>
                          <div className="d-flex gap-2">
                            <button className="btn btn-sm btn-outline-secondary" onClick={()=>{
                              const name = prompt('Branch name', myBranch.name) || myBranch.name;
                              const address = prompt('Address', myBranch.address||'') || '';
                              const latitude = prompt('Latitude', myBranch.latitude ?? '') || '';
                              const longitude = prompt('Longitude', myBranch.longitude ?? '') || '';
                              (async()=>{ try{ await api.put(`/api/auth/branches/${myBranch.id}/`, { name, address, latitude, longitude }); await refreshAll() }catch(e){} })();
                            }}>Edit Branch</button>
                            <button className="btn btn-sm btn-olive" onClick={async()=>{
                              const pos = myBranch._newPos
                              if(!pos){ alert('Pick a location on the map or use the 📍 button first.'); return }
                              try{
                                await api.put(`/api/auth/branches/${myBranch.id}/`, { latitude: pos.lat, longitude: pos.lng, name: myBranch.name, address: myBranch.address||'' })
                                await refreshAll()
                              }catch(e){}
                            }}>Save Location</button>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="col-lg-6">
                      <div className="card shadow-sm">
                        <div className="card-body">
                          <h5 className="card-title">Create Department</h5>
                          <form onSubmit={(e)=>{ e.preventDefault(); setStatus('pending'); const f = new FormData(e.target); const name = f.get('name'); (async()=>{ try{ await api.post('/api/auth/departments/', { branch: myBranch.id, name }); await refreshAll(); setStatus('saved:department'); e.target.reset() }catch(err){ setStatus('error:department') } })(); }}>
                            <input className="form-control mb-2" name="name" placeholder="Department name" required/>
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
                          <h5 className="card-title">Create Unit</h5>
                          <form onSubmit={(e)=>{ e.preventDefault(); setStatus('pending'); const f = new FormData(e.target); const name = f.get('name'); const department = Number(f.get('department')); (async()=>{ try{ await api.post('/api/auth/units/', { name, department }); await refreshAll(); setStatus('saved:unit'); e.target.reset() }catch(err){ setStatus('error:unit') } })(); }}>
                            <select className="form-select mb-2" name="department" required>
                              <option value="">Select Department</option>
                              {myDeps.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                            </select>
                            <input className="form-control mb-2" name="name" placeholder="Unit name" required/>
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
                          <h5 className="card-title">Departments & Units</h5>
                          {myDeps.length === 0 && <div className="text-muted">No departments yet.</div>}
                          {myDeps.map(d => (
                            <div key={d.id} className="small mt-2">
                              <div className="d-flex align-items-center justify-content-between">
                                <div className="fw-semibold">{d.name}</div>
                                <button className="btn btn-sm btn-outline-secondary" onClick={() => {
                                  const newName = prompt('Edit department name (leave empty to delete)', d.name)
                                  if(newName === null) return;
                                  const trimmed = (newName || '').trim()
                                  if(trimmed === ''){
                                    if(confirm('Delete this department and its units?')){
                                      (async()=>{ try{ await api.delete(`/api/auth/departments/${d.id}/`); await refreshAll() }catch(e){} })()
                                    }
                                  }else{
                                    (async()=>{ try{ await api.put(`/api/auth/departments/${d.id}/`, { name: trimmed, branch: d.branch }); await refreshAll() }catch(e){} })()
                                  }
                                }}>Edit</button>
                              </div>
                              <div className="text-muted mt-1">
                                {units.filter(u => u.department === d.id).length === 0 && 'No units'}
                                {units.filter(u => u.department === d.id).map(u => (
                                  <span key={u.id} className="me-2 d-inline-flex align-items-center">
                                    {u.name}
                                    <button className="btn btn-sm btn-link text-decoration-none ms-1" onClick={() => {
                                      const newUnitName = prompt('Edit unit name (leave empty to delete)', u.name)
                                      if(newUnitName === null) return;
                                      const trimmed = (newUnitName || '').trim()
                                      if(trimmed === ''){
                                        if(confirm('Delete this unit?')){
                                          (async()=>{ try{ await api.delete(`/api/auth/units/${u.id}/`); await refreshAll() }catch(e){} })()
                                        }
                                      }else{
                                        (async()=>{ try{ await api.put(`/api/auth/units/${u.id}/`, { name: trimmed, department: u.department }); await refreshAll() }catch(e){} })()
                                      }
                                    }}>Edit</button>
                                  </span>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </>
          )}

          {activeTab==='corpers' && (me?.role==='ORG' || me?.role==='BRANCH') && (
            <>
              <h2 className="mb-3 text-olive">Corpers</h2>
              <div className="card shadow-sm">
                <div className="card-body">
                  <h5 className="card-title">Enroll Corp Member</h5>
                  <form onSubmit={createCorper}>
                    {/* Row 1: Basic details */}
                    <div className="row g-2 align-items-end mb-1">
                      <div className="col-md-4">
                        <label className="form-label">Email</label>
                        <input className="form-control" type="email" name="email" placeholder="corper@example.com" required />
                      </div>
                      <div className="col-md-4">
                        <label className="form-label">Full Name</label>
                        <input className="form-control" name="full_name" placeholder="Surname Firstname Lastname" required />
                      </div>
                      <div className="col-md-2">
                        <label className="form-label">Gender</label>
                        <select className="form-select" name="gender" required>
                          <option value="">Select...</option>
                          <option value="M">Male</option>
                          <option value="F">Female</option>
                          <option value="O">Other</option>
                        </select>
                      </div>
                      <div className="col-md-2">
                        <label className="form-label">Passing Out</label>
                        <input className="form-control" type="date" name="passing_out_date" required />
                      </div>
                    </div>

                    {/* Row 2: Placement */}
                    <div className="row g-2 align-items-end">
                      <div className="col-md-3">
                        <label className="form-label">State Code</label>
                        <input className="form-control" name="state_code" placeholder="AA/00A/0000" required />
                      </div>
                      {me?.role==='ORG' && (
                        <div className="col-md-3">
                          <label className="form-label">Branch</label>
                          <select className="form-select" name="branch" required value={enrollBranch}
                                  onChange={(e)=>{ setEnrollBranch(e.target.value); setEnrollDept('') }}>
                            <option value="">Select branch</option>
                            {enrollBranchOptions().map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                          </select>
                        </div>
                      )}
                      <div className="col-md-3">
                        <label className="form-label">Department (optional)</label>
                        <select className="form-select" name="department" value={enrollDept}
                                onChange={(e)=> setEnrollDept(e.target.value)}>
                          <option value="">Select department</option>
                          {enrollDeptOptions().map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                        </select>
                      </div>
                      <div className="col-md-3">
                        <label className="form-label">Unit (optional)</label>
                        <select className="form-select" name="unit">
                          <option value="">Select unit</option>
                          {enrollUnitOptions().map(u => <option key={u.id} value={u.id}>{u.name}</option>)}
                        </select>
                      </div>
                      <div className="col-12 col-md-2 d-grid">
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
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <div className="small text-muted">Manage and update placements</div>
                  <div style={{minWidth: 260}}>
                    <input className="form-control form-control-sm" placeholder="Search corpers..." value={corperQuery} onChange={(e)=>setCorperQuery(e.target.value)} />
                  </div>
                </div>
                <div className="table-responsive">
                  <table className="table table-sm align-middle">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Full Name</th>
                        <th>Email</th>
                        <th>State Code</th>
                        <th>Branch</th>
                        {(me?.role==='BRANCH' || me?.role==='ORG') && <th>Department</th>}
                        {(me?.role==='BRANCH' || me?.role==='ORG') && <th>Unit</th>}
                        {(me?.role==='BRANCH' || me?.role==='ORG') && <th>Actions</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {(corpers.filter(c=>{
                        const q = corperQuery.trim().toLowerCase(); if(!q) return true;
                        const branchName = branches.find(b=>b.id===c.branch)?.name || ''
                        const hay = `${c.full_name} ${c.email||''} ${c.state_code} ${branchName}`.toLowerCase()
                        return hay.includes(q)
                      })).map((c, idx) => (
                        <tr key={c.id}>
                          <td>{idx+1}</td>
                          <td>{c.full_name}</td>
                          <td>{c.email}</td>
                          <td>{c.state_code}</td>
                          <td>
                            {me?.role==='ORG' ? (
                              <select className="form-select form-select-sm" defaultValue={c.branch || ''} onChange={e=>{ c._newBranch = e.target.value }}>
                                <option value="">—</option>
                                {branches.map(b=> (
                                  <option key={b.id} value={b.id}>{b.name}</option>
                                ))}
                              </select>
                            ) : (
                              (branches.find(b=>b.id===c.branch)?.name) || '—'
                            )}
                          </td>
                          {(me?.role==='BRANCH' || me?.role==='ORG') && (
                            <>
                              <td>
                                <select className="form-select form-select-sm" defaultValue={c.department || ''} onChange={e=>{ c._newDept = e.target.value }}>
                                  <option value="">—</option>
                                  {deps.filter(d=> {
                                    const branchId = Number(c._newBranch || c.branch)
                                    return d.branch === branchId
                                  }).map(d=> (
                                    <option key={d.id} value={d.id}>{d.name}</option>
                                  ))}
                                </select>
                              </td>
                              <td>
                                <select className="form-select form-select-sm" defaultValue={c.unit || ''} onChange={e=>{ c._newUnit = e.target.value }}>
                                  <option value="">—</option>
                                  {units.filter(u=> {
                                    const deptId = Number(c._newDept || c.department)
                                    if (deptId) return u.department === deptId
                                    // No department selected: limit to units whose department is under the selected branch
                                    const branchId = Number(c._newBranch || c.branch)
                                    const deptIdsForBranch = deps.filter(d => d.branch === branchId).map(d => d.id)
                                    return deptIdsForBranch.includes(u.department)
                                  }).map(u=> (
                                    <option key={u.id} value={u.id}>{u.name}</option>
                                  ))}
                                </select>
                              </td>
                              <td className="d-flex gap-2">
                                <button className="btn btn-sm btn-olive" onClick={async()=>{
                                  const payload = {}
                                  if(c._newBranch!==undefined){ payload.branch = c._newBranch || null; payload.department = null; payload.unit = null }
                                  if(c._newDept!==undefined){ payload.department = c._newDept || null; if(!payload.branch && c._newDept) { payload.branch = deps.find(d=>d.id===Number(c._newDept))?.branch } }
                                  if(c._newUnit!==undefined){ payload.unit = c._newUnit || null }
                                  try{
                                    await api.patch(`/api/auth/corpers/${c.id}/`, payload)
                                    setStatus('saved:corper-update')
                                    await refreshAll()
                                  }catch(e){ setStatus('error:corper-update') }
                                }}>Save</button>
                                <a className="btn btn-sm btn-outline-secondary" href={`/api/auth/capture/${c.id}/`} target="_blank" rel="noreferrer">Capture Face</a>
                              </td>
                            </>
                          )}
                        </tr>
                      ))}
                      {corpers.length===0 && (
                        <tr><td colSpan="5" className="text-muted">No corpers enrolled yet.</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
                {status==='saved:corper-update' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Corper updated successfully.</AutoFadeAlert>}
                {status==='error:corper-update' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not update corper.</AutoFadeAlert>}
              </div>
            </div>
            </>
          )}

          {activeTab==='attendance' && (
            <>
              <h2 className="mb-3 text-olive">Attendance</h2>
              {me?.role==='CORPER' && (
                <div className="mb-3">
                  <a className="btn btn-olive" href="/api/auth/attendance/" target="_blank" rel="noreferrer">Mark Attendance</a>
                </div>
              )}
              <div className="row g-3">
                <div className="col-12 col-lg-6">
                  <div className="card shadow-sm"><div className="card-body" style={{height:300}}>
                    <h6 className="card-title">My Attendance (Hours)</h6>
                    <Bar data={{
                      labels: (stats?.attendance?.last7||[]).map(r=> new Date(r.date).toLocaleDateString()),
                      datasets: [{ label: 'Hours', data: (stats?.attendance?.last7||[]).map(r=> r.hours ?? 0), backgroundColor: '#556B2F' }]
                    }} options={{ responsive:true, maintainAspectRatio:false, scales:{ y:{ title:{ display:true, text:'Hours' } } }, plugins:{legend:{display:false}} }} />
                  </div></div>
                </div>
                <div className="col-12 col-lg-6">
                  <div className="card shadow-sm"><div className="card-body" style={{height:300}}>
                    <h6 className="card-title">Today vs Month</h6>
                    <Doughnut data={{
                      labels:['Today','This Month'],
                      datasets:[{ data:[stats?.attendance?.today||0, stats?.attendance?.this_month||0], backgroundColor:['#BDB76B','#556B2F'] }]
                    }} options={{ responsive:true, maintainAspectRatio:false }} />
                  </div></div>
                </div>
              </div>
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
              <div className="mb-2" style={{maxWidth:260}}>
                <input className="form-control form-control-sm" placeholder="Search my leaves..." onChange={(e)=>{ const v=e.target.value.toLowerCase(); const el=document.getElementById('my-leaves-body'); if(!el) return; for(const tr of el.querySelectorAll('tr[data-row]')){ const text=tr.getAttribute('data-hay')||''; tr.style.display = text.includes(v)?'':'none' } }} />
              </div>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead><tr><th>#</th><th>Start</th><th>End</th><th>Status</th></tr></thead>
                  <tbody id="my-leaves-body">
                    {leaves.map((r,idx)=>(<tr key={r.id} data-row data-hay={`${r.start_date} ${r.end_date} ${r.status}`.toLowerCase()}><td>{idx+1}</td><td>{r.start_date}</td><td>{r.end_date}</td><td>{r.status}</td></tr>))}
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
              <div className="mb-2" style={{maxWidth:260}}>
                <input className="form-control form-control-sm" placeholder="Search pending leaves..." onChange={(e)=>{ const v=e.target.value.toLowerCase(); const el=document.getElementById('branch-leaves-body'); if(!el) return; for(const tr of el.querySelectorAll('tr[data-row]')){ const text=tr.getAttribute('data-hay')||''; tr.style.display = text.includes(v)?'':'none' } }} />
              </div>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead><tr><th>#</th><th>Corper</th><th>Start</th><th>End</th><th>Reason</th><th>Actions</th></tr></thead>
                  <tbody id="branch-leaves-body">
                    {leaves.filter(l=>l.status==='PENDING').map((r,idx)=>(
                      <tr key={r.id} data-row data-hay={`${r.corper_name} ${r.start_date} ${r.end_date} ${r.reason}`.toLowerCase()}>
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
              <div className="row g-3">
                <div className="col-6 col-md-3"><div className="card text-center shadow-sm"><div className="card-body"><div className="text-muted small">Period</div><div className="h5 mb-0">{perf?.month || '—'}</div></div></div></div>
                <div className="col-6 col-md-3"><div className="card text-center shadow-sm"><div className="card-body"><div className="text-muted small">Working Days</div><div className="h5 mb-0">{perf?.working_days ?? 0}</div></div></div></div>
                <div className="col-6 col-md-2"><div className="card text-center shadow-sm"><div className="card-body"><div className="text-muted small">Present</div><div className="h5 mb-0">{perf?.present ?? 0}</div></div></div></div>
                <div className="col-6 col-md-2"><div className="card text-center shadow-sm"><div className="card-body"><div className="text-muted small">Absent</div><div className="h5 mb-0">{perf?.absent ?? 0}</div></div></div></div>
                <div className="col-6 col-md-2"><div className="card text-center shadow-sm"><div className="card-body"><div className="text-muted small">Late</div><div className="h5 mb-0">{perf?.late ?? 0}</div></div></div></div>
              </div>
              <div className="mt-3">
                <a className="btn btn-olive" href="/api/auth/performance/clearance/" target="_blank" rel="noreferrer">View Clearance Letter</a>
              </div>
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
