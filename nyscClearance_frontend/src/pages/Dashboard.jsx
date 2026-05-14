// Dashboard: app shell for ORG / BRANCH / CORPER
// - Loads profile, structure, stats, notifications
// - Wallet: funding via Paystack init/verify; modal accepts comma-separated amounts
// - Handles callback params: ?paystack=1&reference=..., ?fund=1
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import api, { ensureCsrf } from '../api/axios'
import { apiHref } from '../api/urls'
import MapPicker from '../components/MapPicker'
import GeofencePicker from '../components/GeofencePicker'
import { Bar, Line } from 'react-chartjs-2'
import AutoFadeAlert from '../components/AutoFadeAlert'
import thankYouAudio from '../assets/thank_you_message.mp3'
import {
  BarChart3,
  Bell,
  Building2,
  FileCheck2,
  FileSearch,
  FileText,
  Layers3,
  LayoutGrid,
  Menu,
  Pencil,
  Search,
  Trash2,
  Users,
  CalendarCheck2,
} from 'lucide-react'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, Filler, Tooltip, Legend } from 'chart.js'
ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Filler, Tooltip, Legend)

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
  const [clSearchOpen, setClSearchOpen] = useState(false)
  const [clPageSize, setClPageSize] = useState(50)
  const [holidays, setHolidays] = useState([])
  const [holidaysAll, setHolidaysAll] = useState([])
  const [leaves, setLeaves] = useState([])
  const [notifications, setNotifications] = useState([])
  const [unreadNotifications, setUnreadNotifications] = useState(0)
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
  const [corperPage, setCorperPage] = useState(1)
  const [corperPageSize, setCorperPageSize] = useState(20)
  const [corperSearchOpen, setCorperSearchOpen] = useState(false)
  const [corperSortKey, setCorperSortKey] = useState('name')
  const [corperSortDir, setCorperSortDir] = useState('asc')
  const [corperFilterBranch, setCorperFilterBranch] = useState('all')
  const [showAddCorper, setShowAddCorper] = useState(false)
  const [corperFormErrors, setCorperFormErrors] = useState({})
  const [editCorper, setEditCorper] = useState(null)
  const [editCorperForm, setEditCorperForm] = useState(null)
  const [selectedCorper, setSelectedCorper] = useState(null)

  useEffect(() => {
    if (!showAddCorper) {
      setCorperFormErrors({})
    }
  }, [showAddCorper])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [structureTab, setStructureTab] = useState('branches')

  const isSaving = status === 'pending'

  const [showAddBranch, setShowAddBranch] = useState(false)
  const [showAddDepartment, setShowAddDepartment] = useState(false)
  const [showAddUnit, setShowAddUnit] = useState(false)
  const [showAddHoliday, setShowAddHoliday] = useState(false)
  const [showEditProfile, setShowEditProfile] = useState(false)
  const [showBranchLocation, setShowBranchLocation] = useState(false)
  const [branchLocationPos, setBranchLocationPos] = useState(null)
  const [editBranch, setEditBranch] = useState(null)
  const [editDepartment, setEditDepartment] = useState(null)
  const [editUnit, setEditUnit] = useState(null)
  const [editHoliday, setEditHoliday] = useState(null)
  const [editHolidayForm, setEditHolidayForm] = useState(null)

  const [newBranchForm, setNewBranchForm] = useState({
    name: '',
    address: '',
    admin_name: '',
    admin_email: '',
    admin_staff_id: '',
    latitude: '',
    longitude: '',
  })
  const [editBranchForm, setEditBranchForm] = useState(null)

  useEffect(() => {
    if (!editBranch) {
      setEditBranchForm(null)
      return
    }
    setEditBranchForm({
      name: editBranch.name || '',
      address: editBranch.address || '',
      admin_name: editBranch.admin_info?.name || '',
      admin_email: editBranch.admin_info?.email || '',
      admin_staff_id: editBranch.admin_info?.staff_id || '',
      latitude: editBranch.latitude ?? '',
      longitude: editBranch.longitude ?? '',
    })
  }, [editBranch])

  useEffect(() => {
    if (!editHoliday) {
      setEditHolidayForm(null)
      return
    }
    setEditHolidayForm({
      title: editHoliday.title || '',
      start_date: editHoliday.start_date || '',
      end_date: editHoliday.end_date || editHoliday.start_date || '',
    })
  }, [editHoliday])

  useEffect(() => {
    if (!editCorper) {
      setEditCorperForm(null)
      return
    }
    setEditCorperForm({
      branch: editCorper.branch || '',
      department: editCorper.department || '',
      unit: editCorper.unit || '',
    })
  }, [editCorper])
  const [structQuery, setStructQuery] = useState('')
  const [structPage, setStructPage] = useState(1)
  const [structPageSize, setStructPageSize] = useState(20)
  const [structSearchOpen, setStructSearchOpen] = useState(false)
  const [structSortKey, setStructSortKey] = useState('name')
  const [structSortDir, setStructSortDir] = useState('asc')
  const [structFilter, setStructFilter] = useState('all')

  const modalOpen = !!(
    showAddBranch ||
    showAddDepartment ||
    showAddUnit ||
    showAddHoliday ||
    showEditProfile ||
    showBranchLocation ||
    editBranch ||
    editDepartment ||
    editUnit ||
    selectedCorper
  )

  useEffect(() => {
    if (!modalOpen) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [modalOpen])

  useEffect(() => {
    if (activeTab === 'structure') {
      setStructQuery('')
      setStructPage(1)
      setStructSearchOpen(false)
      setStructFilter('all')
      // Default sort per section
      setStructSortDir('asc')
      setStructSortKey(structureTab === 'holidays' ? 'date' : structureTab === 'profile' ? 'name' : 'name')
    }
  }, [activeTab, structureTab])

  const chartTheme = useMemo(
    () => ({
      olive: '#556B2F',
      khaki: '#BDB76B',
      oliveSoft: 'rgba(85,107,47,0.18)',
      grid: 'rgba(0,0,0,0.06)',
      text: 'rgba(0,0,0,0.70)',
    }),
    []
  )

  const barOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: chartTheme.text } },
        y: { grid: { color: chartTheme.grid }, ticks: { color: chartTheme.text }, beginAtZero: true },
      },
    }),
    [chartTheme]
  )

  const lineOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { display: false }, ticks: { color: chartTheme.text } },
        y: { grid: { color: chartTheme.grid }, ticks: { color: chartTheme.text }, beginAtZero: true },
      },
    }),
    [chartTheme]
  )

  // First load: ensure CSRF and fetch all data
  useEffect(() => {
    (async () => {
      await ensureCsrf()
      await refreshAll()
    })()
  }, [])

  useEffect(() => {
    if(activeTab !== 'structure') return
    if(me?.role === 'ORG'){
      setStructureTab((t) => (t === 'profile' || t === 'branches' || t === 'departments' || t === 'units' || t === 'holidays' ? t : 'profile'))
    }else if(me?.role === 'BRANCH'){
      setStructureTab((t) => (t === 'branch' || t === 'departments' || t === 'units' || t === 'holidays' ? t : 'branch'))
    }
  }, [activeTab, me?.role])

  // Track unread notifications for corpers (persist across sessions)
  useEffect(() => {
    if (me?.role !== 'CORPER') return
    const list = Array.isArray(notifications) ? notifications : []
    let seenMs = 0
    try{
      const key = `nyscClearance:notifSeen:${me?.email || 'me'}`
      seenMs = Number(localStorage.getItem(key) || 0) || 0
    }catch(e){
      seenMs = 0
    }
    const unread = list.filter(n => {
      const t = new Date(n.created_at).getTime()
      return Number.isFinite(t) && t > seenMs
    }).length
    setUnreadNotifications(unread)
  }, [notifications, me?.role, me?.email])

  useEffect(() => {
    if(me?.role !== 'CORPER') return
    if(activeTab !== 'notifications') return
    const list = Array.isArray(notifications) ? notifications : []
    const latest = list.reduce((mx, n) => {
      const t = new Date(n.created_at).getTime()
      return Number.isFinite(t) ? Math.max(mx, t) : mx
    }, 0)
    try{
      const key = `nyscClearance:notifSeen:${me?.email || 'me'}`
      localStorage.setItem(key, String(latest || Date.now()))
    }catch(e){}
    setUnreadNotifications(0)
  }, [activeTab, me?.role, me?.email, notifications])

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
    if(activeTab === 'clearance'){
      setClPage(1)
    }
  }, [clPageSize, activeTab])

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
      const [m,p,b,d,u,c,s,h,l,n,w,a,cl,ha] = await Promise.all([
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
        api.get('/api/auth/holidays/all/').catch(()=>({data:[]})),
      ])
      setMe(m.data)
      setProfile(p.data)
      setBranches(b.data)
      setDeps(d.data)
      setUnits(u.data)
      setCorpers(c.data)
      setStats(s.data)
      setHolidays(h.data)
      setHolidaysAll(Array.isArray(ha.data) ? ha.data : [])
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
      return true
    }catch(err){ setStatus('error:profile') }
    return false
  }

  const latRef = useRef(null)
  const lngRef = useRef(null)

  async function createBranch(e){
    e.preventDefault(); setStatus('pending')
    const form = new FormData(e.target)
    try{ await api.post('/api/auth/branches/', Object.fromEntries(form)); await refreshAll(); setStatus('saved:branch'); return true }catch(e){ setStatus('error:branch') }
    return false
  }

  async function createDepartment(e){
    e.preventDefault(); setStatus('pending')
    const form = new FormData(e.target)
    try{ await api.post('/api/auth/departments/', Object.fromEntries(form)); await refreshAll(); setStatus('saved:department'); return true }catch(e){ setStatus('error:department') }
    return false
  }

  async function createUnit(e){
    e.preventDefault(); setStatus('pending')
    const form = new FormData(e.target)
    try{ await api.post('/api/auth/units/', Object.fromEntries(form)); await refreshAll(); setStatus('saved:unit'); return true }catch(e){ setStatus('error:unit') }
    return false
  }

  async function createCorper(e){
    e.preventDefault(); setStatus('pending')
    setCorperFormErrors({})
    const fd = new FormData(e.target)
    const data = Object.fromEntries(fd)
    // Drop empty values and branch for branch-admins (defaults server-side)
    Object.keys(data).forEach(k => { if(data[k] === '') delete data[k] })
    if(me?.role === 'BRANCH') delete data.branch
    if(typeof data.state_code === 'string') data.state_code = data.state_code.trim().toUpperCase()
    if(typeof data.full_name === 'string') data.full_name = data.full_name.trim()
    try{
      await api.post('/api/auth/corpers/', data)
      await refreshAll(); setStatus('saved:corper')
      e.target.reset()
      setCorperFormErrors({})
      return true
    }catch(err){
      const payload = err?.response?.data
      if(payload && typeof payload === 'object' && !Array.isArray(payload)){
        setCorperFormErrors(payload)
      }
      const msg = err?.response?.data?.detail
        || Object.values(err?.response?.data || {})?.[0]?.[0]
        || err.message
      setStatus(`error:corper:${msg}`)
    }
    return false
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
    try{ await api.post('/api/auth/holidays/', data); await refreshAll(); setStatus('saved:holiday'); e.target.reset(); return true }catch(e){ setStatus('error:holiday') }
    return false
  }

  async function approveLeave(id){ try{ await api.post(`/api/auth/leaves/${id}/approve/`); await refreshAll() }catch(e){} }
  async function rejectLeave(id){ try{ await api.post(`/api/auth/leaves/${id}/reject/`); await refreshAll() }catch(e){} }

  const logoUrl = profile?.logo || ''

  const navItems = useMemo(() => {
    const items = []
    const add = (key, label, Icon, badge) => items.push({ key, label, Icon, badge })
    add('overview', 'Overview', LayoutGrid)
    if(me?.role === 'ORG' || me?.role === 'BRANCH'){
      add('structure', 'Structure', Layers3)
      add('corpers', 'Corpers', Users)
    }
    if(me?.role === 'ORG'){
      add('wallet', 'Wallet', Building2)
      add('clearance', 'Clearance', LayoutGrid)
    }
    if(me?.role === 'BRANCH'){
      add('wallet', 'Wallet', Building2)
      add('leave', 'Leaves', CalendarCheck2, leaves.filter(l=>l.status==='PENDING').length || null)
      add('query', 'Queries', FileSearch)
      add('report', 'Reports', BarChart3)
      add('clearance', 'Clearance', FileCheck2)
    }
    if(me?.role === 'CORPER'){
      add('attendance', 'Attendance', LayoutGrid)
      add('leave', 'Leaves', LayoutGrid)
      add('performance', 'Clearance', LayoutGrid)
      add('wallet', 'Wallet', Building2)
    }

    add('notifications', 'Notifications', Bell, (me?.role === 'CORPER' ? (unreadNotifications || null) : null))
    return items
  }, [me?.role, leaves, notifications.length, unreadNotifications])

  const tabTitle = useMemo(() => ({
    overview: 'Overview',
    structure: 'Structure',
    corpers: 'Corpers',
    wallet: 'Wallet',
    clearance: 'Performance Clearance',
    leave: 'Leave Management',
    query: 'Query Management',
    report: 'Report',
    attendance: 'Attendance',
    performance: 'Performance Clearance',
    notifications: 'Notifications',
  }), [])

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
    const [txSearchOpen, setTxSearchOpen] = useState(false)
    const [txPageSize, setTxPageSize] = useState(50)
    const [txFilterType, setTxFilterType] = useState('all')
    const [txSortKey, setTxSortKey] = useState('date')
    const [txSortDir, setTxSortDir] = useState('desc')

    const filteredTxs = (() => {
      const q = txSearchOpen ? txQuery.trim().toLowerCase() : ''
      let out = txs
      if(q){
        out = out.filter(t => {
          const hay = [t.description||'', t.reference||'', t.type||'', new Date(t.created_at).toLocaleString()].join(' ').toLowerCase()
          return hay.includes(q)
        })
      }
      if(txFilterType !== 'all'){
        out = out.filter(t => (t.type || '').toUpperCase() === txFilterType)
      }
      const dir = txSortDir === 'asc' ? 1 : -1
      const cmp = (a, b) => {
        if(txSortKey === 'amount') return (Number(a.total_amount||0) - Number(b.total_amount||0)) * dir
        if(txSortKey === 'type') return String(a.type||'').localeCompare(String(b.type||'')) * dir
        // date
        return String(a.created_at||'').localeCompare(String(b.created_at||'')) * dir
      }
      return [...out].sort(cmp)
    })()
    const txTotalPages = Math.max(1, Math.ceil(filteredTxs.length / txPageSize))
    const txStart = (txPage - 1) * txPageSize
    const pageTxs = filteredTxs.slice(txStart, txStart + txPageSize)
    return (
      <div className="row g-3">
        <div className="col-12 col-xxl-4">
          <div className="card shadow-sm dash-card h-100">
            <div className="card-body">
              <div className="dash-card-title mb-2">Wallet</div>
              <div className="text-muted small">Current balance</div>
              <div className="display-6" style={{lineHeight:1.05}}>₦{Number(bal).toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
              <div className="row g-2 mt-3">
                <div className="col-6">
                  <div className="dash-kv">
                    <div className="dash-k">Total credit</div>
                    <div className="dash-v">₦{totals.credit.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                  </div>
                </div>
                <div className="col-6">
                  <div className="dash-kv">
                    <div className="dash-k">Total debit</div>
                    <div className="dash-v">₦{totals.debit.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                  </div>
                </div>
              </div>
              <button className="btn btn-olive mt-3 w-100" onClick={fundWallet}>Fund Wallet</button>
            </div>
          </div>
        </div>

        <div className="col-12 col-xxl-8">
          <div className="card shadow-sm dash-card">
            <div className="card-body">
              <div className="d-flex flex-wrap justify-content-between align-items-center gap-2">
                <div className="dash-card-title mb-0">Transactions</div>
                <div className="d-flex align-items-center gap-2 flex-wrap">
                  <button
                    className={`btn btn-sm ${txSearchOpen ? 'btn-olive' : 'btn-outline-secondary'}`}
                    type="button"
                    aria-label="Search transactions"
                    onClick={() => setTxSearchOpen((v) => !v)}
                  >
                    <Search size={16} />
                  </button>
                  {txSearchOpen && (
                    <div className="dash-table-search">
                      <input
                        className="form-control form-control-sm"
                        placeholder="Search transactions…"
                        value={txQuery}
                        onChange={(e)=>{ setTxQuery(e.target.value); setTxPage(1) }}
                      />
                    </div>
                  )}
                  <select className="form-select form-select-sm" style={{width:140}} value={txFilterType} onChange={(e)=>{ setTxFilterType(e.target.value); setTxPage(1) }} aria-label="Filter">
                    <option value="all">All</option>
                    <option value="CREDIT">Credit</option>
                    <option value="DEBIT">Debit</option>
                  </select>
                  <select className="form-select form-select-sm" style={{width:140}} value={txSortKey} onChange={(e)=>{ setTxSortKey(e.target.value); setTxPage(1) }} aria-label="Sort by">
                    <option value="date">Sort: Date</option>
                    <option value="amount">Sort: Amount</option>
                    <option value="type">Sort: Type</option>
                  </select>
                  <select className="form-select form-select-sm" style={{width:110}} value={txSortDir} onChange={(e)=>{ setTxSortDir(e.target.value); setTxPage(1) }} aria-label="Sort direction">
                    <option value="desc">Desc</option>
                    <option value="asc">Asc</option>
                  </select>
                  <span className="small text-muted">Rows</span>
                  <select className="form-select form-select-sm" style={{width:96}} value={txPageSize} onChange={(e)=>{ setTxPageSize(Number(e.target.value)); setTxPage(1) }}>
                    {[50,100].map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                </div>
              </div>

              <div className="table-responsive mt-2">
              <table className="table table-sm align-middle dash-table">
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
                      <td><div className="text-truncate dash-td-truncate-wide">{t.description}{t.reference?` (${t.reference})`:''}</div></td>
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
            {filteredTxs.length>txPageSize && (
              <div className="d-flex justify-content-between align-items-center mt-2">
                <div className="small text-muted">Page {txPage} of {txTotalPages} · {filteredTxs.length} result(s)</div>
                <div className="btn-group">
                  <button className="btn btn-sm btn-outline-secondary" disabled={txPage===1} onClick={()=>setTxPage(p=>Math.max(1,p-1))}>Prev</button>
                  <button className="btn btn-sm btn-outline-secondary" disabled={txPage===txTotalPages} onClick={()=>setTxPage(p=>Math.min(txTotalPages,p+1))}>Next</button>
                </div>
              </div>
            )}
            </div>
          </div>
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
      <div className="dash-shell">
        <aside className={`dash-sidebar ${sidebarOpen ? 'open' : ''} ${sidebarCollapsed ? 'collapsed' : ''}`}>
          <div className="dash-brand">
            <div className="d-flex align-items-center gap-2">
              {logoUrl ? <img src={logoUrl} alt="Organization Logo" className="org-logo"/> : <div className="org-logo-placeholder"/>}
              <div className="min-w-0">
                <div className="fw-semibold small text-truncate">{me?.name || 'Dashboard'}</div>
                <div className="text-muted small text-truncate">{me?.email}</div>
              </div>
            </div>
            <button className="btn btn-sm btn-outline-secondary d-lg-none" onClick={() => setSidebarOpen(false)} aria-label="Close menu">
              ×
            </button>
          </div>

          <nav className="dash-nav" aria-label="Dashboard navigation">
            {navItems.map(({ key, label, Icon, badge }) => (
              <button
                key={key}
                className={`dash-nav-item ${activeTab === key ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(key)
                  setSidebarOpen(false)
                }}
                type="button"
              >
                <Icon size={18} aria-hidden />
                <span>{label}</span>
                {badge ? <span className="dash-badge">{badge}</span> : null}
              </button>
            ))}
          </nav>
        </aside>

        <div className="dash-main">
          <header className="dash-header">
            <button
              className="btn btn-sm btn-outline-secondary d-lg-none"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
              type="button"
            >
              <Menu size={18} />
            </button>
            <button
              className="btn btn-sm btn-outline-secondary d-none d-lg-inline-flex"
              type="button"
              aria-label="Toggle sidebar"
              onClick={() => setSidebarCollapsed((v) => !v)}
            >
              <Menu size={18} />
            </button>
            <div className="dash-title">
              <div className="fw-semibold">{tabTitle[activeTab] || 'Dashboard'}</div>
              <div className="text-muted small">{me?.role === 'ORG' ? 'Organisation' : me?.role === 'BRANCH' ? 'Admin' : 'Corps Member'}</div>
            </div>

          </header>

          <main className="dash-content">
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
              <div className="row g-3">
                {me?.role !== 'CORPER' && (
                <div className="col-12 col-xxl-9">
                  {me?.role !== 'CORPER' && (
                    <>
                      <div className="dash-section-head" />

                      <div className="row g-3">
                        <div className="col-12 col-sm-6 col-xl-3">
                          <div className="dash-kpi">
                            <div className="dash-kpi-icon"><Users size={18} aria-hidden /></div>
                            <div>
                              <div className="dash-kpi-label">Corpers</div>
                              <div className="dash-kpi-value">{stats?.totals?.corpers ?? 0}</div>
                            </div>
                          </div>
                        </div>
                        <div className="col-12 col-sm-6 col-xl-3">
                          <div className="dash-kpi">
                            <div className="dash-kpi-icon"><Building2 size={18} aria-hidden /></div>
                            <div>
                              <div className="dash-kpi-label">Branches</div>
                              <div className="dash-kpi-value">{stats?.totals?.branches ?? 0}</div>
                            </div>
                          </div>
                        </div>
                        <div className="col-12 col-sm-6 col-xl-3">
                          <div className="dash-kpi">
                            <div className="dash-kpi-icon"><Layers3 size={18} aria-hidden /></div>
                            <div>
                              <div className="dash-kpi-label">Departments</div>
                              <div className="dash-kpi-value">{stats?.totals?.departments ?? 0}</div>
                            </div>
                          </div>
                        </div>
                        <div className="col-12 col-sm-6 col-xl-3">
                          <div className="dash-kpi">
                            <div className="dash-kpi-icon"><LayoutGrid size={18} aria-hidden /></div>
                            <div>
                              <div className="dash-kpi-label">Units</div>
                              <div className="dash-kpi-value">{stats?.totals?.units ?? 0}</div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="dash-section-head mt-4">
                        <div>
                          <div className="dash-section-title">Analytics</div>
                          <div className="dash-section-sub text-muted">Attendance and enrolment breakdown.</div>
                        </div>
                      </div>

                      <div className="row g-3 mt-1">
                        <div className="col-12 col-lg-6">
                          <div className="card shadow-sm dash-card"><div className="card-body" style={{height:340}}>
                            <div className="dash-card-title">Corpers by Branch</div>
                            <Bar data={{
                              labels: (stats?.corpers_by_branch||[]).map(r=>r.branch),
                              datasets: [{
                                label: 'Corpers',
                                data: (stats?.corpers_by_branch||[]).map(r=>r.count),
                                backgroundColor: chartTheme.olive,
                                borderRadius: 10
                              }]
                            }} options={barOptions} />
                          </div></div>
                        </div>
                        <div className="col-12 col-lg-6">
                          <div className="card shadow-sm dash-card"><div className="card-body" style={{height:340}}>
                            <div className="dash-card-title">Attendance trend (Last 7 Days)</div>
                            <Line
                              data={{
                                labels: (stats?.attendance?.last7||[]).map(r=> new Date(r.date).toLocaleDateString()),
                                datasets: [{
                                  label: 'Present',
                                  data: (stats?.attendance?.last7||[]).map(r=> r.count),
                                  borderColor: chartTheme.olive,
                                  backgroundColor: chartTheme.oliveSoft,
                                  tension: 0.35,
                                  fill: true,
                                  pointRadius: 3,
                                  pointHoverRadius: 4,
                                }]
                              }}
                              options={lineOptions}
                            />
                          </div></div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
                )}

                {me?.role === 'CORPER' && (
                  <>
                    <div className="col-12 col-lg-6">
                      <div className="card shadow-sm dash-card"><div className="card-body" style={{height:360}}>
                        <div className="dash-card-title">My Attendance (Last 7 Days)</div>
                        <Bar
                          data={{
                            labels: (stats?.attendance?.last7||[]).map(r=> new Date(r.date).toLocaleDateString()),
                            datasets: [{
                              label: 'Hours',
                              data: (stats?.attendance?.last7||[]).map(r=> r.hours ?? 0),
                              backgroundColor: chartTheme.olive,
                              borderRadius: 10,
                            }]
                          }}
                          options={{
                            ...barOptions,
                            scales: {
                              x: { grid: { display: false }, ticks: { color: chartTheme.text }, title: { display: true, text: 'Day' } },
                              y: { grid: { color: chartTheme.grid }, ticks: { color: chartTheme.text }, beginAtZero: true, title: { display: true, text: 'Hours' } },
                            },
                          }}
                        />
                      </div></div>
                    </div>
                    <div className="col-12 col-lg-6">
                      <div className="card shadow-sm dash-card">
                        <div className="card-body" style={{height:360, overflow:'auto'}}>
                          <div className="dash-card-title">Notifications</div>
                          <div className="dash-feed">
                            {(unreadNotifications ? notifications.slice(0, 4) : []).map((n) => (
                              <div key={n.id} className="dash-feed-item">
                                <div className="fw-semibold">{n.title}</div>
                                <div className="small text-muted">{new Date(n.created_at).toLocaleString()}</div>
                                <div className="small mt-1">{n.message}</div>
                              </div>
                            ))}
                            {unreadNotifications === 0 && <div className="text-muted">No new notifications.</div>}
                          </div>
                          <button className="btn btn-outline-secondary btn-sm mt-3" type="button" onClick={() => setActiveTab('notifications')}>
                            View all
                          </button>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {activeTab==='notifications' && (
            <>
              {(me?.role === 'ORG' || me?.role === 'BRANCH') && (
                <div className="card shadow-sm dash-card mb-3">
                  <div className="card-body">
                    <div className="dash-card-title">Send notification</div>
                    <div className="text-muted small mb-2">
                      {me?.role === 'ORG'
                        ? 'Send to all branches or a specific branch.'
                        : 'Send an update to your branch corpers.'}
                    </div>
                    <form
                      onSubmit={(e) => {
                        e.preventDefault()
                        const data = Object.fromEntries(new FormData(e.target))
                        ;(async () => {
                          try {
                            await api.post('/api/auth/notifications/', data)
                            await refreshAll()
                            setStatus('saved:notification')
                          } catch (err) {
                            setStatus('error:notification')
                          }
                        })()
                        e.target.reset()
                      }}
                    >
                      <div className="row g-2">
                        <div className="col-12">
                          <input className="form-control" name="title" placeholder="Title" required />
                        </div>
                        {me?.role === 'ORG' && (
                          <div className="col-12">
                            <select className="form-select" name="branch">
                              <option value="">All branches</option>
                              {branches.map((b) => (
                                <option key={b.id} value={b.id}>
                                  {b.name}
                                </option>
                              ))}
                            </select>
                          </div>
                        )}
                        <div className="col-12">
                          <textarea className="form-control" name="message" rows="3" placeholder="Message" required />
                        </div>
                        <div className="col-12 d-grid">
                          <button className="btn btn-olive">Send</button>
                        </div>
                      </div>
                    </form>
                  </div>
                </div>
              )}

              {status === 'saved:notification' && (
                <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Notification sent.</AutoFadeAlert>
              )}
              {status === 'error:notification' && (
                <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Failed to send notification.</AutoFadeAlert>
              )}

              {me?.role === 'CORPER' && (
                <div className="card shadow-sm dash-card">
                  <div className="card-body">
                    <div className="dash-card-title">All notifications</div>
                    <div className="dash-feed">
                      {notifications.map((n) => (
                        <div key={n.id} className="dash-feed-item">
                          <div className="fw-semibold">{n.title}</div>
                          <div className="small text-muted">{new Date(n.created_at).toLocaleString()}</div>
                          <div className="small mt-1">{n.message}</div>
                        </div>
                      ))}
                      {notifications.length === 0 && <div className="text-muted">No notifications yet.</div>}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {activeTab==='structure' && me?.role==='ORG' && (
            <>
              <h2 className="mb-3 text-olive">Structure</h2>

              <div className="dash-struct-nav mb-3">
                <button className={`dash-struct-item ${structureTab==='profile'?'active':''}`} type="button" onClick={()=>setStructureTab('profile')}>Organisation Profile</button>
                <button className={`dash-struct-item ${structureTab==='branches'?'active':''}`} type="button" onClick={()=>setStructureTab('branches')}>Branches</button>
                <button className={`dash-struct-item ${structureTab==='departments'?'active':''}`} type="button" onClick={()=>setStructureTab('departments')}>Departments</button>
                <button className={`dash-struct-item ${structureTab==='units'?'active':''}`} type="button" onClick={()=>setStructureTab('units')}>Units</button>
                <button className={`dash-struct-item ${structureTab==='holidays'?'active':''}`} type="button" onClick={()=>setStructureTab('holidays')}>Holidays</button>
              </div>

              <div className="card shadow-sm dash-card">
                <div className="card-body">
                  <div className="d-flex justify-content-between align-items-center gap-2">
                    <div className="dash-card-title mb-0">
                      {structureTab==='branches' ? 'Branches' : structureTab==='departments' ? 'Departments' : structureTab==='units' ? 'Units' : structureTab==='holidays' ? 'Holidays' : 'Organisation Profile'}
                    </div>
                    <div className="d-flex gap-2">
                      {structureTab==='branches' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddBranch(true)}>Add Branch</button>}
                      {structureTab==='departments' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddDepartment(true)}>Add Department</button>}
                      {structureTab==='units' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddUnit(true)}>Add Unit</button>}
                      {structureTab==='holidays' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddHoliday(true)}>Add Holiday</button>}
                      {structureTab==='profile' && <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setShowEditProfile(true)}>Edit Profile</button>}
                    </div>
                  </div>

                  {structureTab !== 'profile' && (
                  <div className="d-flex flex-wrap gap-2 justify-content-between align-items-center mt-3">
                    <div className="d-flex align-items-center gap-2 flex-wrap">
                      <button
                        className={`btn btn-sm ${structSearchOpen ? 'btn-olive' : 'btn-outline-secondary'}`}
                        type="button"
                        aria-label="Search"
                        onClick={() => setStructSearchOpen((v) => !v)}
                      >
                        <Search size={16} />
                      </button>
                      {structSearchOpen && (
                        <div className="dash-table-search">
                          <input
                            className="form-control form-control-sm"
                            placeholder="Search…"
                            value={structQuery}
                            onChange={(e) => {
                              setStructQuery(e.target.value)
                              setStructPage(1)
                            }}
                          />
                        </div>
                      )}

                      {structureTab === 'holidays' && (
                        <select
                          className="form-select form-select-sm"
                          style={{ width: 140 }}
                          value={structFilter}
                          onChange={(e) => {
                            setStructFilter(e.target.value)
                            setStructPage(1)
                          }}
                          aria-label="Filter"
                        >
                          <option value="all">All</option>
                          <option value="manual">Manual</option>
                          <option value="auto">Auto (NG)</option>
                        </select>
                      )}

                      <select
                        className="form-select form-select-sm"
                        style={{ width: 140 }}
                        value={structSortKey}
                        onChange={(e) => {
                          setStructSortKey(e.target.value)
                          setStructPage(1)
                        }}
                        aria-label="Sort by"
                      >
                        {structureTab === 'branches' && (
                          <>
                            <option value="name">Sort: Name</option>
                            <option value="address">Sort: Address</option>
                          </>
                        )}
                        {structureTab === 'departments' && (
                          <>
                            <option value="name">Sort: Name</option>
                            <option value="branch">Sort: Branch</option>
                          </>
                        )}
                        {structureTab === 'units' && (
                          <>
                            <option value="name">Sort: Name</option>
                            <option value="department">Sort: Department</option>
                          </>
                        )}
                        {structureTab === 'holidays' && (
                          <>
                            <option value="date">Sort: Date</option>
                            <option value="title">Sort: Title</option>
                            <option value="type">Sort: Type</option>
                          </>
                        )}
                        {structureTab === 'profile' && <option value="name">Sort: Name</option>}
                      </select>

                      <select
                        className="form-select form-select-sm"
                        style={{ width: 110 }}
                        value={structSortDir}
                        onChange={(e) => {
                          setStructSortDir(e.target.value)
                          setStructPage(1)
                        }}
                        aria-label="Sort direction"
                      >
                        <option value="asc">Asc</option>
                        <option value="desc">Desc</option>
                      </select>
                    </div>
                    <div className="d-flex align-items-center gap-2">
                      <span className="small text-muted">Rows</span>
                      <select
                        className="form-select form-select-sm"
                        style={{ width: 96 }}
                        value={structPageSize}
                        onChange={(e) => {
                          setStructPageSize(Number(e.target.value))
                          setStructPage(1)
                        }}
                      >
                        {[20, 50, 100].map((n) => (
                          <option key={n} value={n}>
                            {n}
                          </option>
                        ))}
                      </select>
                      <span className="small text-muted">Page {structPage}</span>
                    </div>
                  </div>
                  )}

                  <div className={structureTab === 'profile' ? 'mt-3' : 'table-responsive mt-2'}>
                    {structureTab==='branches' && (
                      <table className="table table-sm align-middle dash-table">
                        <thead><tr><th>Name</th><th>Address</th><th>Admin Email</th><th></th></tr></thead>
                        <tbody>
                          {(() => {
                            const q = structSearchOpen ? structQuery.trim().toLowerCase() : ''
                            let filtered = q
                              ? branches.filter((b) => `${b.name} ${b.address || ''} ${b.admin_info?.email || ''}`.toLowerCase().includes(q))
                              : branches
                            const cmp = (a, b) => {
                              const dir = structSortDir === 'desc' ? -1 : 1
                              const av = (structSortKey === 'address' ? (a.address || '') : a.name || '').toLowerCase()
                              const bv = (structSortKey === 'address' ? (b.address || '') : b.name || '').toLowerCase()
                              return av.localeCompare(bv) * dir
                            }
                            filtered = [...filtered].sort(cmp)
                            const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                            const current = Math.min(structPage, totalPages)
                            if (current !== structPage) setStructPage(current)
                            const start = (current - 1) * structPageSize
                            const rows = filtered.slice(start, start + structPageSize)

                            return (
                              <>
                                {rows.map((b) => (
                            <tr key={b.id}>
                              <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{b.name}</div></td>
                              <td><div className="text-truncate dash-td-truncate">{b.address || '—'}</div></td>
                              <td><div className="text-truncate dash-td-truncate">{b.admin_info?.email || '—'}</div></td>
                              <td className="text-end">
                                <div className="btn-group">
                                  <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditBranch(b)} aria-label="Edit branch">
                                    <Pencil size={16} />
                                  </button>
                                  <button
                                    className="btn btn-sm btn-outline-danger"
                                    type="button"
                                    onClick={async () => {
                                      if (!confirm(`Delete branch "${b.name}"? This may remove related departments/units.`)) return
                                      try {
                                        await api.delete(`/api/auth/branches/${b.id}/`)
                                        await refreshAll()
                                      } catch (e) {}
                                    }}
                                    aria-label="Delete branch"
                                  >
                                    <Trash2 size={16} />
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ))}
                                {filtered.length===0 && <tr><td colSpan="4" className="text-muted">No branches found.</td></tr>}
                                {filtered.length>0 && (
                                  <tr>
                                    <td colSpan="4">
                                      <div className="d-flex justify-content-between align-items-center">
                                        <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                        <div className="btn-group">
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructPage(p=>Math.max(1,p-1))}>Prev</button>
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                        </div>
                                      </div>
                                    </td>
                                  </tr>
                                )}
                              </>
                            )
                          })()}
                        </tbody>
                      </table>
                    )}

                    {structureTab==='departments' && (
                      <table className="table table-sm align-middle dash-table">
                        <thead><tr><th>Name</th><th>Branch</th><th></th></tr></thead>
                        <tbody>
                          {(() => {
                            const q = structSearchOpen ? structQuery.trim().toLowerCase() : ''
                            let filtered = q
                              ? deps.filter((d) => `${d.name} ${branches.find(b=>b.id===d.branch)?.name || ''}`.toLowerCase().includes(q))
                              : deps
                            const cmp = (a, b) => {
                              const dir = structSortDir === 'desc' ? -1 : 1
                              const av = (structSortKey === 'branch' ? (branches.find(x=>x.id===a.branch)?.name || '') : a.name || '').toLowerCase()
                              const bv = (structSortKey === 'branch' ? (branches.find(x=>x.id===b.branch)?.name || '') : b.name || '').toLowerCase()
                              return av.localeCompare(bv) * dir
                            }
                            filtered = [...filtered].sort(cmp)
                            const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                            const current = Math.min(structPage, totalPages)
                            if (current !== structPage) setStructPage(current)
                            const start = (current - 1) * structPageSize
                            const rows = filtered.slice(start, start + structPageSize)
                            return (
                              <>
                                {rows.map((d) => (
                            <tr key={d.id}>
                              <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{d.name}</div></td>
                              <td><div className="text-truncate dash-td-truncate">{branches.find(b=>b.id===d.branch)?.name || '—'}</div></td>
                              <td className="text-end">
                                <div className="btn-group">
                                  <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditDepartment(d)} aria-label="Edit department">
                                    <Pencil size={16} />
                                  </button>
                                  <button
                                    className="btn btn-sm btn-outline-danger"
                                    type="button"
                                    onClick={async () => {
                                      if (!confirm(`Delete department "${d.name}"? This may remove related units.`)) return
                                      try {
                                        await api.delete(`/api/auth/departments/${d.id}/`)
                                        await refreshAll()
                                      } catch (e) {}
                                    }}
                                    aria-label="Delete department"
                                  >
                                    <Trash2 size={16} />
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ))}
                                {filtered.length===0 && <tr><td colSpan="3" className="text-muted">No departments found.</td></tr>}
                                {filtered.length>0 && (
                                  <tr>
                                    <td colSpan="3">
                                      <div className="d-flex justify-content-between align-items-center">
                                        <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                        <div className="btn-group">
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructPage(p=>Math.max(1,p-1))}>Prev</button>
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                        </div>
                                      </div>
                                    </td>
                                  </tr>
                                )}
                              </>
                            )
                          })()}
                        </tbody>
                      </table>
                    )}

                    {structureTab==='units' && (
                      <table className="table table-sm align-middle dash-table">
                        <thead><tr><th>Name</th><th>Department</th><th></th></tr></thead>
                        <tbody>
                          {(() => {
                            const q = structSearchOpen ? structQuery.trim().toLowerCase() : ''
                            let filtered = q
                              ? units.filter((u) => `${u.name} ${deps.find(d=>d.id===u.department)?.name || ''}`.toLowerCase().includes(q))
                              : units
                            const cmp = (a, b) => {
                              const dir = structSortDir === 'desc' ? -1 : 1
                              const av = (structSortKey === 'department' ? (deps.find(x=>x.id===a.department)?.name || '') : a.name || '').toLowerCase()
                              const bv = (structSortKey === 'department' ? (deps.find(x=>x.id===b.department)?.name || '') : b.name || '').toLowerCase()
                              return av.localeCompare(bv) * dir
                            }
                            filtered = [...filtered].sort(cmp)
                            const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                            const current = Math.min(structPage, totalPages)
                            if (current !== structPage) setStructPage(current)
                            const start = (current - 1) * structPageSize
                            const rows = filtered.slice(start, start + structPageSize)
                            return (
                              <>
                                {rows.map((u) => (
                            <tr key={u.id}>
                              <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{u.name}</div></td>
                              <td><div className="text-truncate dash-td-truncate">{deps.find(d=>d.id===u.department)?.name || '—'}</div></td>
                              <td className="text-end">
                                <div className="btn-group">
                                  <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditUnit(u)} aria-label="Edit unit">
                                    <Pencil size={16} />
                                  </button>
                                  <button
                                    className="btn btn-sm btn-outline-danger"
                                    type="button"
                                    onClick={async () => {
                                      if (!confirm(`Delete unit "${u.name}"?`)) return
                                      try {
                                        await api.delete(`/api/auth/units/${u.id}/`)
                                        await refreshAll()
                                      } catch (e) {}
                                    }}
                                    aria-label="Delete unit"
                                  >
                                    <Trash2 size={16} />
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ))}
                                {filtered.length===0 && <tr><td colSpan="3" className="text-muted">No units found.</td></tr>}
                                {filtered.length>0 && (
                                  <tr>
                                    <td colSpan="3">
                                      <div className="d-flex justify-content-between align-items-center">
                                        <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                        <div className="btn-group">
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructPage(p=>Math.max(1,p-1))}>Prev</button>
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                        </div>
                                      </div>
                                    </td>
                                  </tr>
                                )}
                              </>
                            )
                          })()}
                        </tbody>
                      </table>
                    )}

                    {structureTab==='holidays' && (
                      <table className="table table-sm align-middle dash-table">
                        <thead><tr><th>Title</th><th>Date</th><th>Type</th><th></th></tr></thead>
                        <tbody>
                          {(() => {
                            const q = structSearchOpen ? structQuery.trim().toLowerCase() : ''
                            const base = holidaysAll.length ? holidaysAll : holidays
                            let filtered = q
                              ? base.filter((h) => `${h.title} ${h.start_date} ${h.end_date} ${h.source||''}`.toLowerCase().includes(q))
                              : base

                            if(structFilter !== 'all'){
                              filtered = filtered.filter((h) => {
                                const isAuto = h.source === 'NATIONAL' || h.deletable === false
                                return structFilter === 'auto' ? isAuto : !isAuto
                              })
                            }

                            const dir = structSortDir === 'desc' ? -1 : 1
                            const cmp = (a, b) => {
                              const typeLabel = (h) => (h.source === 'NATIONAL' || h.deletable === false ? 'auto' : 'manual')
                              const getVal = (h) => {
                                if(structSortKey === 'title') return (h.title || '').toLowerCase()
                                if(structSortKey === 'type') return typeLabel(h)
                                // date
                                return (h.start_date || '').toLowerCase()
                              }
                              return getVal(a).localeCompare(getVal(b)) * dir
                            }
                            filtered = [...filtered].sort(cmp)
                            const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                            const current = Math.min(structPage, totalPages)
                            if (current !== structPage) setStructPage(current)
                            const start = (current - 1) * structPageSize
                            const rows = filtered.slice(start, start + structPageSize)
                            return (
                              <>
                                {rows.map((h) => (
                                  <tr key={h.id}>
                                    <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{h.title}</div></td>
                                    <td>{h.start_date}{h.end_date && h.end_date !== h.start_date ? ` → ${h.end_date}` : ''}</td>
                                    <td>
                                      {h.source === 'NATIONAL' || h.deletable === false ? (
                                        <span className="badge bg-secondary">Auto (NG)</span>
                                      ) : (
                                        <span className="badge bg-olive">Manual</span>
                                      )}
                                    </td>
                                    <td className="text-end">
                                      {h.deletable === false || h.source === 'NATIONAL' ? (
                                        <span className="badge bg-secondary">Auto</span>
                                      ) : (
                                        <div className="btn-group">
                                          <button
                                            className="btn btn-sm btn-outline-secondary"
                                            type="button"
                                            aria-label="Edit holiday"
                                            onClick={() => setEditHoliday(h)}
                                          >
                                            <Pencil size={16} />
                                          </button>
                                          <button
                                            className="btn btn-sm btn-outline-danger"
                                            type="button"
                                            aria-label="Delete holiday"
                                            onClick={async()=>{
                                              if(!confirm(`Delete holiday "${h.title}"?`)) return
                                              try{ await api.delete(`/api/auth/holidays/${h.id}/`); await refreshAll() }catch(e){}
                                            }}
                                          >
                                            <Trash2 size={16} />
                                          </button>
                                        </div>
                                      )}
                                    </td>
                                  </tr>
                                ))}
                                {filtered.length===0 && <tr><td colSpan="4" className="text-muted">No holidays found.</td></tr>}
                                {filtered.length>0 && (
                                  <tr>
                                    <td colSpan="4">
                                      <div className="d-flex justify-content-between align-items-center">
                                        <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                        <div className="btn-group">
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructPage(p=>Math.max(1,p-1))}>Prev</button>
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                        </div>
                                      </div>
                                    </td>
                                  </tr>
                                )}
                              </>
                            )
                          })()}
                        </tbody>
                      </table>
                    )}

                    {structureTab==='profile' && (
                      <div className="dash-profile">
                        <div className="dash-profile-head">
                          <div />
                        </div>

                        <div className="row g-3 mt-1">
                          <div className="col-12 col-lg-6">
                            <div className="dash-profile-card">
                              <div className="dash-profile-card-title">Branding & Sign-off</div>
                              <div className="dash-kv-grid">
                                <div className="dash-kv"><div className="dash-k">Director HR</div><div className="dash-v">{profile?.signatory_name || '—'}</div></div>
                                <div className="dash-kv"><div className="dash-k">Logo</div><div className="dash-v">{profile?.logo ? 'Uploaded' : 'Not uploaded'}</div></div>
                                <div className="dash-kv"><div className="dash-k">Signature</div><div className="dash-v">{profile?.signature ? 'Uploaded' : 'Not uploaded'}</div></div>
                              </div>
                              <div className="dash-profile-previews">
                                <div className="dash-preview">
                                  <div className="dash-preview-label">Logo</div>
                                  {profile?.logo ? (
                                    <img className="dash-preview-img" src={profile.logo} alt="Organisation logo" />
                                  ) : (
                                    <div className="dash-preview-empty">No logo</div>
                                  )}
                                </div>
                                <div className="dash-preview">
                                  <div className="dash-preview-label">Signature</div>
                                  {profile?.signature ? (
                                    <img className="dash-preview-img" src={profile.signature} alt="Director signature" />
                                  ) : (
                                    <div className="dash-preview-empty">No signature</div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>

                          <div className="col-12 col-lg-6">
                            <div className="dash-profile-card">
                              <div className="dash-profile-card-title">Attendance Rules</div>
                              <div className="dash-kv-grid">
                                <div className="dash-kv"><div className="dash-k">Late time</div><div className="dash-v">{profile?.late_time || '—'}</div></div>
                                <div className="dash-kv"><div className="dash-k">Closing time</div><div className="dash-v">{profile?.closing_time || '—'}</div></div>
                                <div className="dash-kv"><div className="dash-k">Max late days</div><div className="dash-v">{profile?.max_days_late ?? '—'}</div></div>
                                <div className="dash-kv"><div className="dash-k">Max absent days</div><div className="dash-v">{profile?.max_days_absent ?? '—'}</div></div>
                              </div>
                            </div>
                          </div>

                          {/* Organisation coordinates are managed at the admin/branch level */}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {showAddBranch && (
                <div className="dash-modal" onClick={() => setShowAddBranch(false)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Add Branch</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowAddBranch(false)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>How it works</h6>
                          <p>Create a branch office so you can group departments, units, and corps members.</p>
                          <ul>
                            <li>Optionally invite an admin by email.</li>
                            <li>Set coordinates if you use location-based attendance.</li>
                            <li>You can edit details later from the table.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form
                            onSubmit={async (e) => {
                              e.preventDefault()
                              setStatus('pending')
                              try {
                                await api.post('/api/auth/branches/', {
                                  ...newBranchForm,
                                  latitude: newBranchForm.latitude === '' ? null : Number(newBranchForm.latitude),
                                  longitude: newBranchForm.longitude === '' ? null : Number(newBranchForm.longitude),
                                })
                                await refreshAll()
                                setStatus('saved:branch')
                                setShowAddBranch(false)
                                setNewBranchForm({
                                  name: '',
                                  address: '',
                                  admin_name: '',
                                  admin_email: '',
                                  admin_staff_id: '',
                                  latitude: '',
                                  longitude: '',
                                })
                              } catch (err) {
                                setStatus('error:branch')
                              }
                            }}
                          >
                            <label className="form-label">Branch name</label>
                            <input
                              className="form-control mb-2"
                              value={newBranchForm.name}
                              onChange={(e) => setNewBranchForm((p) => ({ ...p, name: e.target.value }))}
                              placeholder="e.g., Head Office"
                              required
                            />

                            <GeofencePicker
                              address={newBranchForm.address}
                              onAddressChange={(v) => setNewBranchForm((p) => ({ ...p, address: v }))}
                              lat={newBranchForm.latitude}
                              lng={newBranchForm.longitude}
                              onLatLngChange={({ lat, lng }) =>
                                setNewBranchForm((p) => ({ ...p, latitude: String(lat), longitude: String(lng) }))
                              }
                            />

                            <div className="row g-2 mt-2">
                              <div className="col-md-4">
                                <label className="form-label">Admin Name</label>
                                <input
                                  className="form-control"
                                  value={newBranchForm.admin_name}
                                  onChange={(e) => setNewBranchForm((p) => ({ ...p, admin_name: e.target.value }))}
                                  placeholder="Optional"
                                />
                              </div>
                              <div className="col-md-5">
                                <label className="form-label">Admin Email</label>
                                <input
                                  className="form-control"
                                  type="email"
                                  value={newBranchForm.admin_email}
                                  onChange={(e) => setNewBranchForm((p) => ({ ...p, admin_email: e.target.value }))}
                                  placeholder="Optional"
                                />
                              </div>
                              <div className="col-md-3">
                                <label className="form-label">Staff ID</label>
                                <input
                                  className="form-control"
                                  value={newBranchForm.admin_staff_id}
                                  onChange={(e) => setNewBranchForm((p) => ({ ...p, admin_staff_id: e.target.value }))}
                                  placeholder="Optional"
                                />
                              </div>
                            </div>

                            <div className="dash-modal-actions">
                              <div className="d-grid">
                                <button className="btn btn-olive" disabled={isSaving}>
                                  {isSaving ? 'Adding…' : 'Add Branch'}
                                </button>
                              </div>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {showAddDepartment && (
                <div className="dash-modal" onClick={() => setShowAddDepartment(false)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Add Department</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowAddDepartment(false)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Flow</h6>
                          <p>Departments sit under a branch. Units can then be created under a department.</p>
                          <ul>
                            <li>Select the branch first.</li>
                            <li>Enter a department name.</li>
                            <li>Use Units to add sub-teams.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{ const ok = await createDepartment(e); if(ok) setShowAddDepartment(false) }}>
                            <label className="form-label">Branch</label>
                            <select className="form-select mb-2" name="branch" required>
                              <option value="">Select Branch</option>
                              {branches.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                            </select>
                            <label className="form-label">Department name</label>
                            <input className="form-control mb-3" name="name" placeholder="e.g., HR" required/>
                            <div className="d-grid">
                              <button className="btn btn-olive" disabled={isSaving}>
                                {isSaving ? 'Adding…' : 'Add Department'}
                              </button>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {showAddUnit && (
                <div className="dash-modal" onClick={() => setShowAddUnit(false)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Add Unit</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowAddUnit(false)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Flow</h6>
                          <p>Units are optional sub-groups under a department.</p>
                          <ul>
                            <li>Select a department.</li>
                            <li>Enter a unit name.</li>
                            <li>Assign corps members later during enrolment.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{ const ok = await createUnit(e); if(ok) setShowAddUnit(false) }}>
                            <label className="form-label">Department</label>
                            <select className="form-select mb-2" name="department" required>
                              <option value="">Select Department</option>
                              {deps.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                            </select>
                            <label className="form-label">Unit name</label>
                            <input className="form-control mb-3" name="name" placeholder="e.g., Recruitment" required/>
                            <div className="d-grid">
                              <button className="btn btn-olive" disabled={isSaving}>
                                {isSaving ? 'Adding…' : 'Add Unit'}
                              </button>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {showAddHoliday && (
                <div className="dash-modal" onClick={() => setShowAddHoliday(false)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Add Holiday</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowAddHoliday(false)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>What this affects</h6>
                          <p>Holidays help your attendance and clearance calculations stay accurate.</p>
                          <ul>
                            <li>Select start and end dates.</li>
                            <li>Multi-day holidays are supported.</li>
                            <li>Corps members won’t be penalized on holidays.</li>
                            <li>Synced national holidays appear as Auto (NG) in the Holidays table.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{ const ok = await createHoliday(e); if(ok) setShowAddHoliday(false) }}>
                            <div className="row g-2">
                              <div className="col-12">
                                <label className="form-label">Holiday title</label>
                                <input className="form-control" name="title" placeholder="e.g., Public Holiday" required/>
                              </div>
                              <div className="col-md-6">
                                <label className="form-label">Start date</label>
                                <input className="form-control" type="date" name="start_date" required/>
                              </div>
                              <div className="col-md-6">
                                <label className="form-label">End date</label>
                                <input className="form-control" type="date" name="end_date" required/>
                              </div>
                            </div>
                            <div className="d-grid mt-3">
                              <button className="btn btn-olive" disabled={isSaving}>
                                {isSaving ? 'Adding…' : 'Add Holiday'}
                              </button>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {editHoliday && editHolidayForm && (
                <div className="dash-modal" onClick={() => setEditHoliday(null)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Edit Holiday</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditHoliday(null)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Edit flow</h6>
                          <p>Update the holiday details. National holidays are synced automatically and cannot be edited.</p>
                          <ul>
                            <li>Update title and date range.</li>
                            <li>Save to apply immediately.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form
                            onSubmit={async (e) => {
                              e.preventDefault()
                              setStatus('pending')
                              try{
                                await api.put(`/api/auth/holidays/${editHoliday.id}/`, {
                                  title: editHolidayForm.title,
                                  start_date: editHolidayForm.start_date,
                                  end_date: editHolidayForm.end_date,
                                })
                                await refreshAll()
                                setStatus('saved:holiday')
                                setEditHoliday(null)
                              }catch(err){ setStatus('error:holiday') }
                            }}
                          >
                            <div className="row g-2">
                              <div className="col-12">
                                <label className="form-label">Holiday title</label>
                                <input
                                  className="form-control"
                                  value={editHolidayForm.title}
                                  onChange={(e)=>setEditHolidayForm(p=>({ ...p, title: e.target.value }))}
                                  required
                                />
                              </div>
                              <div className="col-md-6">
                                <label className="form-label">Start date</label>
                                <input
                                  className="form-control"
                                  type="date"
                                  value={editHolidayForm.start_date}
                                  onChange={(e)=>setEditHolidayForm(p=>({ ...p, start_date: e.target.value }))}
                                  required
                                />
                              </div>
                              <div className="col-md-6">
                                <label className="form-label">End date</label>
                                <input
                                  className="form-control"
                                  type="date"
                                  value={editHolidayForm.end_date}
                                  onChange={(e)=>setEditHolidayForm(p=>({ ...p, end_date: e.target.value }))}
                                  required
                                />
                              </div>
                            </div>
                            <div className="dash-modal-actions">
                              <div className="d-grid">
                                <button className="btn btn-olive" disabled={isSaving}>
                                  {isSaving ? 'Saving…' : 'Save changes'}
                                </button>
                              </div>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {showEditProfile && (
                <div className="dash-modal" onClick={() => setShowEditProfile(false)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Edit Organisation Profile</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowEditProfile(false)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Profile setup</h6>
                          <p>These settings affect attendance rules and the generated clearance letters.</p>
                          <ul>
                            <li>Set late/closing times and thresholds.</li>
                            <li>Upload logo/signature used on letters.</li>
                            <li>Set coordinates for location controls.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{ const ok = await saveProfile(e); if(ok) setShowEditProfile(false) }} encType="multipart/form-data">
                            <div className="dash-form-section">
                              <div className="dash-form-title">Branding & Sign-off</div>
                              <div className="row g-2">
                                <div className="col-12">
                                  <label className="form-label">Director HR Name</label>
                                  <input className="form-control" type="text" name="signatory_name" defaultValue={profile?.signatory_name || ''} />
                                </div>
                                <div className="col-12">
                                  <label className="form-label">Logo</label>
                                  <input className="form-control" type="file" name="logo" accept="image/*" />
                                  <div className="form-text">Used on generated clearance letters.</div>
                                </div>
                                <div className="col-12">
                                  <label className="form-label">Signature</label>
                                  <input className="form-control" type="file" name="signature" accept="image/*" />
                                  <div className="form-text">Director HR signature for clearance sign-off.</div>
                                </div>
                              </div>
                            </div>

                            <div className="dash-form-section">
                              <div className="dash-form-title">Attendance Rules</div>
                              <div className="row g-2">
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Late Time</label>
                                  <input className="form-control" type="time" name="late_time" defaultValue={profile?.late_time || ''} />
                                  <div className="form-text">Use 24-hour format (HH:MM), e.g. 08:30.</div>
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Closing Time</label>
                                  <input className="form-control" type="time" name="closing_time" defaultValue={profile?.closing_time || ''} />
                                  <div className="form-text">Use 24-hour format (HH:MM), e.g. 17:00.</div>
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Max Late Days</label>
                                  <input className="form-control" type="number" min="0" step="1" name="max_days_late" defaultValue={profile?.max_days_late ?? ''} />
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Max Absent Days</label>
                                  <input className="form-control" type="number" min="0" step="1" name="max_days_absent" defaultValue={profile?.max_days_absent ?? ''} />
                                </div>
                              </div>
                            </div>
                            <div className="d-grid mt-3">
                              <button className="btn btn-olive" disabled={isSaving}>
                                {isSaving ? 'Saving…' : 'Save'}
                              </button>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {editBranch && (
                <div className="dash-modal" onClick={() => setEditBranch(null)}>
                  <div className="dash-modal-card" onClick={(e) => e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Edit Branch</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditBranch(null)}>
                        Close
                      </button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Edit flow</h6>
                          <p>Update branch details and admin invitation information.</p>
                          <ul>
                            <li>Change name/address if needed.</li>
                            <li>Update admin email to reassign access.</li>
                            <li>Save to apply changes immediately.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          {editBranchForm && (
                            <form
                              onSubmit={async (e) => {
                                e.preventDefault()
                                setStatus('pending')
                                try {
                                  await api.put(`/api/auth/branches/${editBranch.id}/`, {
                                    ...editBranchForm,
                                    latitude: editBranchForm.latitude === '' ? null : Number(editBranchForm.latitude),
                                    longitude: editBranchForm.longitude === '' ? null : Number(editBranchForm.longitude),
                                  })
                                  await refreshAll()
                                  setEditBranch(null)
                                  setStatus('saved:branch')
                                } catch (err) {
                                  setStatus('error:branch')
                                }
                              }}
                            >
                              <div className="row g-2">
                                <div className="col-12">
                                  <label className="form-label">Branch name</label>
                                  <input
                                    className="form-control"
                                    value={editBranchForm.name}
                                    onChange={(e) => setEditBranchForm((p) => ({ ...p, name: e.target.value }))}
                                    required
                                  />
                                </div>
                              </div>

                              <GeofencePicker
                                address={editBranchForm.address}
                                onAddressChange={(v) => setEditBranchForm((p) => ({ ...p, address: v }))}
                                lat={editBranchForm.latitude}
                                lng={editBranchForm.longitude}
                                onLatLngChange={({ lat, lng }) =>
                                  setEditBranchForm((p) => ({ ...p, latitude: String(lat), longitude: String(lng) }))
                                }
                              />

                              <div className="row g-2 mt-2">
                                <div className="col-md-4">
                                  <label className="form-label">Admin Name</label>
                                  <input
                                    className="form-control"
                                    value={editBranchForm.admin_name}
                                    onChange={(e) => setEditBranchForm((p) => ({ ...p, admin_name: e.target.value }))}
                                  />
                                </div>
                                <div className="col-md-5">
                                  <label className="form-label">Admin Email</label>
                                  <input
                                    className="form-control"
                                    type="email"
                                    value={editBranchForm.admin_email}
                                    onChange={(e) => setEditBranchForm((p) => ({ ...p, admin_email: e.target.value }))}
                                  />
                                </div>
                                <div className="col-md-3">
                                  <label className="form-label">Staff ID</label>
                                  <input
                                    className="form-control"
                                    value={editBranchForm.admin_staff_id}
                                    onChange={(e) => setEditBranchForm((p) => ({ ...p, admin_staff_id: e.target.value }))}
                                  />
                                </div>
                              </div>

                              <div className="dash-modal-actions">
                                <div className="d-grid">
                                  <button className="btn btn-olive" disabled={isSaving}>
                                    {isSaving ? 'Saving…' : 'Save changes'}
                                  </button>
                                </div>
                              </div>
                            </form>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {editDepartment && (
                <div className="dash-modal" onClick={() => setEditDepartment(null)}>
                  <div className="dash-modal-card" onClick={(e) => e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Edit Department</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditDepartment(null)}>
                        Close
                      </button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Edit flow</h6>
                          <p>Departments belong to a branch and group units.</p>
                          <ul>
                            <li>Update the name or move to another branch.</li>
                            <li>Deleting a department may remove its units.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form
                            onSubmit={(e) => {
                              e.preventDefault()
                              setStatus('pending')
                              const fd = new FormData(e.target)
                              const payload = Object.fromEntries(fd)
                              ;(async () => {
                                try {
                                  await api.put(`/api/auth/departments/${editDepartment.id}/`, {
                                    name: payload.name,
                                    branch: Number(payload.branch),
                                  })
                                  await refreshAll()
                                  setEditDepartment(null)
                                  setStatus('saved:department')
                                } catch (err) {}
                              })()
                            }}
                          >
                            <label className="form-label">Branch</label>
                            <select className="form-select mb-2" name="branch" defaultValue={String(editDepartment.branch)} required>
                              {branches.map((b) => (
                                <option key={b.id} value={b.id}>
                                  {b.name}
                                </option>
                              ))}
                            </select>
                            <label className="form-label">Department name</label>
                            <input className="form-control" name="name" defaultValue={editDepartment.name} required />
                            <div className="d-grid mt-3">
                              <button className="btn btn-olive" disabled={isSaving}>
                                {isSaving ? 'Saving…' : 'Save changes'}
                              </button>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {editUnit && (
                <div className="dash-modal" onClick={() => setEditUnit(null)}>
                  <div className="dash-modal-card" onClick={(e) => e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Edit Unit</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditUnit(null)}>
                        Close
                      </button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Edit flow</h6>
                          <p>Units are optional sub-groups under departments.</p>
                          <ul>
                            <li>Update the unit name.</li>
                            <li>Move the unit to another department if needed.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form
                            onSubmit={(e) => {
                              e.preventDefault()
                              setStatus('pending')
                              const fd = new FormData(e.target)
                              const payload = Object.fromEntries(fd)
                              ;(async () => {
                                try {
                                  await api.put(`/api/auth/units/${editUnit.id}/`, {
                                    name: payload.name,
                                    department: Number(payload.department),
                                  })
                                  await refreshAll()
                                  setEditUnit(null)
                                  setStatus('saved:unit')
                                } catch (err) {}
                              })()
                            }}
                          >
                            <label className="form-label">Department</label>
                            <select className="form-select mb-2" name="department" defaultValue={String(editUnit.department)} required>
                              {deps.map((d) => (
                                <option key={d.id} value={d.id}>
                                  {d.name}
                                </option>
                              ))}
                            </select>
                            <label className="form-label">Unit name</label>
                            <input className="form-control" name="name" defaultValue={editUnit.name} required />
                            <div className="d-grid mt-3">
                              <button className="btn btn-olive" disabled={isSaving}>
                                {isSaving ? 'Saving…' : 'Save changes'}
                              </button>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
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
              <div className="card shadow-sm dash-card">
                <div className="card-body">
                  <div className="d-flex flex-wrap gap-2 justify-content-between align-items-center">
                    <div className="d-flex align-items-center gap-2 flex-wrap">
                      <button
                        className={`btn btn-sm ${clSearchOpen ? 'btn-olive' : 'btn-outline-secondary'}`}
                        type="button"
                        aria-label="Search"
                        onClick={() => setClSearchOpen((v) => !v)}
                      >
                        <Search size={16} />
                      </button>
                      {clSearchOpen && (
                        <div className="dash-table-search">
                          <input
                            className="form-control form-control-sm"
                            placeholder="Search corpers... (name, code, branch)"
                            value={clQuery}
                            onChange={(e)=>{ setClQuery(e.target.value); setClPage(1) }}
                          />
                        </div>
                      )}
                    </div>

                    <div className="d-flex align-items-center gap-2">
                      <span className="small text-muted">Rows</span>
                      <select className="form-select form-select-sm" style={{ width: 96 }} value={clPageSize} onChange={(e)=>{ setClPageSize(Number(e.target.value)); setClPage(1) }}>
                        {[50, 100].map(n => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </div>
                  </div>

                  <div className="table-responsive mt-2">
                    <table className="table table-sm align-middle dash-table dash-table-auto">
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
                          const pageSize = clPageSize
                          const q = clSearchOpen ? clQuery.trim().toLowerCase() : ''
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
                                <td><div className="text-truncate dash-td-truncate-wide">{row.full_name}</div></td>
                                <td>{row.state_code}</td>
                                <td><div className="text-truncate dash-td-truncate-wide">{row.branch || '—'}</div></td>
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
                    const pageSize = clPageSize
                    const q = clSearchOpen ? clQuery.trim().toLowerCase() : ''
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

              {(() => {
                const myBranch = branches.find(x => x.admin_info && x.admin_info.email === me?.email) || branches[0]
                if(!myBranch){ return (<div className="text-muted">No branch assigned.</div>) }
                const myDeps = deps.filter(d => d.branch === myBranch.id)
                const myUnits = units.filter(u => myDeps.some(d => d.id === u.department))

                return (
                  <>
                    <div className="dash-struct-nav mb-3">
                      <button className={`dash-struct-item ${structureTab==='branch'?'active':''}`} type="button" onClick={()=>setStructureTab('branch')}>My Branch</button>
                      <button className={`dash-struct-item ${structureTab==='departments'?'active':''}`} type="button" onClick={()=>setStructureTab('departments')}>Departments</button>
                      <button className={`dash-struct-item ${structureTab==='units'?'active':''}`} type="button" onClick={()=>setStructureTab('units')}>Units</button>
                      <button className={`dash-struct-item ${structureTab==='holidays'?'active':''}`} type="button" onClick={()=>setStructureTab('holidays')}>Holidays</button>
                    </div>

                    <div className="card shadow-sm dash-card">
                      <div className="card-body">
                        <div className="d-flex justify-content-between align-items-center gap-2">
                          <div className="dash-card-title mb-0">
                            {structureTab==='branch' ? 'My Branch' : structureTab==='departments' ? 'Departments' : structureTab==='units' ? 'Units' : 'Holidays (view only)'}
                          </div>
                          <div className="d-flex gap-2">
                            {structureTab==='branch' && (
                              <button className="btn btn-sm btn-olive" type="button" onClick={() => {
                                setBranchLocationPos(myBranch.latitude && myBranch.longitude ? { lat: myBranch.latitude, lng: myBranch.longitude } : null)
                                setShowBranchLocation(true)
                              }}>Update location</button>
                            )}
                            {structureTab==='departments' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddDepartment(true)}>Add Department</button>}
                            {structureTab==='units' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddUnit(true)}>Add Unit</button>}
                          </div>
                        </div>

                        {structureTab !== 'holidays' ? (
                          <div className="d-flex flex-wrap gap-2 justify-content-between align-items-center mt-3">
                            <div className="dash-table-search">
                              <input
                                className="form-control form-control-sm"
                                placeholder="Search…"
                                value={structQuery}
                                onChange={(e) => {
                                  setStructQuery(e.target.value)
                                  setStructPage(1)
                                }}
                              />
                            </div>
                            <div className="small text-muted">Page {structPage}</div>
                          </div>
                        ) : (
                          <div className="d-flex flex-wrap gap-2 justify-content-between align-items-center mt-3">
                            <div className="d-flex align-items-center gap-2 flex-wrap">
                              <button
                                className={`btn btn-sm ${structSearchOpen ? 'btn-olive' : 'btn-outline-secondary'}`}
                                type="button"
                                aria-label="Search"
                                onClick={() => setStructSearchOpen((v) => !v)}
                              >
                                <Search size={16} />
                              </button>
                              {structSearchOpen && (
                                <div className="dash-table-search">
                                  <input
                                    className="form-control form-control-sm"
                                    placeholder="Search…"
                                    value={structQuery}
                                    onChange={(e) => {
                                      setStructQuery(e.target.value)
                                      setStructPage(1)
                                    }}
                                  />
                                </div>
                              )}

                              <select
                                className="form-select form-select-sm"
                                style={{ width: 140 }}
                                value={structFilter}
                                onChange={(e) => {
                                  setStructFilter(e.target.value)
                                  setStructPage(1)
                                }}
                                aria-label="Filter"
                              >
                                <option value="all">All</option>
                                <option value="manual">Manual</option>
                                <option value="auto">Auto (NG)</option>
                              </select>

                              <select
                                className="form-select form-select-sm"
                                style={{ width: 140 }}
                                value={structSortKey}
                                onChange={(e) => {
                                  setStructSortKey(e.target.value)
                                  setStructPage(1)
                                }}
                                aria-label="Sort by"
                              >
                                <option value="date">Sort: Date</option>
                                <option value="title">Sort: Title</option>
                                <option value="type">Sort: Type</option>
                              </select>

                              <select
                                className="form-select form-select-sm"
                                style={{ width: 110 }}
                                value={structSortDir}
                                onChange={(e) => {
                                  setStructSortDir(e.target.value)
                                  setStructPage(1)
                                }}
                                aria-label="Sort direction"
                              >
                                <option value="asc">Asc</option>
                                <option value="desc">Desc</option>
                              </select>
                            </div>
                            <div className="d-flex align-items-center gap-2">
                              <span className="small text-muted">Rows</span>
                              <select
                                className="form-select form-select-sm"
                                style={{ width: 96 }}
                                value={structPageSize}
                                onChange={(e) => {
                                  setStructPageSize(Number(e.target.value))
                                  setStructPage(1)
                                }}
                              >
                                {[20, 50, 100].map((n) => (
                                  <option key={n} value={n}>
                                    {n}
                                  </option>
                                ))}
                              </select>
                            </div>
                          </div>
                        )}

                        <div className="table-responsive mt-2">
                          {structureTab==='branch' && (
                            <table className="table table-sm align-middle">
                              <thead><tr><th>Name</th><th>Address</th><th>Latitude</th><th>Longitude</th></tr></thead>
                              <tbody>
                                <tr>
                                  <td className="fw-semibold">{myBranch.name}</td>
                                  <td>{myBranch.address || '—'}</td>
                                  <td>{myBranch.latitude ?? '—'}</td>
                                  <td>{myBranch.longitude ?? '—'}</td>
                                </tr>
                              </tbody>
                            </table>
                          )}

                          {structureTab==='departments' && (
                            <table className="table table-sm align-middle dash-table">
                              <thead><tr><th>Name</th><th></th></tr></thead>
                              <tbody>
                                {(() => {
                                  const q = structQuery.trim().toLowerCase()
                                  const filtered = q ? myDeps.filter((d) => d.name.toLowerCase().includes(q)) : myDeps
                                  const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                                  const current = Math.min(structPage, totalPages)
                                  if (current !== structPage) setStructPage(current)
                                  const start = (current - 1) * structPageSize
                                  const rows = filtered.slice(start, start + structPageSize)
                                  return (
                                    <>
                                      {rows.map((d) => (
                                  <tr key={d.id}>
                                    <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{d.name}</div></td>
                                    <td className="text-end">
                                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => {
                                        const newName = prompt('Edit department name (leave empty to delete)', d.name)
                                        if(newName === null) return
                                        const trimmed = (newName || '').trim()
                                        if(trimmed === ''){
                                          if(confirm('Delete this department and its units?')){
                                            ;(async()=>{ try{ await api.delete(`/api/auth/departments/${d.id}/`); await refreshAll() }catch(e){} })()
                                          }
                                        }else{
                                          ;(async()=>{ try{ await api.put(`/api/auth/departments/${d.id}/`, { name: trimmed, branch: d.branch }); await refreshAll() }catch(e){} })()
                                        }
                                      }}>Edit</button>
                                    </td>
                                  </tr>
                                      ))}
                                      {filtered.length===0 && <tr><td colSpan="2" className="text-muted">No departments found.</td></tr>}
                                      {filtered.length>0 && (
                                        <tr>
                                          <td colSpan="2">
                                            <div className="d-flex justify-content-between align-items-center">
                                              <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                              <div className="btn-group">
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructPage(p=>Math.max(1,p-1))}>Prev</button>
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                              </div>
                                            </div>
                                          </td>
                                        </tr>
                                      )}
                                    </>
                                  )
                                })()}
                              </tbody>
                            </table>
                          )}

                          {structureTab==='units' && (
                            <table className="table table-sm align-middle dash-table">
                              <thead><tr><th>Name</th><th>Department</th><th></th></tr></thead>
                              <tbody>
                                {(() => {
                                  const q = structQuery.trim().toLowerCase()
                                  const filtered = q
                                    ? myUnits.filter((u) => `${u.name} ${myDeps.find(d=>d.id===u.department)?.name||''}`.toLowerCase().includes(q))
                                    : myUnits
                                  const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                                  const current = Math.min(structPage, totalPages)
                                  if (current !== structPage) setStructPage(current)
                                  const start = (current - 1) * structPageSize
                                  const rows = filtered.slice(start, start + structPageSize)
                                  return (
                                    <>
                                      {rows.map((u) => (
                                  <tr key={u.id}>
                                    <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{u.name}</div></td>
                                    <td><div className="text-truncate dash-td-truncate">{myDeps.find(d=>d.id===u.department)?.name || '—'}</div></td>
                                    <td className="text-end">
                                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => {
                                        const newUnitName = prompt('Edit unit name (leave empty to delete)', u.name)
                                        if(newUnitName === null) return
                                        const trimmed = (newUnitName || '').trim()
                                        if(trimmed === ''){
                                          if(confirm('Delete this unit?')){
                                            ;(async()=>{ try{ await api.delete(`/api/auth/units/${u.id}/`); await refreshAll() }catch(e){} })()
                                          }
                                        }else{
                                          ;(async()=>{ try{ await api.put(`/api/auth/units/${u.id}/`, { name: trimmed, department: u.department }); await refreshAll() }catch(e){} })()
                                        }
                                      }}>Edit</button>
                                    </td>
                                  </tr>
                                      ))}
                                      {filtered.length===0 && <tr><td colSpan="3" className="text-muted">No units found.</td></tr>}
                                      {filtered.length>0 && (
                                        <tr>
                                          <td colSpan="3">
                                            <div className="d-flex justify-content-between align-items-center">
                                              <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                              <div className="btn-group">
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructPage(p=>Math.max(1,p-1))}>Prev</button>
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                              </div>
                                            </div>
                                          </td>
                                        </tr>
                                      )}
                                    </>
                                  )
                                })()}
                              </tbody>
                            </table>
                          )}

                          {structureTab==='holidays' && (
                            <table className="table table-sm align-middle dash-table">
                              <thead><tr><th>Title</th><th>Date</th><th>Type</th><th></th></tr></thead>
                              <tbody>
                                {(() => {
                                  const q = structSearchOpen ? structQuery.trim().toLowerCase() : ''
                                  const base = holidaysAll.length ? holidaysAll : holidays
                                  let filtered = q
                                    ? base.filter((h) => `${h.title} ${h.start_date} ${h.end_date} ${h.source||''}`.toLowerCase().includes(q))
                                    : base

                                  if(structFilter !== 'all'){
                                    filtered = filtered.filter((h) => {
                                      const isAuto = h.source === 'NATIONAL' || h.deletable === false
                                      return structFilter === 'auto' ? isAuto : !isAuto
                                    })
                                  }

                                  const dir = structSortDir === 'desc' ? -1 : 1
                                  const cmp = (a, b) => {
                                    const typeLabel = (h) => (h.source === 'NATIONAL' || h.deletable === false ? 'auto' : 'manual')
                                    const getVal = (h) => {
                                      if(structSortKey === 'title') return (h.title || '').toLowerCase()
                                      if(structSortKey === 'type') return typeLabel(h)
                                      return (h.start_date || '').toLowerCase()
                                    }
                                    return getVal(a).localeCompare(getVal(b)) * dir
                                  }
                                  filtered = [...filtered].sort(cmp)

                                  const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                                  const current = Math.min(structPage, totalPages)
                                  if (current !== structPage) setStructPage(current)
                                  const start = (current - 1) * structPageSize
                                  const rows = filtered.slice(start, start + structPageSize)

                                  return (
                                    <>
                                      {rows.map((h) => (
                                        <tr key={h.id}>
                                          <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{h.title}</div></td>
                                          <td>{h.start_date}{h.end_date && h.end_date !== h.start_date ? ` → ${h.end_date}` : ''}</td>
                                          <td>
                                            {h.source === 'NATIONAL' || h.deletable === false ? (
                                              <span className="badge bg-secondary">Auto (NG)</span>
                                            ) : (
                                              <span className="badge bg-olive">Manual</span>
                                            )}
                                          </td>
                                          <td className="text-end">
                                            {h.source === 'NATIONAL' || h.deletable === false ? (
                                              <span className="badge bg-secondary">Auto</span>
                                            ) : (
                                              <span className="badge bg-olive">Manual</span>
                                            )}
                                          </td>
                                        </tr>
                                      ))}
                                      {filtered.length===0 && <tr><td colSpan="4" className="text-muted">No holidays found.</td></tr>}
                                      {filtered.length>0 && (
                                        <tr>
                                          <td colSpan="4">
                                            <div className="d-flex justify-content-between align-items-center">
                                              <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                              <div className="btn-group">
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructPage(p=>Math.max(1,p-1))}>Prev</button>
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                              </div>
                                            </div>
                                          </td>
                                        </tr>
                                      )}
                                    </>
                                  )
                                })()}
                              </tbody>
                            </table>
                          )}
                        </div>
                      </div>
                    </div>

                    {showAddDepartment && (
                      <div className="dash-modal" onClick={() => setShowAddDepartment(false)}>
                        <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                          <div className="dash-modal-head">
                            <strong>Add Department</strong>
                            <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowAddDepartment(false)}>Close</button>
                          </div>
                          <div className="card-body">
                            <form onSubmit={async (e)=>{
                              e.preventDefault()
                              setStatus('pending')
                              const f=new FormData(e.target)
                              const name=f.get('name')
                              try{
                                await api.post('/api/auth/departments/', { branch: myBranch.id, name })
                                await refreshAll()
                                setStatus('saved:department')
                                setShowAddDepartment(false)
                              }catch(err){ setStatus('error:department') }
                            }}>
                              <input className="form-control mb-3" name="name" placeholder="Department name" required/>
                              <div className="d-grid"><button className="btn btn-olive">Add Department</button></div>
                            </form>
                          </div>
                        </div>
                      </div>
                    )}

                    {showAddUnit && (
                      <div className="dash-modal" onClick={() => setShowAddUnit(false)}>
                        <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                          <div className="dash-modal-head">
                            <strong>Add Unit</strong>
                            <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowAddUnit(false)}>Close</button>
                          </div>
                          <div className="card-body">
                            <form onSubmit={async (e)=>{
                              e.preventDefault()
                              setStatus('pending')
                              const f=new FormData(e.target)
                              const name=f.get('name')
                              const department=Number(f.get('department'))
                              try{
                                await api.post('/api/auth/units/', { name, department })
                                await refreshAll()
                                setStatus('saved:unit')
                                setShowAddUnit(false)
                              }catch(err){ setStatus('error:unit') }
                            }}>
                              <select className="form-select mb-2" name="department" required>
                                <option value="">Select Department</option>
                                {myDeps.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                              </select>
                              <input className="form-control mb-3" name="name" placeholder="Unit name" required/>
                              <div className="d-grid"><button className="btn btn-olive">Add Unit</button></div>
                            </form>
                          </div>
                        </div>
                      </div>
                    )}

                    {showBranchLocation && (
                      <div className="dash-modal" onClick={() => setShowBranchLocation(false)}>
                        <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                          <div className="dash-modal-head">
                            <strong>Update Branch Location</strong>
                            <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowBranchLocation(false)}>Close</button>
                          </div>
                          <div className="dash-modal-body">
                            <div className="dash-modal-grid">
                              <div className="dash-modal-help">
                                <h6>Flow</h6>
                                <p>Admins can update branch coordinates for location-based attendance.</p>
                                <ul>
                                  <li>Click on the map to set a pin.</li>
                                  <li>Use the 📍 control for current location.</li>
                                  <li>Save to apply immediately.</li>
                                </ul>
                              </div>
                              <div className="dash-modal-form">
                                <MapPicker
                                  value={branchLocationPos}
                                  onChange={(pos) => setBranchLocationPos(pos)}
                                  height={260}
                                  zoom={branchLocationPos ? 14 : 6}
                                />
                              <div className="d-grid mt-3">
                                <button className="btn btn-olive" type="button" disabled={isSaving} onClick={async()=>{
                                  if(!branchLocationPos){ alert('Pick a location on the map first.'); return }
                                  setStatus('pending')
                                  try{
                                    await api.put(`/api/auth/branches/${myBranch.id}/`, { latitude: branchLocationPos.lat, longitude: branchLocationPos.lng, name: myBranch.name, address: myBranch.address||'' })
                                    await refreshAll()
                                    setShowBranchLocation(false)
                                    setStatus('saved:branch')
                                  }catch(e){}
                                }}>
                                  {isSaving ? 'Saving…' : 'Save Location'}
                                </button>
                              </div>
                            </div>
                          </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )
              })()}
            </>
          )}

          {activeTab==='corpers' && (me?.role==='ORG' || me?.role==='BRANCH') && (
            <>
              <h2 className="mb-3 text-olive">Corpers</h2>

              <div className="card shadow-sm dash-card">
                <div className="card-body">
                  <div className="d-flex justify-content-between align-items-center gap-2">
                    <div className="dash-card-title mb-0">Registered Corpers</div>
                    <button className="btn btn-sm btn-olive" type="button" onClick={() => setShowAddCorper(true)}>
                      Add Corper
                    </button>
                  </div>

                  <div className="d-flex flex-wrap gap-2 justify-content-between align-items-center mt-3">
                    <div className="d-flex align-items-center gap-2 flex-wrap">
                      <button
                        className={`btn btn-sm ${corperSearchOpen ? 'btn-olive' : 'btn-outline-secondary'}`}
                        type="button"
                        aria-label="Search"
                        onClick={() => setCorperSearchOpen((v) => !v)}
                      >
                        <Search size={16} />
                      </button>
                      {corperSearchOpen && (
                        <div className="dash-table-search">
                          <input
                            className="form-control form-control-sm"
                            placeholder="Search…"
                            value={corperQuery}
                            onChange={(e) => {
                              setCorperQuery(e.target.value)
                              setCorperPage(1)
                            }}
                          />
                        </div>
                      )}

                      {me?.role === 'ORG' && (
                        <select
                          className="form-select form-select-sm"
                          style={{ width: 170 }}
                          value={corperFilterBranch}
                          onChange={(e) => {
                            setCorperFilterBranch(e.target.value)
                            setCorperPage(1)
                          }}
                          aria-label="Filter by branch"
                        >
                          <option value="all">All branches</option>
                          {branches.map((b) => (
                            <option key={b.id} value={String(b.id)}>
                              {b.name}
                            </option>
                          ))}
                        </select>
                      )}

                      <select
                        className="form-select form-select-sm"
                        style={{ width: 160 }}
                        value={corperSortKey}
                        onChange={(e) => {
                          setCorperSortKey(e.target.value)
                          setCorperPage(1)
                        }}
                        aria-label="Sort by"
                      >
                        <option value="name">Sort: Name</option>
                        <option value="code">Sort: State code</option>
                        <option value="branch">Sort: Branch</option>
                      </select>
                      <select
                        className="form-select form-select-sm"
                        style={{ width: 110 }}
                        value={corperSortDir}
                        onChange={(e) => {
                          setCorperSortDir(e.target.value)
                          setCorperPage(1)
                        }}
                        aria-label="Sort direction"
                      >
                        <option value="asc">Asc</option>
                        <option value="desc">Desc</option>
                      </select>
                    </div>

                    <div className="d-flex align-items-center gap-2">
                      <span className="small text-muted">Rows</span>
                      <select
                        className="form-select form-select-sm"
                        style={{ width: 96 }}
                        value={corperPageSize}
                        onChange={(e) => {
                          setCorperPageSize(Number(e.target.value))
                          setCorperPage(1)
                        }}
                      >
                        {[20, 50, 100].map((n) => (
                          <option key={n} value={n}>
                            {n}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="table-responsive mt-2">
                    <table className="table table-sm align-middle dash-table">
                      <thead>
                        <tr>
                          <th>Full Name</th>
                          <th>State Code</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {(() => {
                          const q = corperSearchOpen ? corperQuery.trim().toLowerCase() : ''
                          let filtered = q
                            ? corpers.filter((c) => `${c.full_name} ${c.email || ''} ${c.state_code}`.toLowerCase().includes(q))
                            : corpers

                          if (me?.role === 'ORG' && corperFilterBranch !== 'all') {
                            filtered = filtered.filter((c) => String(c.branch || '') === corperFilterBranch)
                          }

                          const dir = corperSortDir === 'desc' ? -1 : 1
                          const getBranchName = (id) => branches.find((b) => b.id === id)?.name || ''
                          const getDeptName = (id) => deps.find((d) => d.id === id)?.name || ''
                          const getUnitName = (id) => units.find((u) => u.id === id)?.name || ''
                          const cmp = (a, b) => {
                            const av = (
                              corperSortKey === 'code'
                                ? a.state_code || ''
                                : corperSortKey === 'branch'
                                  ? getBranchName(a.branch)
                                  : a.full_name || ''
                            ).toLowerCase()
                            const bv = (
                              corperSortKey === 'code'
                                ? b.state_code || ''
                                : corperSortKey === 'branch'
                                  ? getBranchName(b.branch)
                                  : b.full_name || ''
                            ).toLowerCase()
                            return av.localeCompare(bv) * dir
                          }
                          filtered = [...filtered].sort(cmp)

                          const totalPages = Math.max(1, Math.ceil(filtered.length / corperPageSize))
                          const current = Math.min(corperPage, totalPages)
                          if (current !== corperPage) setCorperPage(current)
                          const start = (current - 1) * corperPageSize
                          const rows = filtered.slice(start, start + corperPageSize)

                          return (
                            <>
                              {rows.map((c) => (
                                <tr key={c.id} role="button" onClick={() => setSelectedCorper(c)}>
                                  <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{c.full_name}</div></td>
                                  <td>{c.state_code}</td>
                                  <td className="text-end">
                                    <button className="btn btn-sm btn-outline-secondary" type="button" onClick={(e) => { e.stopPropagation(); setSelectedCorper(c) }}>
                                      View
                                    </button>
                                  </td>
                                </tr>
                              ))}
                              {filtered.length === 0 && <tr><td colSpan="3" className="text-muted">No corpers found.</td></tr>}
                              {filtered.length > 0 && (
                                <tr>
                                  <td colSpan="3">
                                    <div className="d-flex justify-content-between align-items-center">
                                      <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                      <div className="btn-group">
                                        <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setCorperPage(p=>Math.max(1,p-1))}>Prev</button>
                                        <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setCorperPage(p=>Math.min(totalPages,p+1))}>Next</button>
                                      </div>
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </>
                          )
                        })()}
                      </tbody>
                    </table>
                  </div>

                  {status==='saved:corper' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Enrollment successful. A verification email was sent to the corper.</AutoFadeAlert>}
                  {status?.startsWith('error:corper:') && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>{status.split(':').slice(2).join(':')}</AutoFadeAlert>}
                  {status==='saved:corper-update' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Corper updated.</AutoFadeAlert>}
                  {status==='error:corper-update' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not update corper.</AutoFadeAlert>}
                </div>
              </div>

              {showAddCorper && (
                <div className="dash-modal" onClick={()=>setShowAddCorper(false)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Add Corper</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setShowAddCorper(false)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Flow</h6>
                          <p>Create a corper account and assign placement.</p>
                          <ul>
                            <li>Corper receives an email to verify and set password.</li>
                            <li>Use Face capture after creation.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{ const ok = await createCorper(e); if(ok) setShowAddCorper(false) }}>
                            <div className="dash-form-section">
                              <div className="dash-form-title">Basic Details</div>
                              <div className="row g-2">
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Email</label>
                                  <input
                                    className={`form-control ${corperFormErrors?.email ? 'is-invalid' : ''}`}
                                    type="email"
                                    name="email"
                                    placeholder="corper@example.com"
                                    required
                                    onChange={() => {
                                      if (corperFormErrors?.email) {
                                        setCorperFormErrors((p) => {
                                          const next = { ...(p || {}) }
                                          delete next.email
                                          return next
                                        })
                                      }
                                    }}
                                  />
                                  {corperFormErrors?.email && (
                                    <div className="invalid-feedback">
                                      {Array.isArray(corperFormErrors.email) ? corperFormErrors.email[0] : String(corperFormErrors.email)}
                                    </div>
                                  )}
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Full Name</label>
                                  <input className="form-control" name="full_name" placeholder="Surname Firstname Lastname" required />
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">Gender</label>
                                  <select className="form-select" name="gender" required>
                                    <option value="">Select...</option>
                                    <option value="M">Male</option>
                                    <option value="F">Female</option>
                                    <option value="O">Other</option>
                                  </select>
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">State Code</label>
                                  <input className="form-control" name="state_code" placeholder="AA/00A/0000" required />
                                  <div className="form-text">Format: `AA/00A/0000`</div>
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">Passing Out Date</label>
                                  <input className="form-control" type="date" name="passing_out_date" required />
                                </div>
                              </div>
                            </div>

                            <div className="dash-form-section">
                              <div className="dash-form-title">Placement</div>
                              <div className="row g-2">
                                {me?.role==='ORG' && (
                                  <div className="col-12 col-md-4">
                                    <label className="form-label">Branch</label>
                                    <select className="form-select" name="branch" required value={enrollBranch} onChange={(e)=>{ setEnrollBranch(e.target.value); setEnrollDept('') }}>
                                      <option value="">Select branch</option>
                                      {enrollBranchOptions().map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                                    </select>
                                  </div>
                                )}
                                <div className="col-12 col-md-4">
                                  <label className="form-label">Department (optional)</label>
                                  <select className="form-select" name="department" value={enrollDept} onChange={(e)=> setEnrollDept(e.target.value)}>
                                    <option value="">Select department</option>
                                    {enrollDeptOptions().map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                                  </select>
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">Unit (optional)</label>
                                  <select className="form-select" name="unit">
                                    <option value="">Select unit</option>
                                    {enrollUnitOptions().map(u => <option key={u.id} value={u.id}>{u.name}</option>)}
                                  </select>
                                </div>
                              </div>
                            </div>
                            <div className="dash-modal-actions">
                              <div className="d-grid">
                                <button className="btn btn-olive" disabled={isSaving}>{isSaving ? 'Adding…' : 'Enroll'}</button>
                              </div>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {editCorper && editCorperForm && (
                <div className="dash-modal" onClick={()=>setEditCorper(null)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Edit Corper Placement</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setEditCorper(null)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Flow</h6>
                          <p>Update branch/department/unit placement.</p>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{
                            e.preventDefault();
                            setStatus('pending')
                            try{
                              await api.patch(`/api/auth/corpers/${editCorper.id}/`, {
                                branch: editCorperForm.branch || null,
                                department: editCorperForm.department || null,
                                unit: editCorperForm.unit || null,
                              })
                              await refreshAll();
                              setStatus('saved:corper-update');
                              setEditCorper(null)
                            }catch(err){ setStatus('error:corper-update') }
                          }}>
                            {me?.role==='ORG' && (
                              <div className="mb-2">
                                <label className="form-label">Branch</label>
                                <select className="form-select" value={editCorperForm.branch} onChange={(e)=>setEditCorperForm(p=>({ ...p, branch: e.target.value, department:'', unit:'' }))}>
                                  <option value="">—</option>
                                  {branches.map(b=> <option key={b.id} value={b.id}>{b.name}</option>)}
                                </select>
                              </div>
                            )}
                            <div className="mb-2">
                              <label className="form-label">Department</label>
                              <select className="form-select" value={editCorperForm.department} onChange={(e)=>setEditCorperForm(p=>({ ...p, department: e.target.value, unit:'' }))}>
                                <option value="">—</option>
                                {deps.filter(d=>{
                                  const bid = Number(editCorperForm.branch || editCorper.branch)
                                  return !bid || d.branch === bid
                                }).map(d=> <option key={d.id} value={d.id}>{d.name}</option>)}
                              </select>
                            </div>
                            <div className="mb-2">
                              <label className="form-label">Unit</label>
                              <select className="form-select" value={editCorperForm.unit} onChange={(e)=>setEditCorperForm(p=>({ ...p, unit: e.target.value }))}>
                                <option value="">—</option>
                                {units.filter(u=>{
                                  const did = Number(editCorperForm.department || editCorper.department)
                                  return !did || u.department === did
                                }).map(u=> <option key={u.id} value={u.id}>{u.name}</option>)}
                              </select>
                            </div>
                            <div className="dash-modal-actions">
                              <div className="d-grid">
                                <button className="btn btn-olive" disabled={isSaving}>{isSaving ? 'Saving…' : 'Save'}</button>
                              </div>
                            </div>
                          </form>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {selectedCorper && (
                <div className="dash-modal" onClick={()=>setSelectedCorper(null)}>
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>Corper Details</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setSelectedCorper(null)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Details</h6>
                          <p>Review the corper’s record and manage placement.</p>
                          <ul>
                            <li>Use Edit placement to change assignment.</li>
                            <li>Use Face capture to enroll face data.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <div className="dash-profile-card">
                            <div className="dash-profile-card-title">Corper</div>
                            <div className="dash-kv-grid">
                              <div className="dash-kv"><div className="dash-k">Full name</div><div className="dash-v">{selectedCorper.full_name}</div></div>
                              <div className="dash-kv"><div className="dash-k">State code</div><div className="dash-v">{selectedCorper.state_code}</div></div>
                              <div className="dash-kv"><div className="dash-k">Email</div><div className="dash-v">{selectedCorper.email || '—'}</div></div>
                              <div className="dash-kv"><div className="dash-k">Gender</div><div className="dash-v">{selectedCorper.gender || '—'}</div></div>
                              <div className="dash-kv"><div className="dash-k">Passing out</div><div className="dash-v">{selectedCorper.passing_out_date || '—'}</div></div>
                            </div>
                          </div>

                          <div className="dash-profile-card mt-3">
                            <div className="dash-profile-card-title">Placement</div>
                            <div className="dash-kv-grid">
                              <div className="dash-kv"><div className="dash-k">Branch</div><div className="dash-v">{branches.find(b=>b.id===selectedCorper.branch)?.name || '—'}</div></div>
                              <div className="dash-kv"><div className="dash-k">Department</div><div className="dash-v">{deps.find(d=>d.id===selectedCorper.department)?.name || '—'}</div></div>
                              <div className="dash-kv"><div className="dash-k">Unit</div><div className="dash-v">{units.find(u=>u.id===selectedCorper.unit)?.name || '—'}</div></div>
                            </div>
                          </div>

                          <div className="dash-modal-actions">
                            <div className="d-flex gap-2">
                              <button className="btn btn-outline-secondary" type="button" onClick={()=>{ setEditCorper(selectedCorper); setSelectedCorper(null) }}>
                                Edit placement
                              </button>
                              <a className="btn btn-outline-secondary" href={apiHref(`/api/auth/capture/${selectedCorper.id}/`)} target="_blank" rel="noreferrer">
                                Face capture
                              </a>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {activeTab==='attendance' && (
            <>
              <h2 className="mb-3 text-olive">Attendance</h2>
              {me?.role==='CORPER' && (
                <div className="mb-3">
                <a className="btn btn-olive" href={apiHref('/api/auth/attendance/')} target="_blank" rel="noreferrer">Mark Attendance</a>
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
                <a className="btn btn-olive" href={apiHref('/api/auth/performance/clearance/')} target="_blank" rel="noreferrer">View Clearance Letter</a>
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

          <div className="dash-mini-footer">
            © {new Date().getFullYear()} Sahab Technology Integrated Limited
          </div>
        </div>
      </div>
    </div>
  )
}
