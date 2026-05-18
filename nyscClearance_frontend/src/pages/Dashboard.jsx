// Dashboard: app shell for ORG / BRANCH / CORPER
// - Loads profile, structure, stats, notifications
// - Wallet: funding via Paystack init/verify; modal accepts comma-separated amounts
// - Handles callback params: ?paystack=1&reference=..., ?fund=1
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import api, { ensureCsrf } from '../api/axios'
import { apiHref } from '../api/urls'
import { CONFIG_REFRESH_MS, fetchAdminConfigVersion } from '../api/configFreshness'
import MapPicker from '../components/MapPicker'
import GeofencePicker from '../components/GeofencePicker'
import { Bar, Doughnut, Line } from 'react-chartjs-2'
import AutoFadeAlert from '../components/AutoFadeAlert'
import thankYouAudio from '../assets/thank_you_message.mp3'
import {
  BarChart3,
  Bell,
  Building2,
  CalendarDays,
  FileCheck2,
  FileSearch,
  FileSpreadsheet,
  FileText,
  Home,
  Layers3,
  LayoutGrid,
  Menu,
  Pencil,
  Search,
  Trash2,
  Users,
  CalendarCheck2,
  CreditCard,
  Download,
  RefreshCw,
  Wallet,
} from 'lucide-react'
import { Chart as ChartJS, CategoryScale, LinearScale, ArcElement, BarElement, LineElement, PointElement, Filler, Tooltip, Legend } from 'chart.js'
ChartJS.register(CategoryScale, LinearScale, ArcElement, BarElement, LineElement, PointElement, Filler, Tooltip, Legend)

function DashboardLoadingScreen(){
  return (
    <div className="dashboard-loading-shell">
      <div className="dashboard-loading-card text-center">
        <div className="dashboard-loading-icon mx-auto">
          <RefreshCw size={24} className="spin-icon" />
        </div>
        <h1 className="h4 text-olive mt-3 mb-2">Loading dashboard</h1>
        <p className="text-muted mb-3">Fetching profile, attendance, wallet, subscription, reports, and permissions.</p>
        <div className="loading-progress" aria-hidden>
          <div className="loading-progress-bar" />
        </div>
        <div className="dashboard-loading-steps mt-3">
          <span>Secure session</span>
          <span>Fresh data</span>
          <span>Ready workspace</span>
        </div>
      </div>
    </div>
  )
}

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
  const [queries, setQueries] = useState([])
  const [queryReplyDrafts, setQueryReplyDrafts] = useState({})
  const [wallet, setWallet] = useState(null)
  const [subscriptionInfo, setSubscriptionInfo] = useState(null)
  const [announcement, setAnnouncement] = useState(null)
  const [showFund, setShowFund] = useState(false)
  const [fundAmount, setFundAmount] = useState('')
  const [clPage, setClPage] = useState(1)
  const [status, setStatus] = useState(null)
  const [dashboardLoading, setDashboardLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('overview')

  const [reportStart, setReportStart] = useState('')
  const [reportEnd, setReportEnd] = useState('')
  const [reportData, setReportData] = useState(null)
  const [reportStatus, setReportStatus] = useState(null)
  const [reportMessage, setReportMessage] = useState('')
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
  const [profileFormErrors, setProfileFormErrors] = useState({})
  const [tourOpen, setTourOpen] = useState(false)
  const [tourStep, setTourStep] = useState(0)
  const [setupDismissed, setSetupDismissed] = useState(false)
  const [showStructureImport, setShowStructureImport] = useState(false)
  const [structureImportFile, setStructureImportFile] = useState(null)
  const [structureImportPreview, setStructureImportPreview] = useState(null)
  const [showCorperImport, setShowCorperImport] = useState(false)
  const [corperImportFile, setCorperImportFile] = useState(null)
  const [corperImportPreview, setCorperImportPreview] = useState(null)
  const [editCorper, setEditCorper] = useState(null)
  const [editCorperForm, setEditCorperForm] = useState(null)
  const [selectedCorper, setSelectedCorper] = useState(null)

  const [myLeaveQuery, setMyLeaveQuery] = useState('')
  const [myLeaveSearchOpen, setMyLeaveSearchOpen] = useState(false)
  const [myLeavePage, setMyLeavePage] = useState(1)
  const [myLeavePageSize, setMyLeavePageSize] = useState(20)

  useEffect(() => {
    if (!showAddCorper) {
      setCorperFormErrors({})
    }
  }, [showAddCorper])

  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [structureTab, setStructureTab] = useState('profile')

  const isSaving = status === 'pending'
  const configVersionRef = useRef('')

  const [showAddBranch, setShowAddBranch] = useState(false)
  const [showAddDepartment, setShowAddDepartment] = useState(false)
  const [showAddUnit, setShowAddUnit] = useState(false)
  const [showAddHoliday, setShowAddHoliday] = useState(false)
  const [showEditProfile, setShowEditProfile] = useState(false)
  const [showBranchLocation, setShowBranchLocation] = useState(false)
  const [branchLocationPos, setBranchLocationPos] = useState(null)
  const [branchLocationAddress, setBranchLocationAddress] = useState('')
  const [branchLocationAddressTouched, setBranchLocationAddressTouched] = useState(false)
  const [showBranchGeoTour, setShowBranchGeoTour] = useState(false)
  const [showCorperTour, setShowCorperTour] = useState(false)
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
  const [forceStructureSetup, setForceStructureSetup] = useState(false)
  const [structurePrompted, setStructurePrompted] = useState(false)

  useEffect(() => {
    if (!showEditProfile) {
      setProfileFormErrors({})
    }
  }, [showEditProfile])

  // On org login: if no branches, prompt from Organisation Profile before Head Office setup.
  useEffect(() => {
    if(me?.role !== 'ORG') return
    const hasBranches = Array.isArray(branches) && branches.length > 0
    if(hasBranches){
      setForceStructureSetup(false)
      return
    }
    setForceStructureSetup(true)
    if(structurePrompted) return
    setStructurePrompted(true)
    setActiveTab('structure')
    setStructureTab('profile')
    setNewBranchForm((p) => ({
      ...p,
      name: p.name || 'Head Office',
    }))
  }, [me?.role, branches, structurePrompted])

  // Lightweight onboarding prompts (branch geo-fence location + corper quick tour).
  useEffect(() => {
    if(!me?.role) return
    try{
      if(me.role === 'BRANCH'){
        const myBranch = branches.find(x => x.admin_info && x.admin_info.email === me?.email) || branches[0]
        const missing = myBranch && !(myBranch.latitude && myBranch.longitude)
        const prompted = sessionStorage.getItem('nysc_branch_geo_prompted') === '1'
        if(missing && !prompted){
          sessionStorage.setItem('nysc_branch_geo_prompted', '1')
          setShowBranchGeoTour(true)
        }
      }

      if(me.role === 'CORPER'){
        const prompted = sessionStorage.getItem('nysc_corper_tour_prompted') === '1'
        if(prompted) return
        sessionStorage.setItem('nysc_corper_tour_prompted', '1')
        const key = `nysc_corper_login_count:${me.email || 'corper'}`
        let count = parseInt(localStorage.getItem(key) || '0', 10)
        if(Number.isNaN(count)) count = 0
        count += 1
        localStorage.setItem(key, String(count))
        if(count <= 3){
          setShowCorperTour(true)
        }
      }
    }catch(e){}
  }, [me, branches])

  // When branch admin uses the map to update coordinates, refresh the prefilled address (unless user typed a custom one).
  useEffect(() => {
    if(!showBranchLocation) return
    const lat = branchLocationPos?.lat
    const lng = branchLocationPos?.lng
    if(typeof lat !== 'number' || typeof lng !== 'number') return
    if(branchLocationAddressTouched) return

    const ctrl = new AbortController()
    const t = setTimeout(async () => {
      try{
        const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lng)}`
        const res = await fetch(url, { signal: ctrl.signal, headers: { 'Accept': 'application/json' } })
        if(!res.ok) return
        const data = await res.json()
        const label = String(data?.display_name || '').trim()
        if(label) setBranchLocationAddress(label)
      }catch(e){}
    }, 500)

    return () => { clearTimeout(t); ctrl.abort() }
  }, [showBranchLocation, branchLocationPos?.lat, branchLocationPos?.lng, branchLocationAddressTouched])

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
      email: editCorper.email || '',
      full_name: editCorper.full_name || '',
      state_code: editCorper.state_code || '',
      gender: editCorper.gender || '',
      passing_out_date: editCorper.passing_out_date || '',
      cds_day: editCorper.cds_day ?? '',
      branch: editCorper.branch || '',
      department: editCorper.department || '',
      unit: editCorper.unit || '',
    })
  }, [editCorper])
  const [structQuery, setStructQuery] = useState('')
  const [structPage, setStructPage] = useState(1)
  const [structBranchesPage, setStructBranchesPage] = useState(1)
  const [structDepartmentsPage, setStructDepartmentsPage] = useState(1)
  const [structUnitsPage, setStructUnitsPage] = useState(1)
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
	    showStructureImport ||
	    showCorperImport ||
	    showEditProfile ||
    showBranchLocation ||
    editBranch ||
    editDepartment ||
    editUnit ||
    editCorper ||
    editHoliday ||
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
    let active = true;
    (async () => {
      setDashboardLoading(true)
      try{
        await ensureCsrf()
        await refreshAll()
      }finally{
        if(active) setDashboardLoading(false)
      }
    })()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if(activeTab !== 'structure') return
    if(me?.role === 'ORG'){
      setStructureTab('profile')
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

  // Query pending badge should persist until fully resolved (OPEN)
  const pendingQueriesBadge = useMemo(() => {
    const list = Array.isArray(queries) ? queries : []
    if(me?.role === 'CORPER'){
      return list.filter(q => q.status === 'OPEN').length
    }
    if(me?.role === 'BRANCH'){
      // Branch sees their own branch queries; badge indicates anything still OPEN
      return list.filter(q => q.status === 'OPEN').length
    }
    return 0
  }, [queries, me?.role])

  function formatDateTime(dt){
    try{
      if(!dt) return ''
      const d = new Date(dt)
      if(Number.isNaN(d.getTime())) return String(dt)
      return d.toLocaleString(undefined, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' })
    }catch(e){
      return String(dt || '')
    }
  }

  function formatMoney(value){
    const n = Number(value || 0)
    if(!Number.isFinite(n) || n <= 0) return 'Free'
    return `₦${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  function extractApiMessage(err, fallback = 'Request failed'){
    const payload = err?.response?.data
    if(typeof payload === 'string') return payload
    if(payload?.detail) return String(payload.detail)
    try{
      const v = Object.values(payload || {})?.[0]
      if(Array.isArray(v)) return String(v[0] || fallback)
      if(v) return String(v)
    }catch(e){}
    return err?.message || fallback
  }

  function fieldError(errors, field){
    const value = errors?.[field]
    if(!value) return ''
    if(Array.isArray(value)){
      return value.map((item) => {
        if(item && typeof item === 'object') return Object.values(item).join(', ')
        return String(item || '')
      }).filter(Boolean).join(' ')
    }
    if(typeof value === 'object'){
      const first = Object.values(value).flat().find(Boolean)
      return first ? String(first) : JSON.stringify(value)
    }
    return String(value)
  }

  function clearFieldError(setter, field){
    setter((prev) => {
      if(!prev?.[field]) return prev
      const next = { ...(prev || {}) }
      delete next[field]
      return next
    })
  }

  const tourSteps = useMemo(() => {
    if(me?.role !== 'ORG') return []
    return [
      { key: 'structure', title: 'Set up your structure', body: 'Start here to confirm your organisation profile and create Head Office / branches.' },
      { key: 'corpers', title: 'Add corps members', body: 'Create and manage corps member accounts, placements, and assignments.' },
      { key: 'report', title: 'Generate reports', body: 'Download attendance and corper reports for any period.' },
      { key: 'wallet', title: 'Fund wallet', body: 'Fund your wallet to keep clearance services running without interruption.' },
      { key: 'subscription', title: 'Manage subscription', body: 'Subscriptions are separate from wallet funding and can be managed here.' },
    ]
  }, [me?.role])

  const setupItems = useMemo(() => {
    if(me?.role !== 'ORG') return []
    const hasBranches = Array.isArray(branches) && branches.length > 0
    const hasCorpers = Array.isArray(corpers) && corpers.length > 0
    const profileReady = !!((profile?.signatory_name || '').trim() || profile?.logo || profile?.signature)
    const walletReady = Number(wallet?.balance || 0) > 0
    const subscriptionActive = !!subscriptionInfo?.current?.is_active
    return [
      {
        key: 'profile',
        label: 'Update organisation profile',
        done: profileReady,
        action: () => { setActiveTab('structure'); setStructureTab('profile'); setShowEditProfile(true) }
      },
      {
        key: 'structure',
        label: 'Create Head Office / branches',
        done: hasBranches,
        action: () => { setActiveTab('structure'); setStructureTab('structure'); setShowAddBranch(true) }
      },
      {
        key: 'corpers',
        label: 'Add your first corps member',
        done: hasCorpers,
        action: () => { setActiveTab('corpers'); setShowAddCorper(true) }
      },
      {
        key: 'funding',
        label: 'Set up funding (wallet or subscription)',
        done: walletReady || subscriptionActive,
        action: () => { setActiveTab(walletReady ? 'wallet' : 'subscription') }
      },
    ]
  }, [me?.role, branches, corpers, profile?.signatory_name, profile?.logo, profile?.signature, wallet?.balance, subscriptionInfo?.current?.is_active])

  const setupProgress = useMemo(() => {
    if(!setupItems.length) return { done: 0, total: 0, pct: 0 }
    const done = setupItems.filter(i => i.done).length
    const total = setupItems.length
    const pct = Math.round((done / total) * 100)
    return { done, total, pct }
  }, [setupItems])

  function startTour(){
    if(me?.role !== 'ORG') return
    setTourStep(0)
    setTourOpen(true)
    try{
      localStorage.removeItem(`nysc_tour_dismissed:${me?.email || 'org'}`)
    }catch(e){}
  }

  useEffect(() => {
    if(!tourOpen) return
    const step = tourSteps[tourStep]
    if(!step) return
    try{
      document.querySelectorAll('.tour-highlight').forEach((el) => el.classList.remove('tour-highlight'))
      const target = document.querySelector(`[data-tour-key="${step.key}"]`)
      if(target){
        target.classList.add('tour-highlight')
        try{ target.scrollIntoView({ block:'center', behavior:'smooth' }) }catch(e){}
      }
    }catch(e){}
    return () => {
      try{ document.querySelectorAll('.tour-highlight').forEach((el) => el.classList.remove('tour-highlight')) }catch(e){}
    }
  }, [tourOpen, tourStep, tourSteps])

  useEffect(() => {
    if(me?.role !== 'ORG') return
    try{
      const key = `nysc_setup_dismissed:${me?.email || 'org'}`
      setSetupDismissed(localStorage.getItem(key) === '1')
    }catch(e){}
  }, [me?.role, me?.email])

  useEffect(() => {
    if(me?.role !== 'ORG') return
    if(dashboardLoading) return
    if(!tourSteps.length) return
    try{
      const doneKey = `nysc_tour_done:${me?.email || 'org'}`
      const dismissedKey = `nysc_tour_dismissed:${me?.email || 'org'}`
      const alreadyDone = localStorage.getItem(doneKey) === '1'
      const dismissed = localStorage.getItem(dismissedKey) === '1'
      if(!alreadyDone && !dismissed){
        setTourStep(0)
        setTourOpen(true)
      }
    }catch(e){}
  }, [me?.role, me?.email, dashboardLoading, tourSteps.length])

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
    const subscriptionPaystack = sp.get('subscription')
    if(subscriptionPaystack && reference){
      (async()=>{
        try{
          const vr = await api.post('/api/auth/subscriptions/verify/', { reference })
          if(vr.data?.status === 'success'){
            setStatus('saved:subscription')
            await refreshAll()
            setActiveTab('subscription')
          }else{
            setStatus('error:subscription')
          }
        }catch(err){
          setStatus(`error:subscription:${extractApiMessage(err, 'Subscription verification failed')}`)
        }
        const url = new URL(window.location.href)
        url.searchParams.delete('subscription'); url.searchParams.delete('reference')
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
      const [m,p,b,d,u,c,s,h,l,n,w,a,cl,ha,qy,sub] = await Promise.all([
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
        api.get('/api/auth/queries/').catch(()=>({data:[]})),
        api.get('/api/auth/subscriptions/status/').catch(()=>({data:null})),
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
      setQueries(Array.isArray(qy.data) ? qy.data : [])
      setWallet(w.data)
      setSubscriptionInfo(sub.data)
      setAnnouncement(a.data)
      setClearance(Array.isArray(cl.data)? cl.data : [])
      if(m.data?.role === 'CORPER'){
        try{ const r = await api.get('/api/auth/performance/summary/'); setPerf(r.data) }catch(e){}
      }
    }catch(e){ setStatus('error:failed to load') }
  }

  useEffect(() => {
    if(!me?.authenticated) return
    let active = true

    async function checkForConfigChanges(){
      try{
        const version = await fetchAdminConfigVersion()
        if(!active || !version) return
        if(!configVersionRef.current){
          configVersionRef.current = version
          return
        }
        if(version !== configVersionRef.current){
          configVersionRef.current = version
          await refreshAll()
          if(active) setStatus('saved:config-refresh')
        }
      }catch(e){}
    }

    checkForConfigChanges()
    const timer = window.setInterval(checkForConfigChanges, CONFIG_REFRESH_MS)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [me?.authenticated])

  async function saveProfile(e){
    e.preventDefault()
    setStatus('pending')
    setProfileFormErrors({})
    const form = new FormData(e.target)
    for(const field of ['logo', 'signature']){
      const file = form.get(field)
      if(typeof File !== 'undefined' && file instanceof File && file.size === 0 && !file.name){
        form.delete(field)
      }
    }
    try{
      const res = await api.put('/api/auth/profile/', form, { headers: { 'Content-Type':'multipart/form-data' } })
      setProfile(res.data)
      setStatus('saved:profile')
      return true
    }catch(err){
      const payload = err?.response?.data
      if(payload && typeof payload === 'object' && !Array.isArray(payload)){
        setProfileFormErrors(payload)
      }
      setStatus(`error:profile:${extractApiMessage(err, 'Failed to save organisation profile')}`)
    }
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
    const data = Object.fromEntries(form)
    try{ await api.post('/api/auth/departments/', data); await refreshAll(); setStatus('saved:department'); return true }catch(e){ setStatus('error:department') }
    return false
  }

  async function createUnit(e){
    e.preventDefault(); setStatus('pending')
    const form = new FormData(e.target)
    const data = Object.fromEntries(form)
    try{ await api.post('/api/auth/units/', data); await refreshAll(); setStatus('saved:unit'); return true }catch(e){ setStatus('error:unit') }
    return false
  }

  async function downloadImportTemplate(kind){
    const endpoint = kind === 'structure' ? '/api/auth/imports/structure/template/' : '/api/auth/imports/corpers/template/'
    const fallback = kind === 'structure' ? 'structure_import_template.xlsx' : 'corpers_import_template.xlsx'
    try{
      const res = await api.get(endpoint, { responseType: 'blob' })
      const blob = new Blob([res.data], { type: res.headers?.['content-type'] || 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
      const cd = res.headers?.['content-disposition'] || ''
      const match = cd.match(/filename="?([^";]+)"?/i)
      const filename = match?.[1] || fallback
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.URL.revokeObjectURL(url)
    }catch(err){
      setStatus(`error:import:${extractApiMessage(err, 'Failed to download template')}`)
    }
  }

  async function previewImport(kind, file){
    if(!file){
      setStatus(`error:import:Please select a CSV or Excel file first.`)
      return null
    }
    const endpoint = kind === 'structure' ? '/api/auth/imports/structure/' : '/api/auth/imports/corpers/'
    const form = new FormData()
    form.append('file', file)
    try{
      setStatus('pending')
      const res = await api.post(endpoint, form, { headers: { 'Content-Type': 'multipart/form-data' } })
      if(kind === 'structure') setStructureImportPreview(res.data)
      else setCorperImportPreview(res.data)
      setStatus(null)
      return res.data
    }catch(err){
      const payload = err?.response?.data
      if(payload && typeof payload === 'object'){
        if(kind === 'structure') setStructureImportPreview(payload)
        else setCorperImportPreview(payload)
      }
      setStatus(`error:import:${extractApiMessage(err, 'Failed to preview import')}`)
      return null
    }
  }

  async function applyImport(kind, file){
    if(!file){
      setStatus(`error:import:Please select a CSV or Excel file first.`)
      return
    }
    const endpoint = kind === 'structure' ? '/api/auth/imports/structure/' : '/api/auth/imports/corpers/'
    const form = new FormData()
    form.append('file', file)
    form.append('apply', '1')
    try{
      setStatus('pending')
      const res = await api.post(endpoint, form, { headers: { 'Content-Type': 'multipart/form-data' } })
      if(kind === 'structure'){
        setStructureImportPreview(res.data)
        setShowStructureImport(false)
        setStructureImportFile(null)
        setStatus('saved:structure-import')
      }else{
        setCorperImportPreview(res.data)
        setShowCorperImport(false)
        setCorperImportFile(null)
        setStatus('saved:corper-import')
      }
      await refreshAll()
    }catch(err){
      const payload = err?.response?.data
      if(payload && typeof payload === 'object'){
        if(kind === 'structure') setStructureImportPreview(payload)
        else setCorperImportPreview(payload)
      }
      setStatus(`error:import:${extractApiMessage(err, 'Failed to apply import')}`)
    }
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
    return deps
  }

  function enrollUnitOptions(){
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
    add('overview', 'Overview', Home)
    if(me?.role === 'ORG' || me?.role === 'BRANCH'){
      add('structure', 'Structure', Layers3)
      add('corpers', 'Corpers', Users)
    }
    if(me?.role === 'ORG'){
      add('wallet', 'Wallet', Building2)
      add('subscription', 'Subscription', CreditCard)
      add('clearance', 'Clearance', LayoutGrid)
      add('report', 'Reports', BarChart3)
    }
    if(me?.role === 'BRANCH'){
      add('wallet', 'Wallet', Building2)
      add('leave', 'Leaves', CalendarCheck2, leaves.filter(l=>l.status==='PENDING').length || null)
      add('query', 'Queries', FileSearch)
      add('report', 'Reports', BarChart3)
      add('clearance', 'Clearance', FileCheck2)
    }
    if(me?.role === 'CORPER'){
      add('attendance', 'Attendance', CalendarCheck2)
      add('leave', 'Leaves', CalendarDays)
      add('performance', 'Clearance', FileCheck2)
      add('wallet', 'Wallet', Wallet)
    }

    const notifBadge = (me?.role === 'CORPER')
      ? ((unreadNotifications || 0) + (pendingQueriesBadge || 0)) || null
      : ((pendingQueriesBadge || 0) || null)
    add('notifications', 'Notifications', Bell, notifBadge)
    return items
  }, [me?.role, leaves, notifications.length, unreadNotifications, pendingQueriesBadge])

  const tabTitle = useMemo(() => ({
    overview: 'Overview',
    structure: 'Structure',
    corpers: 'Corpers',
    wallet: 'Wallet',
    subscription: 'Subscription',
    clearance: 'Performance Clearance',
    leave: 'Leave Management',
    report: 'Report',
    attendance: 'Attendance',
    performance: 'Performance Clearance',
    notifications: 'Notifications',
  }), [])

  const clearanceFiltered = useMemo(() => {
    const q = clSearchOpen ? clQuery.trim().toLowerCase() : ''
    const list = Array.isArray(clearance) ? clearance : []
    return q ? list.filter(r => `${r.full_name} ${r.state_code} ${r.branch}`.toLowerCase().includes(q)) : list
  }, [clearance, clSearchOpen, clQuery])
  const clearanceTotalPages = Math.max(1, Math.ceil(clearanceFiltered.length / clPageSize))
  const clearanceCurrentPage = Math.min(clPage, clearanceTotalPages)
  const clearanceStart = (clearanceCurrentPage - 1) * clPageSize
  const clearanceRows = clearanceFiltered.slice(clearanceStart, clearanceStart + clPageSize)

  useEffect(() => {
    if(clPage > clearanceTotalPages) setClPage(clearanceTotalPages)
  }, [clPage, clearanceTotalPages])

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

    async function downloadWalletStatement(){
      try{
        const res = await api.get('/api/auth/wallet/export/', { responseType: 'blob' })
        const blob = new Blob([res.data], { type: res.headers?.['content-type'] || 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
        let filename = 'wallet_statement.xlsx'
        const cd = res.headers?.['content-disposition'] || ''
        const m = cd.match(/filename=\"?([^\";]+)\"?/i)
        if(m && m[1]) filename = m[1]
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filename
        document.body.appendChild(a)
        a.click()
        a.remove()
        window.URL.revokeObjectURL(url)
        setStatus('saved:wallet-export')
      }catch(e){
        setStatus('error:wallet-export')
      }
    }

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
              <div className="d-grid gap-2 mt-3">
                <button className="btn btn-olive" onClick={fundWallet}>Fund Wallet</button>
                <button className="btn btn-outline-secondary d-inline-flex align-items-center justify-content-center gap-2" type="button" onClick={downloadWalletStatement}>
                  <FileSpreadsheet size={16} /> Export Statement
                </button>
              </div>
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

              <div className="table-responsive mt-2 dash-table-scroll">
              <table className="table table-sm align-middle dash-table dash-table-auto dash-table-wallet mb-0">
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

  function SubscriptionSection(){
    const [cycle, setCycle] = useState('MONTHLY')
    const [loadingPlan, setLoadingPlan] = useState('')
    const plans = Array.isArray(subscriptionInfo?.plans) ? subscriptionInfo.plans : []
    const current = subscriptionInfo?.current || null
    const payments = Array.isArray(subscriptionInfo?.payments) ? subscriptionInfo.payments : []

    function planAmount(plan){
      return cycle === 'YEARLY' ? plan.yearly_price : plan.monthly_price
    }

    function planOriginal(plan){
      return cycle === 'YEARLY' ? plan.original_yearly_price : plan.original_monthly_price
    }

    function isCustomPricing(plan, original){
      return String(plan?.code || '').toUpperCase() === 'ENTERPRISE' && (Boolean(plan?.custom_pricing) || Number(original || 0) <= 0)
    }

    async function subscribe(plan){
      setLoadingPlan(plan.code)
      setStatus(null)
      try{
        const callback = `${window.location.origin}/dashboard?subscription=1`
        const res = await api.post('/api/auth/subscriptions/initialize/', {
          plan: plan.code,
          billing_cycle: cycle,
          callback_url: callback,
        })
        if(res.data?.free || res.data?.status === 'success'){
          await refreshAll()
          setActiveTab('subscription')
          setStatus('saved:subscription')
          return
        }
        const { authorization_url } = res.data || {}
        if(!authorization_url){
          setStatus('error:subscription')
          return
        }
        window.location.href = authorization_url
      }catch(err){
        setStatus(`error:subscription:${extractApiMessage(err, 'Failed to start subscription payment')}`)
      }finally{
        setLoadingPlan('')
      }
    }

    return (
      <div className="row g-3">
        <div className="col-12">
          <div className="card shadow-sm dash-card">
            <div className="card-body d-flex flex-wrap gap-3 justify-content-between align-items-start">
              <div>
                <div className="dash-card-title mb-1">Current Subscription</div>
                {current ? (
                  <>
                    <div className="h4 mb-1 text-olive">{current.plan_name} · {current.billing_cycle?.toLowerCase()}</div>
                    <div className="small text-muted">
                      Active from {formatDateTime(current.starts_at)} to {formatDateTime(current.expires_at)}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="h4 mb-1 text-olive">No active subscription</div>
                    <div className="small text-muted">Choose a plan below. Subscription payments are separate from wallet balance.</div>
                  </>
                )}
              </div>
              <div className="btn-group" role="group" aria-label="Billing cycle">
                <button className={`btn btn-sm ${cycle === 'MONTHLY' ? 'btn-olive' : 'btn-outline-secondary'}`} type="button" onClick={()=>setCycle('MONTHLY')}>Monthly</button>
                <button className={`btn btn-sm ${cycle === 'YEARLY' ? 'btn-olive' : 'btn-outline-secondary'}`} type="button" onClick={()=>setCycle('YEARLY')}>Yearly</button>
              </div>
            </div>
          </div>
        </div>

        <div className="col-12">
          <div className="row g-3">
            {plans.map((plan) => {
              const amount = planAmount(plan)
              const original = planOriginal(plan)
              const customPricing = isCustomPricing(plan, original)
              const discounted = !customPricing && Number(original || 0) > Number(amount || 0)
              const active = current?.plan_code === plan.code && current?.status === 'ACTIVE'
              return (
                <div className="col-md-6 col-xl-3" key={plan.code}>
                  <div className={`card shadow-sm dash-card h-100 ${String(plan.code).toUpperCase() === 'PRO' ? 'border-olive' : ''}`}>
                    <div className="card-body d-flex flex-column">
                      <div className="d-flex justify-content-between align-items-start gap-2">
                        <div>
                          <div className="small text-muted text-uppercase fw-bold">{plan.range_label}</div>
                          <div className="h5 mb-1">{plan.name}</div>
                        </div>
                        {active && <span className="badge bg-success">Active</span>}
                      </div>
                      <div className="mt-3">
                        <div className="display-6 fs-2 fw-bold text-olive mb-0">{customPricing ? 'Contact us for pricing' : formatMoney(amount)}</div>
                        <div className="small text-muted">{customPricing ? 'Custom enterprise plan' : cycle.toLowerCase()}</div>
                        {discounted && (
                          <div className="small text-muted">
                            <span className="text-decoration-line-through">{formatMoney(original)}</span> before discount
                          </div>
                        )}
                      </div>
                      {!customPricing && plan.discount_enabled && Number(plan.discount_percent || 0) > 0 && (
                        <div className="badge bg-light text-olive border mt-3 align-self-start">{Number(plan.discount_percent).toLocaleString()}% discount</div>
                      )}
                      <ul className="small text-muted mt-3 mb-4 ps-3">
                        <li>Attendance and clearance tools</li>
                        <li>Organization and admin access</li>
                        <li>Separate from wallet funding</li>
                      </ul>
                      {customPricing ? (
                        <button className="btn btn-olive mt-auto" type="button" onClick={()=>navigate('/contact')}>
                          Contact Us
                        </button>
                      ) : (
                        <button
                          className="btn btn-olive mt-auto"
                          type="button"
                          disabled={loadingPlan === plan.code}
                          onClick={()=>subscribe(plan)}
                        >
                          {loadingPlan === plan.code ? 'Processing…' : Number(amount || 0) <= 0 ? 'Activate Plan' : `Pay ${formatMoney(amount)}`}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
            {plans.length === 0 && (
              <div className="col-12">
                <div className="text-muted">Subscription plans are not available right now.</div>
              </div>
            )}
          </div>
        </div>

        <div className="col-12">
          <div className="card shadow-sm dash-card">
            <div className="card-body">
              <div className="dash-card-title mb-2">Subscription Payments</div>
              <div className="table-responsive dash-table-scroll">
                <table className="table table-sm align-middle dash-table dash-table-auto dash-table-subscription mb-0">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Plan</th>
                      <th>Cycle</th>
                      <th>Status</th>
                      <th className="text-end">Discount</th>
                      <th className="text-end">Paid</th>
                      <th>Reference</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payments.map((payment) => (
                      <tr key={payment.id || payment.reference}>
                        <td>{formatDateTime(payment.created_at)}</td>
                        <td>{payment.plan_name}</td>
                        <td>{payment.billing_cycle}</td>
                        <td>
                          {payment.status === 'SUCCESS' ? <span className="badge bg-success">SUCCESS</span> : payment.status === 'FAILED' ? <span className="badge bg-danger">FAILED</span> : <span className="badge bg-warning text-dark">PENDING</span>}
                        </td>
                        <td className="text-end">{formatMoney(payment.discount_amount)}</td>
                        <td className="text-end">{formatMoney(payment.amount_charged)}</td>
                        <td><div className="text-truncate dash-td-truncate-wide">{payment.reference}</div></td>
                      </tr>
                    ))}
                    {payments.length === 0 && (
                      <tr><td colSpan="7" className="text-muted">No subscription payments yet.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  function QuerySection({ role }){
    const [showAddQuery, setShowAddQuery] = useState(false)
    const [qSearchOpen, setQSearchOpen] = useState(false)
    const [qQuery, setQQuery] = useState('')
    const [qPage, setQPage] = useState(1)
    const [qPageSize, setQPageSize] = useState(20)
    const [qStatus, setQStatus] = useState('all')
    const [qAlert, setQAlert] = useState(null) // { type: 'success'|'danger', msg: string }

    function extractErrMsg(err){
      const payload = err?.response?.data
      if(!payload) return err?.message || 'Request failed'
      if(typeof payload === 'string') return payload
      if(payload.detail) return String(payload.detail)
      try{
        const v = Object.values(payload || {})?.[0]
        if(Array.isArray(v)) return String(v[0] || 'Request failed')
        if(v) return String(v)
      }catch(e){}
      return 'Request failed'
    }

    const visibleCorp = role === 'BRANCH'
      ? corpers
      : corpers

    const filtered = (() => {
      const q = qSearchOpen ? qQuery.trim().toLowerCase() : ''
      let out = Array.isArray(queries) ? queries : []
      if(q){
        out = out.filter(x => `${x.title||''} ${x.corper_name||''} ${x.corper_state_code||''} ${x.status||''}`.toLowerCase().includes(q))
      }
      if(qStatus !== 'all') out = out.filter(x => (x.status||'').toUpperCase() === qStatus)
      return out
    })()

    const totalPages = Math.max(1, Math.ceil(filtered.length / qPageSize))
    const current = Math.min(qPage, totalPages)
    const start = (current - 1) * qPageSize
    const rows = filtered.slice(start, start + qPageSize)

    return (
      <>
        <h2 className="mb-3 text-olive">Query Management</h2>

        <div className="card shadow-sm dash-card mb-3">
          <div className="card-body">
            <div className="dash-card-title mb-2">Auto Queries</div>
            <div className="small text-muted mb-3">Send queries in bulk for corpers who exceeded attendance thresholds for the previous month.</div>
            <div className="d-flex flex-wrap gap-2">
              <button className="btn btn-outline-secondary" type="button" onClick={async()=>{
                setStatus('pending')
                try{
                  const r = await api.post('/api/auth/queries/auto/', { kind: 'LATE' })
                  await refreshAll()
                  const created = Number(r?.data?.created ?? 0)
                  const skipped = Number(r?.data?.skipped ?? 0)
                  const ym = r?.data?.year_month
                  setQAlert({ type: 'success', msg: `Sent lateness queries${ym?` (${ym})`:''}: created ${created}, skipped ${skipped}.` })
                }catch(e){
                  setQAlert({ type: 'danger', msg: extractErrMsg(e) })
                }finally{
                  setStatus(null)
                }
              }}>Send lateness queries</button>
              <button className="btn btn-outline-secondary" type="button" onClick={async()=>{
                setStatus('pending')
                try{
                  const r = await api.post('/api/auth/queries/auto/', { kind: 'ABSENT' })
                  await refreshAll()
                  const created = Number(r?.data?.created ?? 0)
                  const skipped = Number(r?.data?.skipped ?? 0)
                  const ym = r?.data?.year_month
                  setQAlert({ type: 'success', msg: `Sent absence queries${ym?` (${ym})`:''}: created ${created}, skipped ${skipped}.` })
                }catch(e){
                  setQAlert({ type: 'danger', msg: extractErrMsg(e) })
                }finally{
                  setStatus(null)
                }
              }}>Send absence queries</button>
            </div>
          </div>
        </div>

        <div className="card shadow-sm dash-card">
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-center gap-2">
              <div className="dash-card-title mb-0">Queries</div>
              <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddQuery(true)}>New Query</button>
            </div>

            <div className="d-flex flex-wrap gap-2 justify-content-between align-items-center mt-3">
              <div className="d-flex align-items-center gap-2 flex-wrap">
                <button className={`btn btn-sm ${qSearchOpen ? 'btn-olive':'btn-outline-secondary'}`} type="button" onClick={()=>setQSearchOpen(v=>!v)} aria-label="Search">
                  <Search size={16} />
                </button>
                {qSearchOpen && (
                  <div className="dash-table-search">
                    <input className="form-control form-control-sm" placeholder="Search…" value={qQuery} onChange={(e)=>{ setQQuery(e.target.value); setQPage(1) }} />
                  </div>
                )}
                <select className="form-select form-select-sm" style={{width:140}} value={qStatus} onChange={(e)=>{ setQStatus(e.target.value); setQPage(1) }}>
                  <option value="all">All</option>
                  <option value="OPEN">Open</option>
                  <option value="RESOLVED">Resolved</option>
                </select>
              </div>
              <div className="d-flex align-items-center gap-2">
                <span className="small text-muted">Rows</span>
                <select className="form-select form-select-sm" style={{width:96}} value={qPageSize} onChange={(e)=>{ setQPageSize(Number(e.target.value)); setQPage(1) }}>
                  {[20,50,100].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>

            <div className="table-responsive mt-2">
              <table className="table table-sm align-middle dash-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Corper</th>
                    <th>Title</th>
                    <th>Status</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((x) => (
                    <tr key={x.id}>
                      <td className="small">{new Date(x.created_at).toLocaleString()}</td>
                      <td><div className="text-truncate dash-td-truncate-wide">{x.corper_name} ({x.corper_state_code})</div></td>
                      <td><div className="text-truncate dash-td-truncate-wide">{x.title}</div></td>
                      <td>{x.status === 'RESOLVED' ? <span className="badge bg-success">RESOLVED</span> : <span className="badge bg-warning text-dark">OPEN</span>}</td>
                      <td className="text-end">
                        <div className="btn-group">
                          {x.status !== 'RESOLVED' && (
                            <button className="btn btn-sm btn-outline-secondary" type="button" onClick={async()=>{
                              try{
                                await api.post(`/api/auth/queries/${x.id}/resolve/`)
                                await refreshAll()
                                setQAlert({ type: 'success', msg: 'Query resolved.' })
                              }catch(e){
                                setQAlert({ type: 'danger', msg: extractErrMsg(e) })
                              }
                            }}>Resolve</button>
                          )}
                          <button className="btn btn-sm btn-outline-danger" type="button" onClick={async()=>{
                            if(!confirm('Delete this query?')) return
                            try{
                              await api.delete(`/api/auth/queries/${x.id}/`)
                              await refreshAll()
                              setQAlert({ type: 'success', msg: 'Query deleted.' })
                            }catch(e){
                              setQAlert({ type: 'danger', msg: extractErrMsg(e) })
                            }
                          }}>Delete</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {filtered.length === 0 && <tr><td colSpan="5" className="text-muted">No queries found.</td></tr>}
                  {filtered.length > 0 && totalPages > 1 && (
                    <tr>
                      <td colSpan="5">
                        <div className="d-flex justify-content-between align-items-center">
                          <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                          <div className="btn-group">
                            <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setQPage(p=>Math.max(1,p-1))}>Prev</button>
                            <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setQPage(p=>Math.min(totalPages,p+1))}>Next</button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {qAlert?.type && (
              <AutoFadeAlert type={qAlert.type} onClose={()=>setQAlert(null)}>{qAlert.msg}</AutoFadeAlert>
            )}
          </div>
        </div>

        {showAddQuery && (
          <div className="dash-modal" onClick={()=>setShowAddQuery(false)}>
            <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
              <div className="dash-modal-head">
                <strong>New Query</strong>
                <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setShowAddQuery(false)}>Close</button>
              </div>
              <div className="dash-modal-body">
                <div className="dash-modal-grid">
                  <div className="dash-modal-help">
                    <h6>Flow</h6>
                    <p>Create a query to document an issue and track resolution.</p>
                  </div>
                  <div className="dash-modal-form">
                    <form onSubmit={async (e)=>{
                      e.preventDefault(); setStatus('pending')
                      const data = Object.fromEntries(new FormData(e.target))
                      try{
                        // Ensure corper is sent as an id (not empty string)
                        if(data.corper !== undefined) data.corper = String(data.corper || '').trim()
                        const res = await api.post('/api/auth/queries/', data)
                        await refreshAll()
                        const title = res?.data?.title || data.title || 'Query'
                        setQAlert({ type: 'success', msg: `${title} sent successfully.` })
                        setShowAddQuery(false)
                      }catch(err){
                        setQAlert({ type: 'danger', msg: extractErrMsg(err) })
                      }finally{
                        setStatus(null)
                      }
                    }}>
                      <label className="form-label">Corper</label>
                      <select className="form-select mb-2" name="corper" required>
                        <option value="">Select corper</option>
                        {visibleCorp.map(c => <option key={c.id} value={c.id}>{c.full_name} ({c.state_code})</option>)}
                      </select>
                      <label className="form-label">Title</label>
                      <input className="form-control mb-2" name="title" placeholder="Subject" required />
                      <label className="form-label">Message</label>
                      <textarea className="form-control" name="message" rows="4" placeholder="Details" />
                      <div className="dash-modal-actions">
                        <div className="d-grid">
                          <button className="btn btn-olive" disabled={isSaving}>{isSaving?'Saving…':'Create query'}</button>
                        </div>
                      </div>
                    </form>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </>
    )
  }

  function ReportSection(){
    const qp = []
    if(reportStart) qp.push(`start=${encodeURIComponent(reportStart)}`)
    if(reportEnd) qp.push(`end=${encodeURIComponent(reportEnd)}`)
    const qs = qp.length ? `?${qp.join('&')}` : ''
    const dailyEndpoint = `/api/auth/reports/attendance/${qs}`
    const corpersEndpoint = `/api/auth/reports/corpers/${qs}`
    const logsEndpoint = `/api/auth/reports/attendance/logs/${qs}`
    const downloadExcel = `/api/auth/reports/attendance/export/${qs}`
    const dailyRows = reportData?.daily?.rows || []
    const corperRows = reportData?.corpers?.rows || []
    const logRows = reportData?.logs?.rows || []
    const hasReport = !!reportData
    const summary = reportData?.daily?.summary || {}
    const reportRange = {
      start: summary.start || reportData?.corpers?.summary?.start || reportStart || '',
      end: summary.end || reportData?.corpers?.summary?.end || reportEnd || '',
    }
    const totalCheckins = Number(summary.total_checkins ?? dailyRows.reduce((sum, r) => sum + Number(r.checkins || 0), 0))
    const totalHours = Number(summary.total_hours ?? dailyRows.reduce((sum, r) => sum + Number(r.hours || 0), 0))
    const totalAbsent = corperRows.reduce((sum, r) => sum + Number(r.absent_days || 0), 0)
    const totalLate = corperRows.reduce((sum, r) => sum + Number(r.late_days || 0), 0)
    const reportHasRecords = totalCheckins > 0 || logRows.length > 0

    function reportErrorMessage(err){
      const payload = err?.response?.data
      if(typeof payload === 'string') return payload
      if(payload?.detail) return String(payload.detail)
      if(err?.response?.status === 403) return 'You do not have permission to generate this report.'
      if(err?.response?.status === 404) return 'The report endpoint was not found. Please refresh and try again.'
      if(err?.response?.status >= 500) return 'The server could not generate this report right now. Please try again later.'
      return err?.message || 'Unable to generate the report. Please try again.'
    }

    function inputDate(date){
      const pad = (n) => String(n).padStart(2, '0')
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
    }

    function displayDate(value){
      if(!value) return 'Not set'
      try{
        const d = new Date(`${value}T00:00:00`)
        if(Number.isNaN(d.getTime())) return value
        return d.toLocaleDateString(undefined, { month: 'short', day: '2-digit', year: 'numeric' })
      }catch(e){
        return value
      }
    }

    function setReportPreset(kind){
      const end = new Date()
      const start = new Date(end)
      if(kind === 'today'){
        // same day
      }else if(kind === 'last7'){
        start.setDate(end.getDate() - 6)
      }else if(kind === 'month'){
        start.setDate(1)
      }else{
        start.setDate(end.getDate() - 29)
      }
      setReportStart(inputDate(start))
      setReportEnd(inputDate(end))
    }

    async function downloadViaApi(path, fallbackName){
      setReportMessage('')
      try{
        const res = await api.get(path, { responseType: 'blob' })
        const blob = new Blob([res.data], { type: res.headers?.['content-type'] || 'application/octet-stream' })
        let filename = fallbackName
        const cd = res.headers?.['content-disposition'] || ''
        const m = cd.match(/filename=\"?([^\";]+)\"?/i)
        if(m && m[1]) filename = m[1]
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filename || fallbackName || 'download'
        document.body.appendChild(a)
        a.click()
        a.remove()
        window.URL.revokeObjectURL(url)
      }catch(e){
        setReportStatus('error')
        setReportMessage(reportErrorMessage(e))
      }
    }

    function downloadTextFile(content, filename, type = 'text/csv;charset=utf-8'){
      const blob = new Blob([content], { type })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    }

    function csvCell(value){
      const s = value === null || value === undefined ? '' : String(value)
      return `"${s.replace(/"/g, '""')}"`
    }

    function csvFromRows(headers, rows){
      return [
        headers.map(h => csvCell(h.label)).join(','),
        ...rows.map(row => headers.map(h => csvCell(row[h.key])).join(',')),
      ].join('\n')
    }

    async function getReportRows(kind){
      if(kind === 'daily'){
        const existing = reportData?.daily?.rows
        if(existing) return existing
        const res = await api.get(dailyEndpoint)
        return res.data?.rows || []
      }
      if(kind === 'corpers'){
        const existing = reportData?.corpers?.rows
        if(existing) return existing
        const res = await api.get(corpersEndpoint)
        return res.data?.rows || []
      }
      const existing = reportData?.logs?.rows
      if(existing) return existing
      const res = await api.get(logsEndpoint)
      return res.data?.rows || []
    }

    async function downloadCsvReport(kind){
      setReportMessage('')
      try{
        const rows = await getReportRows(kind)
        if(!rows.length){
          setReportStatus('empty')
          setReportMessage('No records were found for this CSV export in the selected period.')
          return
        }
        if(kind === 'daily'){
          const headers = [
            { key: 'date', label: 'date' },
            { key: 'checkins', label: 'checkins' },
          ]
          downloadTextFile(csvFromRows(headers, rows), 'daily.csv')
        }else if(kind === 'corpers'){
          const headers = [
            { key: 'full_name', label: 'full_name' },
            { key: 'state_code', label: 'state_code' },
            { key: 'email', label: 'email' },
            { key: 'branch', label: 'branch' },
            { key: 'department', label: 'department' },
            { key: 'unit', label: 'unit' },
            { key: 'working_days', label: 'working_days' },
            { key: 'present_days', label: 'present_days' },
            { key: 'absent_days', label: 'absent_days' },
            { key: 'late_days', label: 'late_days' },
            { key: 'hours', label: 'hours' },
          ]
          downloadTextFile(csvFromRows(headers, rows), 'corpers.csv')
        }else{
          const headers = [
            { key: 'date', label: 'date' },
            { key: 'time_in', label: 'time_in' },
            { key: 'time_out', label: 'time_out' },
            { key: 'full_name', label: 'full_name' },
            { key: 'state_code', label: 'state_code' },
            { key: 'branch', label: 'branch' },
          ]
          downloadTextFile(csvFromRows(headers, rows), 'logs.csv')
        }
        setReportStatus('success')
        setReportMessage('CSV download started.')
      }catch(e){
        setReportStatus('error')
        setReportMessage(reportErrorMessage(e))
      }
    }

    async function generateReport(){
      setReportStatus('pending')
      setReportMessage('')
      try{
        const [daily, corp, logs] = await Promise.all([
          api.get(dailyEndpoint),
          api.get(corpersEndpoint),
          api.get(logsEndpoint),
        ])
        const nextReport = { daily: daily.data, corpers: corp.data, logs: logs.data }
        const nextDailyRows = nextReport.daily?.rows || []
        const nextLogRows = nextReport.logs?.rows || []
        const nextSummary = nextReport.daily?.summary || {}
        const nextCheckins = Number(nextSummary.total_checkins ?? nextDailyRows.reduce((sum, r) => sum + Number(r.checkins || 0), 0))
        if(nextCheckins <= 0 && nextLogRows.length === 0){
          setReportData(nextReport)
          setReportStatus('empty')
          setReportMessage('No attendance records were found for the selected period.')
          return
        }
        setReportData(nextReport)
        setReportStatus('success')
        setReportMessage('Report generated and Excel download started.')
        await downloadViaApi(downloadExcel, 'attendance_report.xlsx')
      }catch(e){
        setReportStatus('error')
        setReportMessage(reportErrorMessage(e))
      }
    }

    return (
      <>
        <div className="dash-section-head mb-3">
          <div>
            <h2 className="mb-1 text-olive">Reports</h2>
            <div className="text-muted small">Review attendance performance, export clean records, and audit daily logs.</div>
          </div>
          {hasReport && (
            <div className="small text-muted text-md-end">
              {displayDate(reportRange.start)} - {displayDate(reportRange.end)}
            </div>
          )}
        </div>

        <div className="card shadow-sm dash-card mb-3">
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start gap-3 flex-wrap">
              <div>
                <div className="dash-card-title mb-1">Attendance Report</div>
                <div className="small text-muted">Choose a period, generate the report, then export daily, corper, logs, or full Excel workbook.</div>
              </div>
              <div className="d-flex gap-2 flex-wrap justify-content-end">
                <button className="btn btn-outline-secondary btn-sm d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadCsvReport('daily')}>
                  <Download size={15} /> Daily CSV
                </button>
                <button className="btn btn-outline-secondary btn-sm d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadCsvReport('corpers')}>
                  <Download size={15} /> Corpers CSV
                </button>
                <button className="btn btn-outline-secondary btn-sm d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadCsvReport('logs')}>
                  <Download size={15} /> Logs CSV
                </button>
                <button className="btn btn-outline-secondary btn-sm d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadViaApi(downloadExcel, 'attendance_report.xlsx')}>
                  <FileSpreadsheet size={15} /> Excel
                </button>
              </div>
            </div>

            <div className="d-flex flex-wrap gap-2 mt-3">
              <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setReportPreset('today')}>Today</button>
              <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setReportPreset('last7')}>Last 7 days</button>
              <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setReportPreset('last30')}>Last 30 days</button>
              <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setReportPreset('month')}>This month</button>
            </div>

            <div className="row g-2 mt-2 align-items-end">
              <div className="col-12 col-md-3">
                <label className="form-label">Start</label>
                <input className="form-control" type="date" value={reportStart} onChange={(e)=>setReportStart(e.target.value)} />
              </div>
              <div className="col-12 col-md-3">
                <label className="form-label">End</label>
                <input className="form-control" type="date" value={reportEnd} onChange={(e)=>setReportEnd(e.target.value)} />
              </div>
              <div className="col-12 col-md-3 d-grid">
                <button className="btn btn-olive d-inline-flex align-items-center justify-content-center gap-2" type="button" disabled={reportStatus==='pending'} onClick={generateReport}>
                  {reportStatus==='pending' ? <RefreshCw size={16} className="spin-icon" /> : <FileSpreadsheet size={16} />}
                  {reportStatus==='pending' ? 'Generating...' : 'Generate Excel'}
                </button>
              </div>
            </div>

            {reportStatus==='error' && (
              <AutoFadeAlert type="danger" onClose={()=>setReportStatus(null)}>{reportMessage || 'Unable to generate the report. Please try again.'}</AutoFadeAlert>
            )}
            {reportStatus==='empty' && (
              <AutoFadeAlert type="warning" onClose={()=>setReportStatus(null)}>{reportMessage || 'No attendance records were found for the selected period.'}</AutoFadeAlert>
            )}
            {reportStatus==='success' && (
              <AutoFadeAlert type="success" onClose={()=>setReportStatus(null)}>{reportMessage || 'Report generated and Excel download started.'}</AutoFadeAlert>
            )}
          </div>
        </div>

        {hasReport && reportHasRecords && (
          <div className="row g-3 mb-3">
            <div className="col-6 col-xl-3">
              <div className="dash-kpi h-100">
                <div className="dash-kpi-icon"><CalendarCheck2 size={18} /></div>
                <div>
                  <div className="dash-kpi-label">Check-ins</div>
                  <div className="dash-kpi-value">{totalCheckins.toLocaleString()}</div>
                </div>
              </div>
            </div>
            <div className="col-6 col-xl-3">
              <div className="dash-kpi h-100">
                <div className="dash-kpi-icon"><BarChart3 size={18} /></div>
                <div>
                  <div className="dash-kpi-label">Hours</div>
                  <div className="dash-kpi-value">{totalHours.toLocaleString(undefined, { maximumFractionDigits: 1 })}</div>
                </div>
              </div>
            </div>
            <div className="col-6 col-xl-3">
              <div className="dash-kpi h-100">
                <div className="dash-kpi-icon"><Users size={18} /></div>
                <div>
                  <div className="dash-kpi-label">Corpers</div>
                  <div className="dash-kpi-value">{corperRows.length.toLocaleString()}</div>
                </div>
              </div>
            </div>
            <div className="col-6 col-xl-3">
              <div className="dash-kpi h-100">
                <div className="dash-kpi-icon"><FileText size={18} /></div>
                <div>
                  <div className="dash-kpi-label">Absence / Late</div>
                  <div className="dash-kpi-value">{totalAbsent.toLocaleString()} / {totalLate.toLocaleString()}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {(!hasReport || (hasReport && !reportHasRecords)) && (
          <div className="dash-preview-empty" style={{height: 150}}>
            {hasReport ? 'No attendance records were found for the selected period.' : 'Generate a report to preview daily totals, corper summaries, and raw attendance logs.'}
          </div>
        )}

        {dailyRows.length > 0 && (
          <div className="card shadow-sm dash-card mb-3">
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-center gap-2 mb-2">
                <div>
                  <div className="dash-card-title mb-0">Daily Summary</div>
                  <div className="small text-muted">{dailyRows.length} day(s) in range</div>
                </div>
                <button className="btn btn-outline-secondary btn-sm d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadCsvReport('daily')}>
                  <Download size={15} /> Export
                </button>
              </div>
              <div className="table-responsive">
                <table className="table table-sm align-middle dash-table dash-table-auto mb-0">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th className="text-end">Check-ins</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dailyRows.map(r => (
                      <tr key={r.date}>
                        <td>{displayDate(r.date)}</td>
                        <td className="text-end">{r.checkins}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {corperRows.length > 0 && (
          <div className="card shadow-sm dash-card mb-3">
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-center gap-2 mb-2">
                <div>
                  <div className="dash-card-title mb-0">Corpers Report</div>
                  <div className="small text-muted">Presence, absence, lateness, and hours by corper</div>
                </div>
                <button className="btn btn-outline-secondary btn-sm d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadCsvReport('corpers')}>
                  <Download size={15} /> Export
                </button>
              </div>
              <div className="table-responsive">
                <table className="table table-sm align-middle dash-table dash-table-auto mb-0">
                  <thead>
                    <tr>
                      <th>Corper</th>
                      <th>State Code</th>
                      <th>Branch</th>
                      <th className="text-end">Working Days</th>
                      <th className="text-end">Present</th>
                      <th className="text-end">Absent</th>
                      <th className="text-end">Late</th>
                      <th className="text-end">Hours</th>
                    </tr>
                  </thead>
                  <tbody>
                    {corperRows.map(r => (
                      <tr key={r.corper_id}>
                        <td><div className="text-truncate dash-td-truncate-wide">{r.full_name}</div></td>
                        <td>{r.state_code}</td>
                        <td><div className="text-truncate dash-td-truncate">{r.branch || '—'}</div></td>
                        <td className="text-end">{r.working_days}</td>
                        <td className="text-end">{r.present_days}</td>
                        <td className="text-end">{r.absent_days}</td>
                        <td className="text-end">{r.late_days}</td>
                        <td className="text-end">{Number(r.hours || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {logRows.length > 0 && (
          <div className="card shadow-sm dash-card">
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-center gap-2 mb-2">
                <div>
                  <div className="dash-card-title mb-0">Attendance Logs</div>
                  <div className="small text-muted">{logRows.length} raw record(s)</div>
                </div>
                <button className="btn btn-outline-secondary btn-sm d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadCsvReport('logs')}>
                  <Download size={15} /> Export
                </button>
              </div>
              <div className="table-responsive">
                <table className="table table-sm align-middle dash-table dash-table-auto mb-0">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Corper</th>
                      <th>State Code</th>
                      <th>Branch</th>
                      <th>Time in</th>
                      <th>Time out</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logRows.map((r, idx) => (
                      <tr key={`${r.date}-${idx}`}>
                        <td>{displayDate(r.date)}</td>
                        <td><div className="text-truncate dash-td-truncate-wide">{r.full_name}</div></td>
                        <td>{r.state_code}</td>
                        <td><div className="text-truncate dash-td-truncate">{r.branch || '—'}</div></td>
                        <td>{r.time_in || '—'}</td>
                        <td>{r.time_out || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </>
    )
	  }

  function ImportPreviewPanel({ preview }){
    if(!preview) return null
    const rows = Array.isArray(preview.rows) ? preview.rows : []
    const summary = preview.summary || {}
    const firstRows = rows.slice(0, 12)
    return (
      <div className="mt-3">
        <div className={`alert ${preview.errors_count ? 'alert-warning' : 'alert-success'} py-2 mb-3`}>
          <div className="fw-semibold">{preview.errors_count ? `${preview.errors_count} issue(s) need attention` : 'Preview looks good'}</div>
          <div className="small">
            {Object.entries(summary).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${value}`).join(' · ')}
          </div>
        </div>
        {firstRows.length > 0 && (
          <div className="table-responsive">
            <table className="table table-sm align-middle dash-table mb-0">
              <thead>
                <tr>
                  <th>Row</th>
                  <th>Name</th>
                  <th>Branch</th>
                  <th>Department</th>
                  <th>Unit</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {firstRows.map((row) => (
                  <tr key={row.row}>
                    <td>{row.row}</td>
                    <td>{row.full_name || row.branch_name || '—'}</td>
                    <td>{row.branch_name || '—'}</td>
                    <td>{row.department_name || '—'}</td>
                    <td>{row.unit_name || '—'}</td>
                    <td>
                      {row.status === 'error' ? (
                        <span className="text-danger small">{(row.messages || []).join('; ')}</span>
                      ) : (
                        <span className="text-success small">Ready</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  if(dashboardLoading){
    return <DashboardLoadingScreen />
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

        {showBranchGeoTour && me?.role === 'BRANCH' && (
          <div className="dash-modal" onClick={()=>setShowBranchGeoTour(false)}>
            <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
              <div className="dash-modal-head">
                <strong>Update your branch location</strong>
                <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setShowBranchGeoTour(false)}>Close</button>
              </div>
              <div className="dash-modal-body">
                <div className="dash-modal-grid">
                  <div className="dash-modal-help">
                    <h6>Why this matters</h6>
                    <p>Setting your branch coordinates helps geofencing work accurately for attendance.</p>
                    <ul>
                      <li>Pick the location on the map</li>
                      <li>Add or confirm the branch address</li>
                      <li>Save to apply immediately</li>
                    </ul>
                  </div>
                  <div className="dash-modal-form">
                    <div className="d-grid gap-2">
                      <button className="btn btn-olive" type="button" onClick={()=>{
                        const myBranch = branches.find(x => x.admin_info && x.admin_info.email === me?.email) || branches[0]
                        setActiveTab('structure')
                        setStructureTab('branch')
                        setBranchLocationPos(myBranch?.latitude && myBranch?.longitude ? { lat: myBranch.latitude, lng: myBranch.longitude } : null)
                        setBranchLocationAddress(myBranch?.address || '')
                        setBranchLocationAddressTouched(false)
                        setShowBranchGeoTour(false)
                        setShowBranchLocation(true)
                      }}>Update location now</button>
                      <button className="btn btn-outline-secondary" type="button" onClick={()=>setShowBranchGeoTour(false)}>Later</button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {showCorperTour && me?.role === 'CORPER' && (
          <div className="dash-modal" onClick={()=>setShowCorperTour(false)}>
            <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
              <div className="dash-modal-head">
                <strong>Welcome</strong>
                <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setShowCorperTour(false)}>Close</button>
              </div>
              <div className="dash-modal-body">
                <div className="dash-modal-grid">
                  <div className="dash-modal-help">
                    <h6>Quick guide</h6>
                    <ul>
                      <li>Use <strong>Attendance</strong> to clock-in and clock-out</li>
                      <li>Keep your face capture clear and well-lit</li>
                      <li>Download your clearance letter when qualified</li>
                    </ul>
                  </div>
                  <div className="dash-modal-form">
                    <div className="d-grid gap-2">
                      <button className="btn btn-olive" type="button" onClick={()=>setShowCorperTour(false)}>Got it</button>
                    </div>
                    <div className="form-text mt-2">This guide shows during your first three logins.</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
	      {tourOpen && tourSteps[tourStep] && (
	        <div className="tour-backdrop" onClick={() => {
	          try{ localStorage.setItem(`nysc_tour_dismissed:${me?.email || 'org'}`, '1') }catch(e){}
	          setTourOpen(false)
	        }}>
	          <div className="tour-card" onClick={(e)=>e.stopPropagation()}>
	            <div className="d-flex justify-content-between align-items-start gap-3">
	              <div className="min-w-0">
	                <div className="tour-kicker">Quick tour • Step {tourStep + 1} of {tourSteps.length}</div>
	                <div className="tour-title">{tourSteps[tourStep].title}</div>
	                <div className="tour-body">{tourSteps[tourStep].body}</div>
	              </div>
	              <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => {
	                try{ localStorage.setItem(`nysc_tour_dismissed:${me?.email || 'org'}`, '1') }catch(e){}
	                setTourOpen(false)
	              }}>Skip</button>
	            </div>
	            <div className="d-flex justify-content-between align-items-center gap-2 mt-3">
	              <button className="btn btn-outline-secondary" type="button" disabled={tourStep === 0} onClick={() => setTourStep((s)=>Math.max(0, s - 1))}>
	                Back
	              </button>
	              {tourStep < tourSteps.length - 1 ? (
	                <button className="btn btn-olive" type="button" onClick={() => setTourStep((s)=>Math.min(tourSteps.length - 1, s + 1))}>
	                  Next
	                </button>
	              ) : (
	                <button className="btn btn-olive" type="button" onClick={() => {
	                  try{ localStorage.setItem(`nysc_tour_done:${me?.email || 'org'}`, '1') }catch(e){}
	                  setTourOpen(false)
	                }}>
	                  Done
	                </button>
	              )}
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
	                data-tour-key={key}
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
        {status==='saved:wallet-export' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Wallet statement download started.</AutoFadeAlert>
        )}
        {status==='error:wallet-export' && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Failed to export wallet statement.</AutoFadeAlert>
        )}
        {status==='saved:config-refresh' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Latest admin settings applied.</AutoFadeAlert>
        )}
        {status==='saved:profile' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Organisation profile saved successfully.</AutoFadeAlert>
        )}
        {status?.startsWith('error:profile:') && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>{status.split(':').slice(2).join(':')}</AutoFadeAlert>
        )}
        {status==='saved:structure-import' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Structure import completed.</AutoFadeAlert>
        )}
        {status==='saved:corper-import' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Corper import completed. Face capture remains live per corper.</AutoFadeAlert>
        )}
        {status?.startsWith('error:import:') && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>{status.split(':').slice(2).join(':')}</AutoFadeAlert>
        )}
        {status==='saved:subscription' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Subscription updated successfully.</AutoFadeAlert>
        )}
        {status==='error:subscription' && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Subscription payment could not be completed.</AutoFadeAlert>
        )}
        {status?.startsWith('error:subscription:') && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>{status.split(':').slice(2).join(':')}</AutoFadeAlert>
        )}
        {status==='saved:query-reply' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Reply sent. Awaiting admin resolution.</AutoFadeAlert>
        )}
        {status==='error:query-reply' && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Failed to send reply. Please try again.</AutoFadeAlert>
        )}
        {status==='saved:query-resolve' && (
          <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Query resolved.</AutoFadeAlert>
        )}
        {status==='error:query-resolve' && (
          <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Failed to resolve query.</AutoFadeAlert>
        )}
	          {activeTab==='overview' && (
	            <>
	              <div className="row g-3">
	                {me?.role === 'ORG' && !setupDismissed && setupItems.length > 0 && (
	                  <div className="col-12">
	                    <div className="card shadow-sm dash-card">
	                      <div className="card-body">
	                        <div className="d-flex justify-content-between align-items-start gap-3 flex-wrap">
	                          <div className="min-w-0">
	                            <div className="dash-card-title mb-1">Getting started</div>
	                            <div className="small text-muted">
	                              Complete these steps to finish setup ({setupProgress.done}/{setupProgress.total}).
	                            </div>
	                          </div>
	                          <div className="d-flex gap-2">
	                            <button className="btn btn-sm btn-outline-secondary" type="button" onClick={startTour}>Take a tour</button>
	                            <button className="btn btn-sm btn-outline-secondary" type="button" aria-label="Dismiss" onClick={() => {
	                              setSetupDismissed(true)
	                              try{ localStorage.setItem(`nysc_setup_dismissed:${me?.email || 'org'}`, '1') }catch(e){}
	                            }}>×</button>
	                          </div>
	                        </div>
	                        <div className="progress mt-3" style={{height:8}}>
	                          <div className="progress-bar bg-olive" role="progressbar" style={{width:`${setupProgress.pct}%`}} aria-valuenow={setupProgress.pct} aria-valuemin="0" aria-valuemax="100" />
	                        </div>
	                        <div className="row g-2 mt-3">
	                          {setupItems.map((item) => (
	                            <div className="col-12 col-md-6 col-xl-3" key={item.key}>
	                              <button
	                                type="button"
	                                className={`btn w-100 text-start d-flex align-items-center justify-content-between gap-2 btn-outline-secondary ${item.done ? '' : 'border-olive text-olive'}`}
	                                onClick={item.action}
	                              >
	                                <span className="text-truncate">{item.label}</span>
	                                <span className={`badge ${item.done ? 'bg-success' : 'bg-secondary'}`}>{item.done ? 'Done' : 'Start'}</span>
	                              </button>
	                            </div>
	                          ))}
	                        </div>
	                      </div>
	                    </div>
	                  </div>
	                )}
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
                <>
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

                  {me?.role === 'BRANCH' && (
                  <div className="card shadow-sm dash-card mb-3">
                    <div className="card-body">
                      <div className="d-flex justify-content-between align-items-center gap-2 flex-wrap">
                        <div>
                          <div className="dash-card-title mb-0">Query replies</div>
                          <div className="small text-muted">Pending queries (remain pending until resolved).</div>
                        </div>
                        {me?.role === 'BRANCH' && (
                          <button className="btn btn-outline-secondary btn-sm" type="button" onClick={()=>setActiveTab('query')}>Open queries</button>
                        )}
                      </div>
                      <div className="dash-feed mt-3">
                        {queries.filter(q => (q.status === 'OPEN')).map((q) => (
                          <div key={q.id} className="dash-feed-item">
                            <div className="d-flex justify-content-between align-items-start gap-2">
                              <div className="fw-semibold">{q.title}</div>
                              <div className="small text-muted">{formatDateTime(q.replied_at || q.created_at)}</div>
                            </div>
                            <div className="small text-muted">
                              {q.corper_name} ({q.corper_state_code})
                              {q.replied_at ? ' · replied' : ' · sent'}
                            </div>
                            {q.message && (
                              <div className="small mt-2" style={{whiteSpace:'pre-wrap'}}>
                                <span className="text-muted">Query:</span> {q.message}
                              </div>
                            )}
                            {q.replied_at && (
                              <div className="small mt-2" style={{whiteSpace:'pre-wrap'}}>
                                <span className="text-muted">Reply:</span> {q.corper_reply || '—'}
                              </div>
                            )}
                            <div className="d-flex gap-2 mt-2">
                              <button className="btn btn-olive btn-sm" type="button" onClick={async()=>{
                                try{ await api.post(`/api/auth/queries/${q.id}/resolve/`); await refreshAll(); setStatus('saved:query-resolve') }catch(e){ setStatus('error:query-resolve') }
                              }}>Resolve</button>
                            </div>
                          </div>
                        ))}
                        {queries.filter(q => (q.status === 'OPEN')).length === 0 && (
                          <div className="text-muted">No query replies yet.</div>
                        )}
                      </div>
                    </div>
                  </div>
                  )}
                </>
              )}

              {status === 'saved:notification' && (
                <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Notification sent.</AutoFadeAlert>
              )}
              {status === 'error:notification' && (
                <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Failed to send notification.</AutoFadeAlert>
              )}

              {me?.role === 'CORPER' && (
                <>
                  <div className="card shadow-sm dash-card mb-3">
                    <div className="card-body">
                      <div className="dash-card-title">Pending queries</div>
                      <div className="small text-muted mb-2">Queries remain pending until resolved by admin.</div>
                      <div className="dash-feed">
                        {queries.filter(q => (q.status === 'OPEN')).map((q) => (
                          <div key={q.id} className="dash-feed-item">
                            <div className="d-flex justify-content-between align-items-start gap-2">
                              <div className="fw-semibold">{q.title}</div>
                              <div className="small text-muted">{formatDateTime(q.created_at)}</div>
                            </div>
                            {q.message && (
                              <div className="small mt-2" style={{whiteSpace:'pre-wrap'}}>
                                <span className="text-muted">Query:</span> {q.message}
                              </div>
                            )}
                            {q.replied_at ? (
                              <div className="small mt-2" style={{whiteSpace:'pre-wrap'}}>
                                <div className="text-muted">Your reply ({formatDateTime(q.replied_at)}):</div>
                                <div>{q.corper_reply || '—'}</div>
                                <div className="text-muted mt-2">Status: awaiting admin resolution.</div>
                              </div>
                            ) : (
                              <>
                                <textarea
                                  className="form-control form-control-sm mt-2"
                                  rows="3"
                                  placeholder="Type your reply…"
                                  value={queryReplyDrafts[q.id] || ''}
                                  onChange={(e)=>setQueryReplyDrafts((m)=>({ ...m, [q.id]: e.target.value }))}
                                />
                                <div className="d-flex justify-content-end mt-2">
                                  <button className="btn btn-olive btn-sm" type="button" onClick={async()=>{
                                    const reply = String(queryReplyDrafts[q.id] || '').trim()
                                    if(!reply){ alert('Reply is required'); return }
                                    try{
                                      await api.post(`/api/auth/queries/${q.id}/reply/`, { reply })
                                      setQueryReplyDrafts((m)=>{ const x = { ...m }; delete x[q.id]; return x })
                                      await refreshAll()
                                      setStatus('saved:query-reply')
                                    }catch(e){
                                      setStatus('error:query-reply')
                                    }
                                  }}>Send reply</button>
                                </div>
                              </>
                            )}
                          </div>
                        ))}
                        {queries.filter(q => (q.status === 'OPEN')).length === 0 && (
                          <div className="text-muted">No pending queries.</div>
                        )}
                      </div>
                    </div>
                  </div>

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
                </>
              )}
            </>
          )}

          {activeTab==='structure' && me?.role==='ORG' && (
            <>
              <h2 className="mb-3 text-olive">Structure</h2>
              {forceStructureSetup && (
                <div className="alert alert-warning">
                  <div className="fw-semibold">Please complete your organisation setup</div>
                  <div className="small">
                    Start with your organisation profile, then create your Head Office and assign an admin for daily attendance operations.
                  </div>
                  <div className="small mt-1">
                    Flow: update profile, create Head Office, assign admin, admin verifies email, then add departments, units, and corpers.
                  </div>
                  <div className="mt-2 d-flex gap-2 flex-wrap">
                    <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setShowEditProfile(true)}>
                      Review Profile
                    </button>
                    <button className="btn btn-sm btn-olive" type="button" onClick={()=>{
                      setNewBranchForm((p) => ({ ...p, name: p.name || 'Head Office' }))
                      setShowAddBranch(true)
                    }}>
                      Create Head Office
                    </button>
                  </div>
                </div>
              )}

              <div className="dash-struct-nav mb-3">
                <button className={`dash-struct-item ${structureTab==='profile'?'active':''}`} type="button" onClick={()=>setStructureTab('profile')}>Organisation Profile</button>
                <button className={`dash-struct-item ${structureTab==='structure'?'active':''}`} type="button" onClick={()=>setStructureTab('structure')}>Structure</button>
                <button className={`dash-struct-item ${structureTab==='holidays'?'active':''}`} type="button" onClick={()=>setStructureTab('holidays')}>Holidays</button>
              </div>

              <div className="card shadow-sm dash-card">
                <div className="card-body">
                  <div className="d-flex justify-content-between align-items-center gap-2">
                    <div className="dash-card-title mb-0">
                      {structureTab==='structure' ? 'Structure' : structureTab==='holidays' ? 'Holidays' : 'Organisation Profile'}
                    </div>
	                    <div className="d-flex gap-2 flex-wrap justify-content-end">
	                      <button className="btn btn-sm btn-outline-secondary d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadImportTemplate('structure')}>
	                        <Download size={15} /> Template
	                      </button>
	                      <button className="btn btn-sm btn-outline-secondary d-inline-flex align-items-center gap-1" type="button" onClick={()=>setShowStructureImport(true)}>
	                        <FileSpreadsheet size={15} /> Import
	                      </button>
	                      {structureTab==='structure' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddBranch(true)}>Add Branch</button>}
	                      {structureTab==='structure' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddDepartment(true)}>Add Department</button>}
	                      {structureTab==='structure' && <button className="btn btn-sm btn-olive" type="button" onClick={()=>setShowAddUnit(true)}>Add Unit</button>}
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
                              setStructBranchesPage(1)
                              setStructDepartmentsPage(1)
                              setStructUnitsPage(1)
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
                          setStructBranchesPage(1)
                          setStructDepartmentsPage(1)
                          setStructUnitsPage(1)
                        }}
                        aria-label="Sort by"
                      >
                        {structureTab === 'structure' && (
                          <>
                            <option value="name">Sort: Name</option>
                            <option value="address">Sort: Address</option>
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
                          setStructBranchesPage(1)
                          setStructDepartmentsPage(1)
                          setStructUnitsPage(1)
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
                          setStructBranchesPage(1)
                          setStructDepartmentsPage(1)
                          setStructUnitsPage(1)
                        }}
                      >
                        {[20, 50, 100].map((n) => (
                          <option key={n} value={n}>
                            {n}
                          </option>
                        ))}
                      </select>
                      {structureTab === 'holidays' && <span className="small text-muted">Page {structPage}</span>}
                    </div>
                  </div>
                  )}

                  <div className={structureTab === 'profile' ? 'mt-3' : 'mt-2'}>
                    {structureTab==='structure' && (
                      <>
                        <div className="mb-4">
                          <div className="fw-semibold mb-2">Branches</div>
                          <div className="table-responsive">
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
                                  const current = Math.min(structBranchesPage, totalPages)
                                  if (current !== structBranchesPage) setStructBranchesPage(current)
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
                                                  if (!confirm(`Delete branch "${b.name}"? Corps members in this branch may be affected.`)) return
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
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructBranchesPage(p=>Math.max(1,p-1))}>Prev</button>
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructBranchesPage(p=>Math.min(totalPages,p+1))}>Next</button>
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
                        </div>

                        <div className="mb-4">
                          <div className="fw-semibold mb-2">Departments</div>
                          <div className="table-responsive">
                            <table className="table table-sm align-middle dash-table">
                              <thead><tr><th>Name</th><th></th></tr></thead>
                              <tbody>
                                {(() => {
                                  const q = structSearchOpen ? structQuery.trim().toLowerCase() : ''
                                  let filtered = q ? deps.filter((d) => `${d.name}`.toLowerCase().includes(q)) : deps
                                  const cmp = (a, b) => {
                                    const dir = structSortDir === 'desc' ? -1 : 1
                                    const av = (a.name || '').toLowerCase()
                                    const bv = (b.name || '').toLowerCase()
                                    return av.localeCompare(bv) * dir
                                  }
                                  filtered = [...filtered].sort(cmp)
                                  const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                                  const current = Math.min(structDepartmentsPage, totalPages)
                                  if (current !== structDepartmentsPage) setStructDepartmentsPage(current)
                                  const start = (current - 1) * structPageSize
                                  const rows = filtered.slice(start, start + structPageSize)
                                  return (
                                    <>
                                      {rows.map((d) => (
                                        <tr key={d.id}>
                                          <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{d.name}</div></td>
                                          <td className="text-end">
                                            <div className="btn-group">
                                              <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setEditDepartment(d)} aria-label="Edit department">
                                                <Pencil size={16} />
                                              </button>
                                              <button
                                                className="btn btn-sm btn-outline-danger"
                                                type="button"
                                                onClick={async () => {
                                                  if (!confirm(`Delete department "${d.name}"?`)) return
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
                                      {filtered.length===0 && <tr><td colSpan="2" className="text-muted">No departments found.</td></tr>}
                                      {filtered.length>0 && (
                                        <tr>
                                          <td colSpan="2">
                                            <div className="d-flex justify-content-between align-items-center">
                                              <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                              <div className="btn-group">
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructDepartmentsPage(p=>Math.max(1,p-1))}>Prev</button>
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructDepartmentsPage(p=>Math.min(totalPages,p+1))}>Next</button>
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
                        </div>

                        <div className="mb-2">
                          <div className="fw-semibold mb-2">Units</div>
                          <div className="table-responsive">
                            <table className="table table-sm align-middle dash-table">
                              <thead><tr><th>Name</th><th></th></tr></thead>
                              <tbody>
                                {(() => {
                                  const q = structSearchOpen ? structQuery.trim().toLowerCase() : ''
                                  let filtered = q ? units.filter((u) => `${u.name}`.toLowerCase().includes(q)) : units
                                  const cmp = (a, b) => {
                                    const dir = structSortDir === 'desc' ? -1 : 1
                                    const av = (a.name || '').toLowerCase()
                                    const bv = (b.name || '').toLowerCase()
                                    return av.localeCompare(bv) * dir
                                  }
                                  filtered = [...filtered].sort(cmp)
                                  const totalPages = Math.max(1, Math.ceil(filtered.length / structPageSize))
                                  const current = Math.min(structUnitsPage, totalPages)
                                  if (current !== structUnitsPage) setStructUnitsPage(current)
                                  const start = (current - 1) * structPageSize
                                  const rows = filtered.slice(start, start + structPageSize)
                                  return (
                                    <>
                                      {rows.map((u) => (
                                        <tr key={u.id}>
                                          <td className="fw-semibold"><div className="text-truncate dash-td-truncate">{u.name}</div></td>
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
                                      {filtered.length===0 && <tr><td colSpan="2" className="text-muted">No units found.</td></tr>}
                                      {filtered.length>0 && (
                                        <tr>
                                          <td colSpan="2">
                                            <div className="d-flex justify-content-between align-items-center">
                                              <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                              <div className="btn-group">
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setStructUnitsPage(p=>Math.max(1,p-1))}>Prev</button>
                                                <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setStructUnitsPage(p=>Math.min(totalPages,p+1))}>Next</button>
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
                        </div>
                      </>
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
                <div
                  className="dash-modal"
                  onClick={() => {
                    setShowAddBranch(false)
                  }}
                >
                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
                    <div className="dash-modal-head">
                      <strong>{forceStructureSetup ? 'Create Head Office' : 'Add Branch'}</strong>
                      <button
                        className="btn btn-sm btn-outline-secondary"
                        type="button"
                        onClick={() => setShowAddBranch(false)}
                      >
                        Close
                      </button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>How it works</h6>
                          <p>Create your Head Office (HQ) first so you can complete setup and start adding corpers.</p>
                          <ul>
                            <li>Create “Head Office” (or any HQ name you prefer).</li>
                            <li>Assign an admin by name and email.</li>
                            <li>Admin verifies email → sets password → manages daily operations.</li>
                            <li>Set the required coordinates for attendance verification.</li>
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
                                setForceStructureSetup(false)
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
                                  placeholder={forceStructureSetup ? 'Required' : 'Optional'}
                                  required={forceStructureSetup}
                                />
                              </div>
                              <div className="col-md-5">
                                <label className="form-label">Admin Email</label>
                                <input
                                  className="form-control"
                                  type="email"
                                  value={newBranchForm.admin_email}
                                  onChange={(e) => setNewBranchForm((p) => ({ ...p, admin_email: e.target.value }))}
                                  placeholder={forceStructureSetup ? 'Required' : 'Optional'}
                                  required={forceStructureSetup}
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
	                          <p>Create departments once per organisation. You can assign corps members during enrolment.</p>
                          <ul>
                            <li>Enter a department name.</li>
                            <li>Manage units separately if you use them.</li>
                          </ul>
                        </div>
	                        <div className="dash-modal-form">
	                          <form onSubmit={async (e)=>{ const ok = await createDepartment(e); if(ok) setShowAddDepartment(false) }}>
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
                          <p>Units are optional groups you can assign during enrolment.</p>
                          <ul>
                            <li>Enter a unit name.</li>
                            <li>Assign corps members later during enrolment.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{ const ok = await createUnit(e); if(ok) setShowAddUnit(false) }}>
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

	              {showStructureImport && (
	                <div className="dash-modal" onClick={() => setShowStructureImport(false)}>
	                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
	                    <div className="dash-modal-head">
	                      <strong>Import Structure</strong>
	                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={() => setShowStructureImport(false)}>Close</button>
	                    </div>
	                    <div className="dash-modal-body">
	                      <div className="dash-modal-grid">
	                        <div className="dash-modal-help">
	                          <h6>Bulk setup</h6>
	                          <p>Upload branches, departments, and units in one file. Departments and units are organisation-wide.</p>
	                          <button className="btn btn-sm btn-outline-secondary d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadImportTemplate('structure')}>
	                            <Download size={15} /> Download template
	                          </button>
	                        </div>
	                        <div className="dash-modal-form">
	                          <label className="form-label">CSV or Excel file</label>
	                          <input
	                            className="form-control"
	                            type="file"
	                            accept=".xlsx,.xlsm,.csv,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	                            onChange={(e)=>{ setStructureImportFile(e.target.files?.[0] || null); setStructureImportPreview(null) }}
	                          />
	                          <div className="d-flex justify-content-end gap-2 mt-3">
	                            <button className="btn btn-outline-secondary" type="button" onClick={()=>previewImport('structure', structureImportFile)}>
	                              Preview
	                            </button>
	                            <button className="btn btn-olive" type="button" disabled={!structureImportPreview || structureImportPreview.errors_count > 0 || isSaving} onClick={()=>applyImport('structure', structureImportFile)}>
	                              {isSaving ? 'Applying...' : 'Apply Import'}
	                            </button>
	                          </div>
	                          <ImportPreviewPanel preview={structureImportPreview} />
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
                            <li>Attendance coordinates are completed on Head Office and branch records.</li>
                          </ul>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{ const ok = await saveProfile(e); if(ok) setShowEditProfile(false) }} encType="multipart/form-data">
                            <div className="dash-form-section">
                              <div className="dash-form-title">Branding & Sign-off</div>
                              <div className="row g-2">
                                <div className="col-12">
                                  <label className="form-label">Director HR Name</label>
                                  <input
                                    className={`form-control ${fieldError(profileFormErrors, 'signatory_name') ? 'is-invalid' : ''}`}
                                    type="text"
                                    name="signatory_name"
                                    defaultValue={profile?.signatory_name || ''}
                                    onChange={() => clearFieldError(setProfileFormErrors, 'signatory_name')}
                                  />
                                  {fieldError(profileFormErrors, 'signatory_name') && (
                                    <div className="invalid-feedback">{fieldError(profileFormErrors, 'signatory_name')}</div>
                                  )}
                                </div>
                                <div className="col-12">
                                  <label className="form-label">Logo</label>
                                  <input
                                    className={`form-control ${fieldError(profileFormErrors, 'logo') ? 'is-invalid' : ''}`}
                                    type="file"
                                    name="logo"
                                    accept=".png,.jpg,.jpeg,.svg,.webp,.gif,.bmp,image/png,image/jpeg,image/svg+xml,image/webp,image/gif,image/bmp"
                                    onChange={() => clearFieldError(setProfileFormErrors, 'logo')}
                                  />
                                  {fieldError(profileFormErrors, 'logo') && (
                                    <div className="invalid-feedback">{fieldError(profileFormErrors, 'logo')}</div>
                                  )}
                                  <div className="form-text">Upload PNG, JPG, SVG, WEBP, GIF, or BMP for generated clearance letters.</div>
                                </div>
                                <div className="col-12">
                                  <label className="form-label">Signature</label>
                                  <input
                                    className={`form-control ${fieldError(profileFormErrors, 'signature') ? 'is-invalid' : ''}`}
                                    type="file"
                                    name="signature"
                                    accept="image/*"
                                    onChange={() => clearFieldError(setProfileFormErrors, 'signature')}
                                  />
                                  {fieldError(profileFormErrors, 'signature') && (
                                    <div className="invalid-feedback">{fieldError(profileFormErrors, 'signature')}</div>
                                  )}
                                  <div className="form-text">Director HR signature for clearance sign-off.</div>
                                </div>
                              </div>
                            </div>

                            <div className="dash-form-section">
                              <div className="dash-form-title">Attendance Rules</div>
                              <div className="row g-2">
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Late Time</label>
                                  <input
                                    className={`form-control ${fieldError(profileFormErrors, 'late_time') ? 'is-invalid' : ''}`}
                                    type="time"
                                    name="late_time"
                                    defaultValue={profile?.late_time || ''}
                                    onChange={() => clearFieldError(setProfileFormErrors, 'late_time')}
                                  />
                                  {fieldError(profileFormErrors, 'late_time') && (
                                    <div className="invalid-feedback">{fieldError(profileFormErrors, 'late_time')}</div>
                                  )}
                                  <div className="form-text">Use 24-hour format (HH:MM), e.g. 08:30.</div>
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Closing Time</label>
                                  <input
                                    className={`form-control ${fieldError(profileFormErrors, 'closing_time') ? 'is-invalid' : ''}`}
                                    type="time"
                                    name="closing_time"
                                    defaultValue={profile?.closing_time || ''}
                                    onChange={() => clearFieldError(setProfileFormErrors, 'closing_time')}
                                  />
                                  {fieldError(profileFormErrors, 'closing_time') && (
                                    <div className="invalid-feedback">{fieldError(profileFormErrors, 'closing_time')}</div>
                                  )}
                                  <div className="form-text">Use 24-hour format (HH:MM), e.g. 17:00.</div>
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Max Late Days</label>
                                  <input
                                    className={`form-control ${fieldError(profileFormErrors, 'max_days_late') ? 'is-invalid' : ''}`}
                                    type="number"
                                    min="0"
                                    step="1"
                                    name="max_days_late"
                                    defaultValue={profile?.max_days_late ?? ''}
                                    onChange={() => clearFieldError(setProfileFormErrors, 'max_days_late')}
                                  />
                                  {fieldError(profileFormErrors, 'max_days_late') && (
                                    <div className="invalid-feedback">{fieldError(profileFormErrors, 'max_days_late')}</div>
                                  )}
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Max Absent Days</label>
                                  <input
                                    className={`form-control ${fieldError(profileFormErrors, 'max_days_absent') ? 'is-invalid' : ''}`}
                                    type="number"
                                    min="0"
                                    step="1"
                                    name="max_days_absent"
                                    defaultValue={profile?.max_days_absent ?? ''}
                                    onChange={() => clearFieldError(setProfileFormErrors, 'max_days_absent')}
                                  />
                                  {fieldError(profileFormErrors, 'max_days_absent') && (
                                    <div className="invalid-feedback">{fieldError(profileFormErrors, 'max_days_absent')}</div>
                                  )}
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
	                          <p>Update the department name. Departments are organisation-wide.</p>
	                          <ul>
	                            <li>Deleting a department can affect corps members assigned to it.</li>
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
	                                  })
	                                  await refreshAll()
	                                  setEditDepartment(null)
	                                  setStatus('saved:department')
	                                } catch (err) {}
	                              })()
	                            }}
	                          >
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
                          <p>Update the unit name. Units are organisation-wide.</p>
                          <ul>
                            <li>Update the unit name.</li>
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
                                  })
                                  await refreshAll()
                                  setEditUnit(null)
                                  setStatus('saved:unit')
                                } catch (err) {}
                              })()
                            }}
                          >
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

          {activeTab==='subscription' && me?.role==='ORG' && (
            <>
              <h2 className="mb-3 text-olive">Subscription</h2>
              <SubscriptionSection />
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
                        {[20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </div>
                  </div>

                  <div className="table-responsive mt-2 dash-table-scroll">
                    <table className="table table-sm align-middle dash-table dash-table-auto dash-table-clearance mb-0">
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
                          <th className="dash-action-cell"></th>
                        </tr>
                      </thead>
                      <tbody>
                            {clearanceRows.map((row, idx) => (
                              <tr key={row.id}>
                                <td>{clearanceStart + idx + 1}</td>
                                <td><div className="text-truncate dash-td-truncate-wide">{row.full_name}</div></td>
                                <td>{row.state_code}</td>
                                <td><div className="text-truncate dash-td-truncate-wide">{row.branch || '—'}</div></td>
                                <td className="text-end">{row.absent}</td>
                                <td className="text-end">{row.late}</td>
                                <td>{row.qualified ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span>}</td>
                                <td>{row.downloaded ? <span className="badge bg-primary">Yes</span> : 'No'}</td>
                                <td className="text-end dash-action-cell">
                                  {!row.qualified && !row.override && !row.downloaded && (
                                    <button className="btn btn-sm btn-outline-secondary" onClick={async()=>{
                                      try{ await api.post('/api/auth/clearance/approve/', { corper: row.id }); await refreshAll() }catch(e){}
                                    }}>Approve</button>
                                  )}
                                </td>
                              </tr>
                            ))}
                            {clearanceFiltered.length===0 && (
                              <tr><td colSpan="9" className="text-muted">No corpers found.</td></tr>
                            )}
                      </tbody>
                    </table>
                  </div>
                  <div className="d-flex flex-wrap gap-2 justify-content-between align-items-center mt-2">
                    <div className="small text-muted">Page {clearanceCurrentPage} of {clearanceTotalPages} · {clearanceFiltered.length} result(s)</div>
                    <div className="btn-group">
                      <button className="btn btn-sm btn-outline-secondary" disabled={clearanceCurrentPage===1} onClick={()=>setClPage(1)}>First</button>
                      <button className="btn btn-sm btn-outline-secondary" disabled={clearanceCurrentPage===1} onClick={()=>setClPage(p=>Math.max(1,p-1))}>Prev</button>
                      <button className="btn btn-sm btn-outline-secondary" disabled={clearanceCurrentPage===clearanceTotalPages} onClick={()=>setClPage(p=>Math.min(clearanceTotalPages,p+1))}>Next</button>
                      <button className="btn btn-sm btn-outline-secondary" disabled={clearanceCurrentPage===clearanceTotalPages} onClick={()=>setClPage(clearanceTotalPages)}>Last</button>
                    </div>
                  </div>
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

                return (
                  <>
                    <div className="card shadow-sm dash-card">
                      <div className="card-body">
                        <div className="d-flex justify-content-between align-items-center gap-2">
                          <div className="dash-card-title mb-0">My Branch</div>
                          <div className="d-flex gap-2">
                            <button className="btn btn-sm btn-olive" type="button" onClick={() => {
                              setBranchLocationPos(myBranch.latitude && myBranch.longitude ? { lat: myBranch.latitude, lng: myBranch.longitude } : null)
                              setBranchLocationAddress(myBranch.address || '')
                              setBranchLocationAddressTouched(false)
                              setShowBranchLocation(true)
                            }}>Update location</button>
                          </div>
                        </div>

                        <div className="table-responsive mt-3">
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
                        </div>

                        <div className="small text-muted mt-2">
                          Departments and units are managed by the organisation.
                        </div>

                      </div>
                    </div>

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
                                <p>Admins can update the branch coordinates used for attendance verification.</p>
                                <ul>
                                  <li>Click on the map to set a pin.</li>
                                  <li>Use the 📍 control for current location.</li>
                                  <li>Save to apply immediately.</li>
                                </ul>
                              </div>
                              <div className="dash-modal-form">
                                <label className="form-label">Branch address</label>
                                <input
                                  className="form-control mb-2"
                                  value={branchLocationAddress}
                                  onChange={(e)=>{ setBranchLocationAddressTouched(true); setBranchLocationAddress(e.target.value) }}
                                  placeholder="e.g., 12 Marina, Lagos"
                                />
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
                                    await api.put(`/api/auth/branches/${myBranch.id}/`, { latitude: branchLocationPos.lat, longitude: branchLocationPos.lng, name: myBranch.name, address: String(branchLocationAddress || '').trim() })
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
	                  <div className="d-flex justify-content-between align-items-center gap-2 flex-wrap">
	                    <div className="dash-card-title mb-0">Registered Corpers</div>
	                    <div className="d-flex gap-2 flex-wrap justify-content-end">
	                      <button className="btn btn-sm btn-outline-secondary d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadImportTemplate('corpers')}>
	                        <Download size={15} /> Template
	                      </button>
	                      <button className="btn btn-sm btn-outline-secondary d-inline-flex align-items-center gap-1" type="button" onClick={()=>setShowCorperImport(true)}>
	                        <FileSpreadsheet size={15} /> Import
	                      </button>
	                      <button className="btn btn-sm btn-olive" type="button" onClick={() => setShowAddCorper(true)}>
	                        Add Corper
	                      </button>
	                    </div>
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

	              {showCorperImport && (
	                <div className="dash-modal" onClick={()=>setShowCorperImport(false)}>
	                  <div className="dash-modal-card" onClick={(e)=>e.stopPropagation()}>
	                    <div className="dash-modal-head">
	                      <strong>Import Corpers</strong>
	                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setShowCorperImport(false)}>Close</button>
	                    </div>
	                    <div className="dash-modal-body">
	                      <div className="dash-modal-grid">
	                        <div className="dash-modal-help">
	                          <h6>Bulk enrolment</h6>
	                          <p>Upload corper records after structure setup. The import creates accounts and sends verification emails; face capture still happens live for each corper.</p>
	                          <button className="btn btn-sm btn-outline-secondary d-inline-flex align-items-center gap-1" type="button" onClick={()=>downloadImportTemplate('corpers')}>
	                            <Download size={15} /> Download template
	                          </button>
	                        </div>
	                        <div className="dash-modal-form">
	                          <label className="form-label">CSV or Excel file</label>
	                          <input
	                            className="form-control"
	                            type="file"
	                            accept=".xlsx,.xlsm,.csv,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	                            onChange={(e)=>{ setCorperImportFile(e.target.files?.[0] || null); setCorperImportPreview(null) }}
	                          />
	                          <div className="d-flex justify-content-end gap-2 mt-3">
	                            <button className="btn btn-outline-secondary" type="button" onClick={()=>previewImport('corpers', corperImportFile)}>
	                              Preview
	                            </button>
	                            <button className="btn btn-olive" type="button" disabled={!corperImportPreview || corperImportPreview.errors_count > 0 || isSaving} onClick={()=>applyImport('corpers', corperImportFile)}>
	                              {isSaving ? 'Applying...' : 'Apply Import'}
	                            </button>
	                          </div>
	                          <ImportPreviewPanel preview={corperImportPreview} />
	                        </div>
	                      </div>
	                    </div>
	                  </div>
	                </div>
	              )}

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
                                <div className="col-12 col-md-4">
                                  <label className="form-label">CDS Day</label>
                                  <select className="form-select" name="cds_day" required>
                                    <option value="">Select CDS day</option>
                                    <option value="0">Monday</option>
                                    <option value="1">Tuesday</option>
                                    <option value="2">Wednesday</option>
                                    <option value="3">Thursday</option>
                                    <option value="4">Friday</option>
                                  </select>
                                  <div className="form-text">CDS day is excluded from required attendance days.</div>
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
                      <strong>Edit Corper</strong>
                      <button className="btn btn-sm btn-outline-secondary" type="button" onClick={()=>setEditCorper(null)}>Close</button>
                    </div>
                    <div className="dash-modal-body">
                      <div className="dash-modal-grid">
                        <div className="dash-modal-help">
                          <h6>Flow</h6>
                          <p>Update corper details and placement. If you change the email, an activation link will be sent so the corper can set a new password.</p>
                        </div>
                        <div className="dash-modal-form">
                          <form onSubmit={async (e)=>{
                            e.preventDefault();
                            setStatus('pending')
                            try{
                              const payload = {
                                email: editCorperForm.email,
                                full_name: editCorperForm.full_name,
                                state_code: String(editCorperForm.state_code || '').trim().toUpperCase(),
                                gender: editCorperForm.gender,
                                passing_out_date: editCorperForm.passing_out_date,
                                cds_day: editCorperForm.cds_day === '' ? null : Number(editCorperForm.cds_day),
                                department: editCorperForm.department || null,
                                unit: editCorperForm.unit || null,
                              }
                              if(me?.role === 'ORG') payload.branch = editCorperForm.branch || null
                              await api.patch(`/api/auth/corpers/${editCorper.id}/`, payload)
                              await refreshAll();
                              setStatus('saved:corper-update');
                              setEditCorper(null)
                            }catch(err){
                              const msg = err?.response?.data?.detail
                                || Object.values(err?.response?.data || {})?.[0]?.[0]
                                || err.message
                              setStatus(`error:corper-update:${msg}`)
                            }
                          }}>
                            <div className="dash-form-section">
                              <div className="dash-form-title">Details</div>
                              <div className="row g-2">
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Email</label>
                                  <input className="form-control" type="email" value={editCorperForm.email} onChange={(e)=>setEditCorperForm(p=>({ ...p, email: e.target.value }))} required />
                                </div>
                                <div className="col-12 col-md-6">
                                  <label className="form-label">Full Name</label>
                                  <input className="form-control" value={editCorperForm.full_name} onChange={(e)=>setEditCorperForm(p=>({ ...p, full_name: e.target.value }))} required />
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">Gender</label>
                                  <select className="form-select" value={editCorperForm.gender} onChange={(e)=>setEditCorperForm(p=>({ ...p, gender: e.target.value }))} required>
                                    <option value="">Select...</option>
                                    <option value="M">Male</option>
                                    <option value="F">Female</option>
                                    <option value="O">Other</option>
                                  </select>
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">State Code</label>
                                  <input className="form-control" value={editCorperForm.state_code} onChange={(e)=>setEditCorperForm(p=>({ ...p, state_code: e.target.value }))} required />
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">Passing Out Date</label>
                                  <input className="form-control" type="date" value={editCorperForm.passing_out_date} onChange={(e)=>setEditCorperForm(p=>({ ...p, passing_out_date: e.target.value }))} required />
                                </div>
                                <div className="col-12 col-md-4">
                                  <label className="form-label">CDS Day</label>
                                  <select className="form-select" value={editCorperForm.cds_day} onChange={(e)=>setEditCorperForm(p=>({ ...p, cds_day: e.target.value }))}>
                                    <option value="">—</option>
                                    <option value="0">Monday</option>
                                    <option value="1">Tuesday</option>
                                    <option value="2">Wednesday</option>
                                    <option value="3">Thursday</option>
                                    <option value="4">Friday</option>
                                  </select>
                                </div>
                              </div>
                            </div>

                            <div className="dash-form-section">
                              <div className="dash-form-title">Placement</div>
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
                                {deps.map(d=> <option key={d.id} value={d.id}>{d.name}</option>)}
                              </select>
                            </div>
                            <div className="mb-2">
                              <label className="form-label">Unit</label>
                              <select className="form-select" value={editCorperForm.unit} onChange={(e)=>setEditCorperForm(p=>({ ...p, unit: e.target.value }))}>
                                <option value="">—</option>
                                {units.map(u=> <option key={u.id} value={u.id}>{u.name}</option>)}
                              </select>
                            </div>
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
                              <div className="dash-kv"><div className="dash-k">CDS day</div><div className="dash-v">{selectedCorper.cds_day === 0 ? 'Monday' : selectedCorper.cds_day === 1 ? 'Tuesday' : selectedCorper.cds_day === 2 ? 'Wednesday' : selectedCorper.cds_day === 3 ? 'Thursday' : selectedCorper.cds_day === 4 ? 'Friday' : '—'}</div></div>
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
                                Edit
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
              {(() => {
                const last7 = stats?.attendance?.last7 || []
                const totalHours7 = last7.reduce((sum, r) => sum + Number(r.hours || 0), 0)
                const totalDays7 = last7.length
                const avgHours = totalDays7 ? (totalHours7 / totalDays7) : 0
                const checkins7 = last7.reduce((sum, r) => sum + Number(r.count || 0), 0)
                const monthCheckins = Number(stats?.attendance?.this_month || 0)
                const todayCheckins = Number(stats?.attendance?.today || 0)
                return (
                  <>
                    <div className="row g-3">
                      <div className="col-12 col-sm-6 col-lg-3">
                        <div className="dash-kpi">
                          <div className="dash-kpi-icon"><CalendarCheck2 size={18} aria-hidden /></div>
                          <div>
                            <div className="dash-kpi-label">Today</div>
                            <div className="dash-kpi-value">{todayCheckins}</div>
                          </div>
                        </div>
                      </div>
                      <div className="col-12 col-sm-6 col-lg-3">
                        <div className="dash-kpi">
                          <div className="dash-kpi-icon"><CalendarDays size={18} aria-hidden /></div>
                          <div>
                            <div className="dash-kpi-label">This Month</div>
                            <div className="dash-kpi-value">{monthCheckins}</div>
                          </div>
                        </div>
                      </div>
                      <div className="col-12 col-sm-6 col-lg-3">
                        <div className="dash-kpi">
                          <div className="dash-kpi-icon"><BarChart3 size={18} aria-hidden /></div>
                          <div>
                            <div className="dash-kpi-label">Hours (7 days)</div>
                            <div className="dash-kpi-value">{totalHours7.toFixed(1)}</div>
                          </div>
                        </div>
                      </div>
                      <div className="col-12 col-sm-6 col-lg-3">
                        <div className="dash-kpi">
                          <div className="dash-kpi-icon"><LayoutGrid size={18} aria-hidden /></div>
                          <div>
                            <div className="dash-kpi-label">Avg Hours/Day</div>
                            <div className="dash-kpi-value">{avgHours.toFixed(1)}</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="row g-3 mt-1">
                      <div className="col-12 col-lg-6">
                        <div className="card shadow-sm dash-card"><div className="card-body" style={{height:320}}>
                          <div className="dash-card-title">Check-ins by Day (Last 7 Days)</div>
                          <Bar data={{
                            labels: last7.map(r=> new Date(r.date).toLocaleDateString()),
                            datasets: [{ label: 'Check-ins', data: last7.map(r=> r.count ?? 0), backgroundColor: chartTheme.khaki, borderRadius: 10 }]
                          }} options={{
                            ...barOptions,
                            scales: {
                              x: { grid: { display:false }, ticks: { color: chartTheme.text }, title: { display:true, text:'Day' } },
                              y: { grid: { color: chartTheme.grid }, ticks: { color: chartTheme.text }, beginAtZero:true, title: { display:true, text:'Check-ins' } },
                            }
                          }} />
                        </div></div>
                      </div>
                      <div className="col-12 col-lg-6">
                        <div className="card shadow-sm dash-card"><div className="card-body" style={{height:320}}>
                          <div className="dash-card-title">Hours by Day (Last 7 Days)</div>
                          <Line
                            data={{
                              labels: last7.map(r=> new Date(r.date).toLocaleDateString()),
                              datasets: [{
                                label: 'Hours',
                                data: last7.map(r=> r.hours ?? 0),
                                borderColor: chartTheme.olive,
                                backgroundColor: chartTheme.oliveSoft,
                                tension: 0.35,
                                fill: true,
                                pointRadius: 3,
                              }]
                            }}
                            options={lineOptions}
                          />
                        </div></div>
                      </div>
                    </div>
                  </>
                )
              })()}
            </>
          )}

        {activeTab==='leave' && me?.role==='CORPER' && (
          <>
            <h2 className="mb-3 text-olive">Leave Management</h2>
            <div className="row g-3">
              <div className="col-12 col-xl-4">
                <div className="card shadow-sm dash-card h-100">
                  <div className="card-body">
                    <div className="dash-card-title mb-2">Apply for Leave</div>
                    <div className="text-muted small mb-3">Submit a leave request for approval.</div>
                    <form onSubmit={createLeave}>
                      <div className="row g-2">
                        <div className="col-12">
                          <label className="form-label">Start Date</label>
                          <input className="form-control" type="date" name="start_date" required/>
                        </div>
                        <div className="col-12">
                          <label className="form-label">End Date</label>
                          <input className="form-control" type="date" name="end_date" required/>
                        </div>
                        <div className="col-12">
                          <label className="form-label">Reason</label>
                          <textarea className="form-control" name="reason" rows="3" placeholder="Optional details" />
                        </div>
                      </div>
                      <div className="dash-modal-actions">
                        <div className="d-grid">
                          <button className="btn btn-olive" disabled={status==='pending'}>
                            {status==='pending' ? 'Submitting…' : 'Submit'}
                          </button>
                        </div>
                      </div>
                    </form>
                    {status==='saved:leave' && <AutoFadeAlert type="success" onClose={()=>setStatus(null)}>Leave request submitted.</AutoFadeAlert>}
                    {status==='error:leave' && <AutoFadeAlert type="danger" onClose={()=>setStatus(null)}>Could not submit leave request.</AutoFadeAlert>}
                  </div>
                </div>
              </div>

              <div className="col-12 col-xl-8">
                <div className="card shadow-sm dash-card">
                  <div className="card-body">
                    <div className="d-flex justify-content-between align-items-center gap-2">
                      <div className="dash-card-title mb-0">My Leave Requests</div>
                      <div className="d-flex align-items-center gap-2">
                        <button
                          className={`btn btn-sm ${myLeaveSearchOpen ? 'btn-olive' : 'btn-outline-secondary'}`}
                          type="button"
                          aria-label="Search"
                          onClick={() => setMyLeaveSearchOpen((v)=>!v)}
                        >
                          <Search size={16} />
                        </button>
                        {myLeaveSearchOpen && (
                          <div className="dash-table-search">
                            <input
                              className="form-control form-control-sm"
                              placeholder="Search…"
                              value={myLeaveQuery}
                              onChange={(e)=>{ setMyLeaveQuery(e.target.value); setMyLeavePage(1) }}
                            />
                          </div>
                        )}
                        <span className="small text-muted">Rows</span>
                        <select className="form-select form-select-sm" style={{width:96}} value={myLeavePageSize} onChange={(e)=>{ setMyLeavePageSize(Number(e.target.value)); setMyLeavePage(1) }}>
                          {[20,50,100].map(n => <option key={n} value={n}>{n}</option>)}
                        </select>
                      </div>
                    </div>

                    <div className="table-responsive mt-2">
                      <table className="table table-sm align-middle dash-table">
                        <thead>
                          <tr><th>#</th><th>Start</th><th>End</th><th>Status</th></tr>
                        </thead>
                        <tbody>
                          {(() => {
                            const q = myLeaveSearchOpen ? myLeaveQuery.trim().toLowerCase() : ''
                            let filtered = q ? leaves.filter(r => `${r.start_date} ${r.end_date} ${r.status}`.toLowerCase().includes(q)) : leaves
                            const totalPages = Math.max(1, Math.ceil(filtered.length / myLeavePageSize))
                            const current = Math.min(myLeavePage, totalPages)
                            if(current !== myLeavePage) setMyLeavePage(current)
                            const start = (current - 1) * myLeavePageSize
                            const rows = filtered.slice(start, start + myLeavePageSize)
                            return (
                              <>
                                {rows.map((r, idx) => (
                                  <tr key={r.id}>
                                    <td>{start + idx + 1}</td>
                                    <td>{r.start_date}</td>
                                    <td>{r.end_date}</td>
                                    <td>
                                      {r.status === 'APPROVED' ? <span className="badge bg-success">APPROVED</span> : r.status === 'REJECTED' ? <span className="badge bg-danger">REJECTED</span> : <span className="badge bg-warning text-dark">PENDING</span>}
                                    </td>
                                  </tr>
                                ))}
                                {filtered.length===0 && <tr><td colSpan="4" className="text-muted">No leave requests found.</td></tr>}
                                {filtered.length>0 && totalPages>1 && (
                                  <tr>
                                    <td colSpan="4">
                                      <div className="d-flex justify-content-between align-items-center">
                                        <div className="small text-muted">Page {current} of {totalPages} · {filtered.length} result(s)</div>
                                        <div className="btn-group">
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===1} onClick={()=>setMyLeavePage(p=>Math.max(1,p-1))}>Prev</button>
                                          <button className="btn btn-sm btn-outline-secondary" type="button" disabled={current===totalPages} onClick={()=>setMyLeavePage(p=>Math.min(totalPages,p+1))}>Next</button>
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
                  </div>
                </div>
              </div>
            </div>
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
            <QuerySection role="BRANCH" />
          )}

          {activeTab==='report' && me?.role==='BRANCH' && (
            <ReportSection role="BRANCH" />
          )}

          {activeTab==='report' && me?.role==='ORG' && (
            <ReportSection role="ORG" />
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
            © {new Date().getFullYear()} Sahab Technology Integrated Limited. All rights reserved.
          </div>
        </div>
      </div>
    </div>
  )
}
