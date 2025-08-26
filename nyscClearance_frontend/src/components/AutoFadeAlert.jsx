import React, { useEffect, useState } from 'react'

export default function AutoFadeAlert({ type='info', onClose, children, timeout=3000 }){
  const [show, setShow] = useState(true)
  useEffect(() => {
    const t = setTimeout(() => setShow(false), timeout - 150)
    const t2 = setTimeout(() => onClose && onClose(), timeout)
    return () => { clearTimeout(t); clearTimeout(t2) }
  }, [timeout, onClose])
  return (
    <div className={`alert alert-${type} fade ${show?'show':''}`} role="alert">
      {children}
    </div>
  )
}

