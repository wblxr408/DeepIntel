import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Existing components use native fetch. Make every API request carry the
// HttpOnly administrator session without duplicating credentials options.
const nativeFetch = window.fetch.bind(window)
window.fetch = (input, init = {}) => nativeFetch(input, { ...init, credentials: init.credentials ?? 'include' })

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
