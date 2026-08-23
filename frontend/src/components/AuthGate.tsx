import { FormEvent, useState } from 'react'

type AuthState = { initialized: boolean; authenticated: boolean }

async function readDetail(response: Response): Promise<string> {
  const body = await response.json().catch(() => ({})) as { detail?: string }
  return body.detail || '请求未成功，请检查服务状态。'
}

export default function AuthGate({ state, onAuthenticated }: { state: AuthState; onAuthenticated: () => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const initialize = !state.initialized

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError(null)
    if (initialize && password !== confirmPassword) {
      setError('两次输入的密码不一致。')
      return
    }
    setSubmitting(true)
    try {
      const response = await fetch(initialize ? '/api/v1/auth/initialize' : '/api/v1/auth/login', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!response.ok) throw new Error(await readDetail(response))
      onAuthenticated()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '认证失败。')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="min-h-screen grid place-items-center bg-xmgray-50 px-6">
      <section className="w-full max-w-sm border border-xmgray-200 bg-white p-8 shadow-xm-md" aria-labelledby="auth-title">
        <div className="mb-8 border-l-4 border-xm-500 pl-4">
          <p className="text-xs font-semibold uppercase text-xm-500">DeepIntel Console</p>
          <h1 id="auth-title" className="mt-1 text-2xl font-semibold text-xmgray-900">{initialize ? '初始化管理员' : '管理员登录'}</h1>
          <p className="mt-2 text-sm leading-6 text-xmgray-500">{initialize ? '首次部署仅可创建一个管理员。密码至少 12 位。' : '使用管理员凭据继续进入研究工作台。'}</p>
        </div>
        <form className="space-y-4" onSubmit={submit}>
          <label className="block text-sm text-xmgray-700">用户名
            <input className="input-xm mt-1 rounded-md py-3" autoComplete="username" value={username} onChange={event => setUsername(event.target.value)} required />
          </label>
          <label className="block text-sm text-xmgray-700">密码
            <input className="input-xm mt-1 rounded-md py-3" type="password" autoComplete={initialize ? 'new-password' : 'current-password'} value={password} onChange={event => setPassword(event.target.value)} required minLength={12} />
          </label>
          {initialize && <label className="block text-sm text-xmgray-700">确认密码
            <input className="input-xm mt-1 rounded-md py-3" type="password" autoComplete="new-password" value={confirmPassword} onChange={event => setConfirmPassword(event.target.value)} required minLength={12} />
          </label>}
          {error && <p className="border-l-2 border-red-600 bg-red-50 px-3 py-2 text-sm text-red-700" role="alert">{error}</p>}
          <button className="btn-primary w-full rounded-md" type="submit" disabled={submitting}>{submitting ? '正在验证…' : initialize ? '创建管理员并继续' : '登录'}</button>
        </form>
      </section>
    </main>
  )
}
