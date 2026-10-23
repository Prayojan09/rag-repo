import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import {
  Activity, ArrowLeft, ArrowRight, BookOpenText, CheckCircle2,
  ChevronDown, CircleHelp, Clock3, FileText, FileUp, Files, Filter, FolderOpen,
  Gauge, HardDrive, Layers3, LoaderCircle, LogOut, MessageSquareText, MoreHorizontal,
  Moon, Plus, RefreshCw, Send, Settings2, ShieldCheck, Sparkles, Sun,
  Trash2, TriangleAlert, UploadCloud, X, XCircle, ThumbsDown, ThumbsUp,
} from 'lucide-react'
import { ApiError, api, initialApiConfig, setCsrfToken, type ApiConfig, type AuthUser, type ChatResponse, type CitationDetail, type ConversationSummary, type DocumentDetail, type DocumentRecord } from './api'
import './structured.css'

type View = 'chat' | 'documents'
type Message = { id: string; role: 'user' | 'assistant'; text: string; answer?: ChatResponse; error?: boolean }
type SearchMode = 'hybrid' | 'dense' | 'keyword'
type Theme = 'light' | 'dark'
type AuthStatus = 'checking' | 'authenticated' | 'signed_out' | 'unavailable'
type RememberedAccount = Pick<AuthUser, 'email' | 'display_name'>

const dateLabel = (value: string) => new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value))
const sizeLabel = (bytes: number) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`
const maxUploadBytes = 100 * 1024 * 1024
const maxUploadMessage = 'The selected file exceeds the 100 MB upload limit. Please select a smaller file.'
const makeId = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
const rememberedAccountsKey = 'atlas-remembered-accounts'

function readRememberedAccounts(): RememberedAccount[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(rememberedAccountsKey) || '[]')
    if (!Array.isArray(stored)) return []
    return stored.filter((account): account is RememberedAccount =>
      typeof account?.email === 'string' && typeof account?.display_name === 'string',
    )
  } catch {
    return []
  }
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('atlas-theme')
    const initial = saved === 'dark' || saved === 'light'
      ? saved
      : window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    document.documentElement.dataset.theme = initial
    return initial
  })
  const [config] = useState<ApiConfig>(initialApiConfig)
  const [authStatus, setAuthStatus] = useState<AuthStatus>('checking')
  const [googleEnabled, setGoogleEnabled] = useState(false)
  const [googleLoginUrl, setGoogleLoginUrl] = useState<string | null>(null)
  const [authMessage, setAuthMessage] = useState('')
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null)
  const [rememberedAccounts, setRememberedAccounts] = useState<RememberedAccount[]>(readRememberedAccounts)
  const [signOutConfirm, setSignOutConfirm] = useState(false)
  const [signingOut, setSigningOut] = useState(false)
  const [view, setView] = useState<View>('chat')
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [totalDocuments, setTotalDocuments] = useState(0)
  const [apiStatus, setApiStatus] = useState<'checking' | 'ready' | 'error'>('checking')
  const [notice, setNotice] = useState('')
  const [accountOpen, setAccountOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('hybrid')
  const [rerank, setRerank] = useState(true)
  const [messages, setMessages] = useState<Message[]>([])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [conversationToDelete, setConversationToDelete] = useState<ConversationSummary | null>(null)
  const [deletingConversation, setDeletingConversation] = useState(false)
  const [chatBusy, setChatBusy] = useState(false)
  const [citation, setCitation] = useState<CitationDetail | null>(null)
  const [citationHighlightTerms, setCitationHighlightTerms] = useState<string[]>([])
  const [previewDocument, setPreviewDocument] = useState<DocumentDetail | null>(null)
  const [deleteCandidate, setDeleteCandidate] = useState<DocumentRecord | null>(null)
  const [deletingDocument, setDeletingDocument] = useState(false)
  const [citationBusy, setCitationBusy] = useState(false)
  const [offset, setOffset] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const chatEnd = useRef<HTMLDivElement>(null)
  const accountArea = useRef<HTMLDivElement>(null)

  const refresh = useCallback(async (current = config) => {
    if (authStatus !== 'authenticated') return false
    setApiStatus('checking')
    try {
      const [, page] = await Promise.all([api.ready(current), api.documents(current, offset)])
      setDocuments(page.items)
      setTotalDocuments(page.total)
      setApiStatus('ready')
      setNotice('')
      return true
    } catch (error) {
      setApiStatus('error')
      setNotice(error instanceof Error ? error.message : 'Could not connect to the backend.')
      return false
    }
  }, [authStatus, config, offset])

  useEffect(() => {
    let active = true
    async function initializeAuthentication() {
      try {
        const auth = await api.authConfig(config)
        if (!active) return
        setGoogleEnabled(auth.google_enabled)
        setGoogleLoginUrl(auth.google_login_url)
        if (!auth.authentication_required) {
          setAuthStatus('authenticated')
          return
        }
        if (!auth.google_enabled) {
          setAuthMessage('Google sign-in has not been configured by the workspace administrator.')
          setAuthStatus('unavailable')
          return
        }
        const user = await api.me(config)
        if (!active) return
        setCsrfToken(user.csrf_token)
        setCurrentUser(user)
        setAuthStatus('authenticated')
      } catch (error) {
        if (!active) return
        setAuthMessage(error instanceof ApiError && error.status === 401
          ? 'Your sign-in session has expired. Continue with Google to sign in again.'
          : 'Could not reach the authentication service. Check the backend and try again.')
        setAuthStatus(error instanceof ApiError && error.status === 401 ? 'signed_out' : 'unavailable')
      }
    }
    void initializeAuthentication()
    return () => { active = false }
  }, [config])
  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('atlas-theme', theme)
  }, [theme])
  useEffect(() => {
    if (!currentUser?.email) return
    const nextAccounts = [
      { email: currentUser.email, display_name: currentUser.display_name || currentUser.email },
      ...readRememberedAccounts().filter((account) => account.email !== currentUser.email),
    ]
    setRememberedAccounts(nextAccounts)
    try { localStorage.setItem(rememberedAccountsKey, JSON.stringify(nextAccounts)) } catch { /* Account switching remains available for this session. */ }
  }, [currentUser])
  useEffect(() => {
    if (!accountOpen) return
    const dismissOutside = (event: MouseEvent) => {
      if (event.target instanceof Node && !accountArea.current?.contains(event.target)) {
        setAccountOpen(false)
      }
    }
    document.addEventListener('mousedown', dismissOutside)
    return () => document.removeEventListener('mousedown', dismissOutside)
  }, [accountOpen])
  useEffect(() => {
    if (authStatus !== 'authenticated') return
    void api.conversations(config).then((result) => setConversations(result.items)).catch(() => setConversations([]))
  }, [authStatus, config, messages])
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, chatBusy])
  useEffect(() => {
    if (!notice) return
    const timer = window.setTimeout(() => setNotice(''), 6500)
    return () => window.clearTimeout(timer)
  }, [notice])

  const readyCount = useMemo(() => documents.filter((doc) => doc.status === 'ready').length, [documents])
  const hasProcessingDocuments = useMemo(
    () => documents.some((doc) => doc.status === 'queued' || doc.status === 'indexing' || doc.status === 'processing'),
    [documents],
  )

  useEffect(() => {
    if (authStatus !== 'authenticated') return
    if (!hasProcessingDocuments) return
    const timer = window.setInterval(() => {
      void api.documents(config, offset).then((page) => {
        setDocuments(page.items)
        setTotalDocuments(page.total)
      }).catch(() => undefined)
    }, 2500)
    return () => window.clearInterval(timer)
  }, [authStatus, config, hasProcessingDocuments, offset])

  async function signOut() {
    if (signingOut) return
    setSigningOut(true)
    try {
      await api.logout(config)
      setCsrfToken('')
      setCurrentUser(null)
      setAuthStatus('signed_out')
      setMessages([])
      setConversations([])
      setSignOutConfirm(false)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not sign out safely.')
    } finally {
      setSigningOut(false)
    }
  }

  function switchAccount(email?: string) {
    setAccountOpen(false)
    const startUrl = new URL(googleLoginUrl || api.googleLoginUrl(config), window.location.origin)
    if (email) startUrl.searchParams.set('login_hint', email)
    window.location.assign(startUrl.toString())
  }

  async function uploadFiles(files: FileList | File[]) {
    if (!files.length) return
    const selectedFiles = Array.from(files)
    const oversized = selectedFiles.filter((file) => file.size > maxUploadBytes)
    const uploadableFiles = selectedFiles.filter((file) => file.size <= maxUploadBytes)
    const uploadErrors = oversized.map((file) => `${file.name}: ${maxUploadMessage}`)
    if (!uploadableFiles.length) {
      setNotice(uploadErrors.join(' · '))
      return
    }
    setUploading(true)
    setView('documents')
    let uploadedCount = 0
    for (const file of uploadableFiles) {
      try {
        await api.upload(config, file)
        uploadedCount += 1
      } catch (error) {
        uploadErrors.push(`${file.name}: ${error instanceof Error ? error.message : 'Upload failed.'}`)
      }
    }
    setUploading(false)
    await refresh()
    if (uploadErrors.length) setNotice(uploadErrors.join(' · '))
    else if (uploadedCount) setNotice(`${uploadedCount} ${uploadedCount === 1 ? 'file' : 'files'} uploaded. Processing continues in the background.`)
  }

  function handleFiles(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) void uploadFiles(event.target.files)
    event.target.value = ''
  }

  async function confirmRemoveDocument() {
    if (!deleteCandidate || deletingDocument) return
    setDeletingDocument(true)
    try {
      await api.remove(config, deleteCandidate.id)
      setDeleteCandidate(null)
      await refresh()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not delete this document.')
    } finally {
      setDeletingDocument(false)
    }
  }

  async function runDocumentAction(document: DocumentRecord) {
    const isReprocessing = document.status === 'ready'
    if (isReprocessing) {
      setDocuments((items) => items.map((item) => item.id === document.id
        ? { ...item, status: 'processing', error_message: null }
        : item))
      setNotice(`Reprocessing ${document.filename}…`)
    }
    try {
      await api.retry(config, document.id)
      await refresh()
      setNotice(`${isReprocessing ? 'Reprocessing' : 'Retry'} completed for ${document.filename}.`)
    } catch (error) {
      await refresh()
      setNotice(error instanceof Error ? error.message : 'The document action failed.')
    }
  }

  async function sendQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const question = query.trim()
    if (!question || chatBusy) return
    setQuery('')
    setMessages((items) => [...items, { id: makeId(), role: 'user', text: question }])
    setChatBusy(true)
    try {
      let activeId = conversationId
      if (!activeId) {
        const conversation = await api.createConversation(config, question.slice(0, 80))
        activeId = conversation.id
        setConversationId(activeId)
      }
      const answer = await api.chat(config, { question, mode, top_k: 8, rerank, conversation_id: activeId })
      setMessages((items) => [...items, { id: makeId(), role: 'assistant', text: answer.answer, answer }])
    } catch (error) {
      setMessages((items) => [...items, { id: makeId(), role: 'assistant', text: error instanceof Error ? error.message : 'The request failed.', error: true }])
    } finally { setChatBusy(false) }
  }

  async function openConversation(id: string) {
    try {
      const history = await api.conversation(config, id)
      setConversationId(history.id)
      setMessages(history.messages.map((item) => ({
        id: item.id, role: item.role, text: item.content,
        answer: item.response_data ?? undefined,
      })))
      setView('chat')
    } catch (error) { setNotice(error instanceof Error ? error.message : 'Conversation could not be opened.') }
  }

  async function confirmDeleteConversation() {
    if (!conversationToDelete || deletingConversation) return
    setDeletingConversation(true)
    try {
      await api.removeConversation(config, conversationToDelete.id)
      setConversations((items) => items.filter((item) => item.id !== conversationToDelete.id))
      if (conversationId === conversationToDelete.id) {
        setConversationId(null)
        setMessages([])
      }
      setNotice('Chat history deleted.')
      setConversationToDelete(null)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not delete this chat history.')
    } finally {
      setDeletingConversation(false)
    }
  }

  function startConversation() {
    setConversationId(null)
    setMessages([])
    setView('chat')
  }

  async function showCitation(id: string, highlightTerms: string[] = []) {
    setPreviewDocument(null)
    setCitationBusy(true)
    setCitation(null)
    setCitationHighlightTerms(highlightTerms)
    try { setCitation(await api.citation(config, id)) }
    catch (error) { setNotice(error instanceof Error ? error.message : 'Citation could not be opened.') }
    finally { setCitationBusy(false) }
  }

  async function showDocument(document: DocumentRecord) {
    setCitation(null)
    setCitationBusy(true)
    try { setPreviewDocument(await api.document(config, document.id)) }
    catch (error) { setNotice(error instanceof Error ? error.message : 'Document preview could not be opened.') }
    finally { setCitationBusy(false) }
  }

  async function rateAnswer(answerId: string, rating: 'helpful' | 'not_helpful') {
    try { await api.feedback(config, answerId, rating); setNotice('Thanks — your feedback was saved.'); return true }
    catch (error) { setNotice(error instanceof Error ? error.message : 'Feedback could not be saved.'); return false }
  }

  if (authStatus === 'checking') return <main className="auth-screen"><section className="auth-card"><span className="brand-icon"><Layers3 size={19} /></span><h1>Preparing your workspace</h1><p>Checking your sign-in session…</p></section></main>
  const googleLoginFailed = new URLSearchParams(window.location.search).get('auth') === 'failed'
  if (authStatus === 'signed_out') return <main className="public-landing">
    <header className="landing-header"><a className="landing-brand" href="#home"><span className="brand-icon"><Layers3 size={19} strokeWidth={2.2} /></span><span>atlas<span className="brand-period">.</span></span></a><button className="button button-primary landing-signin" onClick={() => window.location.assign(googleLoginUrl || api.googleLoginUrl(config))} disabled={!googleEnabled}><GoogleMark />{googleEnabled ? 'Sign in' : 'Sign-in unavailable'}</button></header>
    <section className="landing-hero"><div className="landing-copy"><div className="eyebrow"><Sparkles size={14} /> DOCUMENT INTELLIGENCE</div><h1>Your documents,<br /><span>ready to answer.</span></h1><p>Find clear, evidence-backed answers across your documents. Atlas searches your private library and shows the source behind each response.</p><div className="landing-login-notice"><ShieldCheck size={17} /><span><strong>Sign in to chat with your documents or upload files.</strong><small>Use your Google account to access your private workspace.</small></span></div><button className="button button-primary landing-hero-cta" onClick={() => window.location.assign(googleLoginUrl || api.googleLoginUrl(config))} disabled={!googleEnabled}>{googleEnabled ? <><GoogleMark />Continue with Google<ArrowRight size={16} /></> : 'Sign-in is currently unavailable'}</button>{googleLoginFailed && <p className="landing-error" role="alert">Google sign-in did not complete. Please try again.</p>}{!googleEnabled && <p className="landing-error" role="status">Sign-in is not configured yet. Please contact your workspace administrator.</p>}</div><div className="landing-visual" aria-hidden="true"><div className="landing-window"><div className="landing-window-top"><span /><span /><span /><small>Atlas answer</small></div><div className="landing-window-body"><div className="landing-question"><MessageSquareText size={15} /><span>What does this policy say?</span></div><div className="landing-answer"><span className="landing-answer-mark"><Layers3 size={15} /></span><div><strong>Here’s what your documents say</strong><i /><i /><i className="short" /><div className="landing-source"><FileText size={13} /><span>Employee handbook · page 12</span><CheckCircle2 size={13} /></div></div></div></div></div></div></section>
    <section className="landing-features" aria-label="Atlas features"><article><span><MessageSquareText size={17} /></span><div><strong>Ask in plain language</strong><p>Ask questions using the context of your own documents.</p></div></article><article><span><FileText size={17} /></span><div><strong>See the evidence</strong><p>Answers include document and page references.</p></div></article><article><span><ShieldCheck size={17} /></span><div><strong>Private workspace</strong><p>Your library and saved conversations belong to your account.</p></div></article></section>
    <footer className="landing-footer">Atlas · Private document intelligence</footer>
  </main>
  if (authStatus === 'unavailable') return <main className="auth-screen"><section className="auth-card"><span className="brand-icon"><Layers3 size={19} /></span><div className="eyebrow">ATLAS DOCUMENT INTELLIGENCE</div><h1>Sign-in is unavailable</h1><p>{authMessage}</p><button className="button button-subtle" onClick={() => window.location.reload()}>Try again</button><small>Your documents and chat history are isolated to your signed-in account.</small></section></main>

  return (
    <div className="app-shell" onDragStart={(event) => event.preventDefault()}>
      <input ref={fileInput} className="sr-only" type="file" accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.tif,.tiff,.bmp,application/pdf,text/plain,image/*" multiple onChange={handleFiles} />
      <aside className="sidebar">
        <a className="brand" href="#home" onClick={(event) => { event.preventDefault(); setView('chat') }}>
          <span className="brand-icon"><Layers3 size={19} strokeWidth={2.2} /></span><span>atlas<span className="brand-period">.</span></span>
        </a>
        <div className="workspace-switch"><span className="workspace-avatar">W</span><span className="workspace-copy"><strong>Workspace</strong><small>Private knowledge base</small></span><ChevronDown size={15} /></div>
        <div className="nav-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main navigation">
          <button className={view === 'chat' ? 'nav-item active' : 'nav-item'} onClick={() => setView('chat')}><MessageSquareText size={17} />Ask Atlas<span className="nav-shortcut">⌘ K</span></button>
          <button className={view === 'documents' ? 'nav-item active' : 'nav-item'} onClick={() => setView('documents')}><Files size={17} />Documents<span className="nav-count">{totalDocuments}</span></button>
        </nav>
        <div className="history-heading"><span>RECENT CHATS</span><button aria-label="New conversation" title="New conversation" onClick={startConversation}><Plus size={14} /></button></div>
        <div className="conversation-history">{conversations.map((item) => <div key={item.id} className={conversationId === item.id ? 'history-row active' : 'history-row'}><button className="history-item" onClick={() => void openConversation(item.id)} title={item.title}><MessageSquareText size={14} /><span>{item.title}</span></button><button className="history-delete" onClick={() => setConversationToDelete(item)} title={`Delete ${item.title}`} aria-label={`Delete chat ${item.title}`}><Trash2 size={14} /></button></div>)}</div>
        <div className="sidebar-spacer" />
        <div className="storage-card">
          <div className="storage-heading"><HardDrive size={15} /><span>Knowledge base</span><span className="live-dot" /></div>
          <strong>{totalDocuments} <small>documents</small></strong>
          <div className="storage-track"><span style={{ width: totalDocuments ? `${Math.max(8, Math.min(100, readyCount * 4))}%` : '0%' }} /></div>
          <div className="storage-foot"><span>{readyCount} indexed</span><button onClick={() => void refresh()} title="Refresh library"><RefreshCw size={13} /></button></div>
        </div>
        <div className="profile-area" ref={accountArea}><button className="profile-button" aria-expanded={accountOpen} aria-haspopup="dialog" onClick={() => setAccountOpen((open) => !open)}><span className="profile-avatar">{currentUser?.display_name?.slice(0, 1).toUpperCase() || 'A'}</span><span className="profile-copy"><strong>{currentUser?.display_name || currentUser?.email || 'Account'}</strong><small>{currentUser?.email || 'Signed-in account'}</small></span><Settings2 size={16} /></button>{accountOpen && <AccountPopover user={currentUser} accounts={rememberedAccounts} onSwitchAccount={switchAccount} />}{currentUser && <button className="signout-button" onClick={() => { setAccountOpen(false); setSignOutConfirm(true) }}><LogOut size={14} />Sign out</button>}</div>
      </aside>

      <main className={view === 'chat' ? 'main-area chat-active' : 'main-area'}>
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>{view === 'chat' ? 'Ask Atlas' : 'Documents'}</strong></div>
          <div className="topbar-actions"><span className={`system-status ${apiStatus}`}><span />{apiStatus === 'ready' ? 'All systems operational' : apiStatus === 'checking' ? 'Checking services' : 'Connection issue'}</span><button className="theme-toggle" type="button" onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} aria-pressed={theme === 'dark'} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>{theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}<span>{theme === 'dark' ? 'Light' : 'Dark'}</span></button><button className="icon-button help-button" title="About Atlas" onClick={() => setNotice('Atlas searches your connected document library and cites source passages in its answers.')}><CircleHelp size={17} /></button><button className="top-avatar" title="Your account" aria-label="Open your account" onClick={() => setAccountOpen(true)}>{currentUser?.display_name?.slice(0, 1).toUpperCase() || 'A'}</button></div>
        </header>
        {notice && <div className="toast" role="status"><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Dismiss"><X size={15} /></button></div>}

        {view === 'chat' && <section className="chat-page">
          {messages.length === 0 ? <div className="welcome-block">
            <div className="eyebrow"><Sparkles size={14} /> YOUR KNOWLEDGE, IN CONTEXT</div>
            <h1>Good morning.<br /><span>What do you need to know?</span></h1>
            <p>Ask a question across your company documents. Atlas finds relevant passages and shows where every answer came from.</p>
            <div className="suggestion-grid">
              {['What is our parental leave policy?', 'How do I submit an expense report?', 'Summarize the latest security guidelines'].map((item, i) => <button key={item} className="suggestion" onClick={() => setQuery(item)}><span className="suggestion-icon">{i === 0 ? <BookOpenText size={17} /> : i === 1 ? <FileText size={17} /> : <ShieldCheck size={17} />}</span><span>{item}</span><ArrowRight size={15} /></button>)}
            </div>
            {totalDocuments === 0 && <button className="welcome-upload" onClick={() => fileInput.current?.click()}><FileUp size={16} /> Upload your first documents</button>}
          </div> : <div className="conversation">
            <div className="conversation-head"><div><div className="eyebrow"><Sparkles size={13} /> KNOWLEDGE ASSISTANT</div><h1>Ask Atlas</h1></div><button className="button button-subtle" onClick={startConversation}><Plus size={15} /> New conversation</button></div>
            <div className="message-list">{messages.map((message) => <div className={`message-row ${message.role}`} key={message.id}><div className={message.role === 'user' ? 'message-avatar user-avatar' : 'message-avatar assistant-avatar'}>{message.role === 'user' ? 'A' : <Layers3 size={16} />}</div><div className="message-content"><div className="message-author">{message.role === 'user' ? 'You' : 'Atlas'}<span>{message.role === 'user' ? 'Just now' : message.answer?.evidence_verified ? <><ShieldCheck size={12} /> Evidence checked</> : 'Response'}</span></div>{message.role === 'assistant' && !message.error ? <AnswerText text={message.text} /> : <div className={message.error ? 'message-text error-text' : 'message-text'}>{message.text}</div>}{message.answer && <AnswerEvidence answer={message.answer} onCitation={showCitation} onFeedback={rateAnswer} />}</div></div>)}{chatBusy && <div className="message-row assistant"><div className="message-avatar assistant-avatar"><Layers3 size={16} /></div><div className="message-content"><div className="message-author">Atlas <span>Searching your documents</span></div><div className="thinking"><span /><span /><span /></div></div></div>}<div ref={chatEnd} /></div>
          </div>}
          <div className="composer-wrap"><form className="composer" onSubmit={sendQuestion}><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder={chatBusy ? 'You can draft your next question while Atlas responds…' : 'Ask a question about your documents…'} rows={1} maxLength={4000} /><div className="composer-footer"><div className="composer-options"><label className="select-wrap"><span className="sr-only">Retrieval mode</span><Filter size={14} /><select value={mode} onChange={(event) => setMode(event.target.value as SearchMode)}><option value="hybrid">Hybrid search</option><option value="dense">Semantic</option><option value="keyword">Keyword</option></select><ChevronDown size={12} /></label><button type="button" className={rerank ? 'rerank-toggle selected' : 'rerank-toggle'} onClick={() => setRerank(!rerank)}><Gauge size={14} /> Rerank <span className="toggle-dot" /></button></div><button className="send-button" type="submit" disabled={!query.trim() || chatBusy} aria-label={chatBusy ? 'Wait for the current answer before sending' : 'Send question'} title={chatBusy ? 'You can draft a question while Atlas finishes this answer' : 'Send question'}><Send size={16} /></button></div></form><div className="composer-caption"><ShieldCheck size={12} />{chatBusy ? 'The chat stays available while Atlas answers. You can draft a follow-up now.' : 'Answers include source citations and may abstain when evidence is insufficient.'}</div></div>
        </section>}

        {view === 'documents' && <section className="page-content documents-page">
          <div className="page-heading"><div><div className="eyebrow">YOUR SOURCE LIBRARY</div><h1>Documents <span className="heading-count">{totalDocuments}</span></h1><p>Manage the files Atlas uses to answer questions.</p></div><button className="button button-primary" onClick={() => fileInput.current?.click()} disabled={uploading}>{uploading ? <LoaderCircle className="spin" size={16} /> : <Plus size={17} />}{uploading ? 'Uploading…' : 'Add documents'}</button></div>
          <div className="library-stats"><div className="stat-card"><div className="stat-icon blue"><Files size={17} /></div><div><strong>{totalDocuments}</strong><span>Total documents</span></div></div><div className="stat-card"><div className="stat-icon green"><CheckCircle2 size={17} /></div><div><strong>{readyCount}</strong><span>Ready to search</span></div></div><div className="stat-card"><div className="stat-icon amber"><Clock3 size={17} /></div><div><strong>{documents.filter((doc) => ['queued', 'indexing'].includes(doc.status)).length}</strong><span>Processing</span></div></div></div>
          {documents.length === 0 ? <div className="empty-state"><div className="empty-icon"><FolderOpen size={25} /></div><h2>Your library is ready</h2><p>Upload policies, handbooks, and other company files to make them searchable.</p><button className="button button-primary" onClick={() => fileInput.current?.click()}><FileUp size={16} /> Browse files</button><span>PDF, DOCX, TXT, or images · max 100 MB per file</span></div> : <div className="table-card"><div className="table-toolbar"><div><strong>All documents</strong><span>{totalDocuments} items in your workspace</span></div><button className="icon-button" onClick={() => void refresh()} title="Refresh"><RefreshCw size={15} /></button></div><div className="document-table-wrap"><table className="document-table"><thead><tr><th>NAME</th><th>STATUS</th><th>SIZE</th><th>ADDED</th><th><span className="sr-only">Actions</span></th></tr></thead><tbody>{documents.map((doc) => <DocumentRow key={doc.id} document={doc} onDelete={setDeleteCandidate} onAction={runDocumentAction} onOpen={() => void showDocument(doc)} />)}</tbody></table></div><div className="table-pagination"><span>Showing {offset + 1}–{Math.min(offset + documents.length, totalDocuments)} of {totalDocuments}</span><div><button className="icon-button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}><ArrowLeft size={15} /></button><button className="icon-button" disabled={offset + 50 >= totalDocuments} onClick={() => setOffset(offset + 50)}><ArrowRight size={15} /></button></div></div></div>}
          <div className="upload-hint"><UploadCloud size={16} /><span><strong>Quick upload</strong> Choose PDF, DOCX, TXT, or image files. Max 100 MB per file.</span><button onClick={() => fileInput.current?.click()}>Browse files <ArrowRight size={14} /></button></div>
        </section>}

      </main>

      {signOutConfirm && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !signingOut) setSignOutConfirm(false) }}><section className="settings-modal delete-confirm" role="alertdialog" aria-modal="true" aria-labelledby="signout-title" aria-describedby="signout-description"><div className="modal-header"><div className="settings-logo danger-logo"><LogOut size={18} /></div><button className="icon-button" onClick={() => setSignOutConfirm(false)} aria-label="Cancel sign out" disabled={signingOut}><X size={17} /></button></div><div className="modal-copy"><div className="eyebrow">ACCOUNT</div><h2 id="signout-title">Sign out of Atlas?</h2><p id="signout-description">You can sign in again at any time. Your documents and saved chat history will remain in your workspace.</p></div><div className="modal-footer"><span /><div><button className="button button-subtle" onClick={() => setSignOutConfirm(false)} disabled={signingOut}>Cancel</button><button className="button button-danger" onClick={() => void signOut()} disabled={signingOut}>{signingOut ? <LoaderCircle className="spin" size={15} /> : <LogOut size={15} />}{signingOut ? 'Signing out…' : 'Sign out'}</button></div></div></section></div>}
      {deleteCandidate && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !deletingDocument) setDeleteCandidate(null) }}><section className="settings-modal delete-confirm" role="alertdialog" aria-modal="true" aria-labelledby="delete-title" aria-describedby="delete-description"><div className="modal-header"><div className="settings-logo danger-logo"><Trash2 size={18} /></div><button className="icon-button" onClick={() => setDeleteCandidate(null)} aria-label="Cancel deletion" disabled={deletingDocument}><X size={17} /></button></div><div className="modal-copy"><div className="eyebrow">REMOVE DOCUMENT</div><h2 id="delete-title">Delete this document?</h2><p id="delete-description">“{deleteCandidate.filename}” and its indexed chunks will be removed from your workspace.</p></div><div className="modal-footer"><span /><div><button className="button button-subtle" onClick={() => setDeleteCandidate(null)} disabled={deletingDocument}>Cancel</button><button className="button button-danger" onClick={() => void confirmRemoveDocument()} disabled={deletingDocument}>{deletingDocument ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />}{deletingDocument ? 'Deleting…' : 'Delete document'}</button></div></div></section></div>}
      {conversationToDelete && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !deletingConversation) setConversationToDelete(null) }}><section className="settings-modal delete-confirm" role="alertdialog" aria-modal="true" aria-labelledby="delete-chat-title" aria-describedby="delete-chat-description"><div className="modal-header"><div className="settings-logo danger-logo"><Trash2 size={18} /></div><button className="icon-button" onClick={() => setConversationToDelete(null)} aria-label="Cancel deletion" disabled={deletingConversation}><X size={17} /></button></div><div className="modal-copy"><div className="eyebrow">REMOVE CHAT HISTORY</div><h2 id="delete-chat-title">Delete this chat?</h2><p id="delete-chat-description">“{conversationToDelete.title}” and its messages will be permanently deleted.</p></div><div className="modal-footer"><span /><div><button className="button button-subtle" onClick={() => setConversationToDelete(null)} disabled={deletingConversation}>Cancel</button><button className="button button-danger" onClick={() => void confirmDeleteConversation()} disabled={deletingConversation}>{deletingConversation ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />}{deletingConversation ? 'Deleting…' : 'Delete chat'}</button></div></div></section></div>}
      {(citation || (citationBusy && !previewDocument)) && <CitationPanel citation={citation} highlightTerms={citationHighlightTerms} busy={citationBusy} onClose={() => { setCitation(null); setCitationBusy(false) }} />}
      {previewDocument && <DocumentPreviewPanel document={previewDocument} onClose={() => setPreviewDocument(null)} />}
    </div>
  )
}

function DocumentRow({ document, onDelete, onAction, onOpen }: { document: DocumentRecord; onDelete: (d: DocumentRecord) => void; onAction: (d: DocumentRecord) => void; onOpen: () => void }) {
  const [menuPosition, setMenuPosition] = useState<{ top: number; right: number } | null>(null)
  const toggleMenu = (event: React.MouseEvent<HTMLButtonElement>) => {
    if (menuPosition) {
      setMenuPosition(null)
      return
    }
    const rect = event.currentTarget.getBoundingClientRect()
    const menuHeight = 82
    const top = rect.bottom + menuHeight <= window.innerHeight - 8
      ? rect.bottom + 4
      : Math.max(8, rect.top - menuHeight - 4)
    setMenuPosition({ top, right: Math.max(8, window.innerWidth - rect.right) })
  }
  const closeMenu = () => setMenuPosition(null)
  return <tr><td><button className="file-cell" onClick={onOpen}><span className={`file-icon ${document.filename.toLowerCase().endsWith('.pdf') ? 'pdf' : document.filename.toLowerCase().endsWith('.docx') ? 'docx' : 'txt'}`}><FileText size={17} /></span><span className="file-details"><strong>{document.filename}</strong><small>Version {document.version} · {document.media_type || 'Document'}</small></span></button></td><td><StatusBadge status={document.status} />{document.status === 'failed' && document.error_message && <small className="status-error">{document.error_message}</small>}</td><td>{sizeLabel(document.size_bytes)}</td><td>{dateLabel(document.created_at)}</td><td className="row-actions"><button className="icon-button" aria-label={`Actions for ${document.filename}`} aria-expanded={menuPosition !== null} onClick={toggleMenu}><MoreHorizontal size={17} /></button>{menuPosition && <div className="action-menu" style={menuPosition}>{['failed', 'queued', 'indexing'].includes(document.status) && <button onClick={() => { onAction(document); closeMenu() }}><RefreshCw size={14} /> Retry ingestion</button>}{document.status === 'ready' && <button onClick={() => { onAction(document); closeMenu() }}><RefreshCw size={14} /> Reprocess document</button>}<button className="danger-menu" onClick={() => { onDelete(document); closeMenu() }}><Trash2 size={14} /> Delete document</button></div>}</td></tr>
}

function StatusBadge({ status }: { status: string }) {
  const busy = status === 'indexing' || status === 'queued' || status === 'processing'
  const Icon = status === 'ready' ? CheckCircle2 : status === 'failed' ? XCircle : busy ? LoaderCircle : Activity
  const badgeClass = status === 'processing' ? 'indexing' : status
  const label = status === 'ready' ? 'Indexed' : status === 'failed' ? 'Failed' : status === 'indexing' ? 'Indexing' : status === 'queued' ? 'Queued' : status === 'processing' ? 'Reprocessing…' : status
  return <span className={`status-badge ${badgeClass}`}><Icon size={13} className={busy ? 'spin' : ''} />{label}</span>
}

function GoogleMark() {
  return <svg aria-hidden="true" viewBox="0 0 48 48" width="18" height="18"><path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.5l6.7-6.7C35.6 2.4 30.3 0 24 0 14.6 0 6.5 5.4 2.6 13.3l7.8 6.1C12.3 13.6 17.7 9.5 24 9.5Z"/><path fill="#4285F4" d="M46.1 24.5c0-1.6-.2-3.1-.5-4.5H24v9h12.4c-.5 2.9-2.1 5.4-4.6 7.1l7.4 5.7c4.3-4 6.9-9.9 6.9-17.3Z"/><path fill="#FBBC05" d="M10.4 28.6c-.5-1.4-.8-3-.8-4.6s.3-3.2.8-4.6l-7.8-6.1C.9 16.8 0 20.3 0 24s.9 7.2 2.6 10.7l7.8-6.1Z"/><path fill="#34A853" d="M24 48c6.3 0 11.6-2.1 15.4-5.9L32 36.4c-2.1 1.4-4.8 2.2-8 2.2-6.3 0-11.7-4.1-13.6-9.8l-7.8 6.1C6.5 42.6 14.6 48 24 48Z"/></svg>
}

function AnswerEvidence({ answer, onCitation, onFeedback }: { answer: ChatResponse; onCitation: (id: string, highlightTerms: string[]) => void; onFeedback: (id: string, rating: 'helpful' | 'not_helpful') => Promise<boolean> }) {
  const [savedRating, setSavedRating] = useState<'helpful' | 'not_helpful' | null>(null)
  const [savingRating, setSavingRating] = useState(false)
  async function submitRating(rating: 'helpful' | 'not_helpful') {
    if (savingRating) return
    setSavingRating(true)
    if (await onFeedback(answer.answer_id, rating)) setSavedRating(rating)
    setSavingRating(false)
  }
  const citationById = new Map(answer.citations.map((item) => [item.chunk_id, item]))
  const citationIds = [...new Set(answer.claims.flatMap((claim) => claim.source_chunk_ids))]
  return <div className="answer-evidence">{answer.abstained && <div className="abstain-note"><TriangleAlert size={15} />{answer.abstention_reason || 'There is not enough evidence in your documents to answer confidently.'}</div>}{citationIds.length > 0 && <div className="citation-list"><div className="citation-heading"><span>SOURCES</span><span>{answer.evidence_verified ? 'Evidence checked' : 'Sources'}</span></div>{citationIds.map((id, index) => { const item = citationById.get(id); return <button className="citation-chip" key={id} onClick={() => onCitation(id, getHighlightTerms(answer.question))}><span className="citation-index">{index + 1}</span><span className="citation-info"><strong>{item?.filename ?? 'Source document'}</strong><small>{item?.page_number ? `Page ${item.page_number}` : ''}{item?.heading ? ` · ${item.heading}` : ''}</small></span><ArrowRight size={14} /></button> })}</div>}<div className="answer-meta"><span>{answer.llm_model ?? answer.llm_provider} · {answer.retrieval_mode} search</span><div><span>{savedRating ? 'Thanks for reviewing' : savingRating ? 'Saving?' : 'Was this helpful?'}</span><button type="button" className={savedRating === 'helpful' ? 'review-selected' : ''} title="Helpful" aria-label="Mark answer helpful" aria-pressed={savedRating === 'helpful'} disabled={savingRating} onClick={() => void submitRating('helpful')}><ThumbsUp size={14} /></button><button type="button" className={savedRating === 'not_helpful' ? 'review-selected' : ''} title="Not helpful" aria-label="Mark answer not helpful" aria-pressed={savedRating === 'not_helpful'} disabled={savingRating} onClick={() => void submitRating('not_helpful')}><ThumbsDown size={14} /></button></div></div></div>
}

function AnswerText({ text }: { text: string }) {
  const summary = text.match(/^## Summary\s*\n\s*([\s\S]*)$/)
  if (summary) {
    const points = summary[1].split('\n').map((line) => line.replace(/^\s*[•*-]\s*/, '').trim()).filter(Boolean)
    return <div className="message-text answer-text summary-answer"><h3>Summary</h3><ul>{points.map((point, index) => <li key={`${index}-${point}`}>{point}</li>)}</ul></div>
  }
  const paragraphs = text.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean)
  return <div className="message-text answer-text">{paragraphs.map((paragraph, index) => <p key={`${index}-${paragraph}`}>{paragraph}</p>)}</div>
}

function CitationPanel({ citation, highlightTerms, busy, onClose }: { citation: CitationDetail | null; highlightTerms: string[]; busy: boolean; onClose: () => void }) {
  return <><button className="drawer-backdrop" onClick={onClose} aria-label="Close source preview" /><aside className="citation-drawer citation-reading-view"><button className="icon-button citation-close" onClick={onClose} aria-label="Close"><X size={18} /></button>{busy || !citation ? <div className="drawer-loading"><LoaderCircle className="spin" size={20} />Loading source…</div> : <article className="citation-reading">{citation.page_number && <div className="citation-page-number">Page {citation.page_number}</div>}<h2>{citation.heading || 'Document content'}</h2><div className="citation-content"><HighlightedPassage text={contentWithoutHeading(citation.text, citation.heading)} terms={highlightTerms} /></div></article>}</aside></>
}

const highlightStopWords = new Set([
  'about', 'after', 'again', 'also', 'among', 'are', 'been', 'being', 'between',
  'could', 'did', 'does', 'doing', 'down', 'during', 'each', 'from', 'have',
  'having', 'here', 'into', 'more', 'most', 'other', 'over', 'please', 'same',
  'should', 'some', 'such', 'than', 'that', 'their', 'there', 'these', 'they',
  'this', 'those', 'through', 'under', 'very', 'what', 'when', 'where', 'which',
  'while', 'with', 'would', 'your', 'chapter', 'document', 'summarize', 'summary',
  'tell', 'explain', 'describe', 'give', 'make', 'main', 'point', 'points',
])

function getHighlightTerms(question: string): string[] {
  const words = question.toLocaleLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []
  return [...new Set(words.filter((word) => !highlightStopWords.has(word)))].slice(0, 12)
}

type PassageBlock = { kind: 'paragraph' | 'unordered' | 'ordered'; text: string }

function splitPassageBlocks(text: string): PassageBlock[] {
  const blocks: PassageBlock[] = []
  let paragraph: string[] = []
  let listKind: 'unordered' | 'ordered' | null = null
  let listItems: string[] = []
  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ kind: 'paragraph', text: paragraph.join(' ').trim() })
    paragraph = []
  }
  const flushList = () => {
    if (listItems.length && listKind) blocks.push({ kind: listKind, text: listItems.join('\n') })
    listItems = []
    listKind = null
  }
  const normalizedText = text
    .replace(/\r/g, '')
    .replace(/[\uF0B7\uF0A7\u2022\u25AA\u25E6\u25CF]\s*/gu, '\n\u2022 ')
  for (const rawLine of normalizedText.split('\n')) {
    const line = rawLine.trim()
    if (!line) {
      flushParagraph()
      flushList()
      continue
    }
    const match = line.match(/^(?:([-*\u2022\u25AA\u25E6])\s+|(\d+)[.)]\s+)/u)
    if (match) {
      flushParagraph()
      const nextKind = match[2] ? 'ordered' : 'unordered'
      if (listKind && listKind !== nextKind) flushList()
      listKind = nextKind
      listItems.push(line.replace(/^(?:[-*\u2022\u25AA\u25E6]\s+|\d+[.)]\s+)/u, ''))
    } else {
      flushList()
      paragraph.push(line)
    }
  }
  flushParagraph()
  flushList()
  return blocks.filter((block) => block.text.trim())
}

function matchingTermCount(text: string, terms: string[]): number {
  const words = new Set(text.toLocaleLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? [])
  return terms.filter((term) => words.has(term)).length
}

function matchesQuestion(text: string, terms: string[], wholeBlock = false): boolean {
  if (!terms.length) return false
  const overlap = matchingTermCount(text, terms)
  const threshold = wholeBlock
    ? Math.min(terms.length, Math.max(2, Math.ceil(terms.length * 0.45)))
    : Math.min(terms.length, terms.length <= 2 ? 1 : Math.max(2, Math.ceil(terms.length * 0.3)))
  return overlap >= Math.min(threshold, terms.length)
}

function HighlightedSentenceText({ text, terms }: { text: string; terms: string[] }) {
  const sentences = text.split(/(?<=[.!?])\s+/u)
  return <>{sentences.map((sentence, index) => matchesQuestion(sentence, terms) ? <mark className="evidence-match evidence-sentence" key={`${index}-${sentence}`}>{sentence}{index < sentences.length - 1 ? ' ' : ''}</mark> : <span key={`${index}-${sentence}`}>{sentence}{index < sentences.length - 1 ? ' ' : ''}</span>)}</>
}

function HighlightedPassage({ text, terms }: { text: string; terms: string[] }) {
  const blocks = splitPassageBlocks(text)
  return <div className="passage-blocks">{blocks.map((block, index) => {
    if (block.kind === 'paragraph') {
      const fullMatch = matchesQuestion(block.text, terms, true)
      return <p className={fullMatch ? 'passage-paragraph paragraph-match' : 'passage-paragraph'} key={`${index}-${block.text}`}>
        {fullMatch ? <mark className="evidence-match evidence-paragraph-match">{block.text}</mark> : <HighlightedSentenceText text={block.text} terms={terms} />}
      </p>
    }
    const items = block.text.split('\n')
    const List = block.kind === 'ordered' ? 'ol' : 'ul'
    return <List className="passage-list" key={`${index}-${block.text}`}>{items.map((item, itemIndex) => matchesQuestion(item, terms) ? <li className="list-match" key={`${itemIndex}-${item}`}><mark className="evidence-match evidence-list-match">{item}</mark></li> : <li key={`${itemIndex}-${item}`}>{item}</li>)}</List>
  })}</div>
}
function contentWithoutHeading(text: string, heading: string | null): string {
  if (!heading) return text
  const lines = text.replace(/\r/g, '').split('\n')
  if (lines[0]?.trim().toLocaleLowerCase() !== heading.trim().toLocaleLowerCase()) return text
  return lines.slice(1).join('\n').trimStart()
}

type ReadableChunk = DocumentDetail['chunks'][number] & { displayText: string; displayHeading: string | null }
type ReadablePage = { pageNumber: number | null; chunks: ReadableChunk[] }

function normalizedTextPositions(text: string): { value: string; positions: number[] } {
  let value = ''
  const positions: number[] = []
  let inWhitespace = false
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]
    if (/\s/u.test(character)) {
      if (value && !inWhitespace) {
        value += ' '
        positions.push(index)
      }
      inWhitespace = true
    } else {
      value += character.toLocaleLowerCase()
      positions.push(index)
      inWhitespace = false
    }
  }
  return { value: value.trimEnd(), positions: positions.slice(0, value.trimEnd().length) }
}

function removeVisibleChunkOverlap(previous: string, current: string): string {
  if (!previous || !current) return current
  const left = normalizedTextPositions(previous)
  const right = normalizedTextPositions(current)
  const maxOverlap = Math.min(600, left.value.length, right.value.length)
  for (let length = maxOverlap; length >= 40; length -= 1) {
    const start = left.value.length - length
    const end = length
    const leftBefore = left.value[start - 1]
    const rightFirst = right.value[0]
    const leftLast = left.value[left.value.length - 1]
    const rightAfter = right.value[end]
    const isWord = (value: string | undefined) => Boolean(value && /[\p{L}\p{N}]/u.test(value))
    if (isWord(leftBefore) && isWord(rightFirst)) continue
    if (isWord(leftLast) && isWord(rightAfter)) continue
    if (left.value.endsWith(right.value.slice(0, length))) {
      const originalEnd = right.positions[length - 1] + 1
      return current.slice(originalEnd).trimStart()
    }
  }
  return current
}

function buildReadableChunks(chunks: DocumentDetail['chunks']): ReadableChunk[] {
  let previousText = ''
  let previousHeading = ''
  let previousPage: number | null = null
  return chunks.map((chunk) => {
    const content = contentWithoutHeading(chunk.text, chunk.heading)
    const displayText = chunk.page_number === previousPage ? removeVisibleChunkOverlap(previousText, content) : content
    const normalizedHeading = chunk.heading?.trim().toLocaleLowerCase() ?? ''
    const displayHeading = normalizedHeading && (normalizedHeading !== previousHeading || chunk.page_number !== previousPage) ? chunk.heading : null
    previousText = content
    previousHeading = normalizedHeading
    previousPage = chunk.page_number
    return { ...chunk, displayText, displayHeading }
  })
}

function groupReadableChunks(chunks: ReadableChunk[]): ReadablePage[] {
  const pages: ReadablePage[] = []
  for (const chunk of chunks) {
    const current = pages[pages.length - 1]
    if (!current || current.pageNumber !== chunk.page_number) {
      pages.push({ pageNumber: chunk.page_number, chunks: [chunk] })
    } else {
      current.chunks.push(chunk)
    }
  }
  return pages
}

function DocumentPreviewPanel({ document, onClose }: { document: DocumentDetail; onClose: () => void }) {
  const readablePages = groupReadableChunks(buildReadableChunks(document.chunks))
  const [showSource, setShowSource] = useState(!document.structured_content)
  useEffect(() => setShowSource(!document.structured_content), [document.id, document.structured_content])
  const hasContent = showSource ? document.chunks.length > 0 : Boolean(document.structured_content)
  return <><button className="drawer-backdrop" onClick={onClose} aria-label="Close document preview" /><aside className="citation-drawer"><div className="drawer-top"><div><div className="eyebrow">DOCUMENT PREVIEW</div><h2>Extracted content</h2></div><button className="icon-button" onClick={onClose} aria-label="Close"><X size={17} /></button></div><div className="drawer-file"><span className="file-icon docx"><FileText size={18} /></span><div><strong>{document.filename}</strong><small>Version {document.version} {"\u00b7"} {document.status}</small></div></div>{document.status === 'failed' && document.error_message && <div className="doc-error-note"><TriangleAlert size={15} />{document.error_message}</div>}{document.structured_content && <div className="structured-preview-toolbar"><span>{showSource ? 'Original extracted text' : 'AI-structured preview; original text remains the search and citation source.'}</span><button className="button button-subtle" onClick={() => setShowSource(!showSource)}>{showSource ? 'Show structured view' : 'Show source extraction'}</button></div>}{hasContent ? showSource ? <div className="preview-pages">{readablePages.map((page, pageIndex) => <section className="preview-page" key={`${page.pageNumber ?? 'unpaginated'}-${pageIndex}`}>{page.pageNumber !== null && <div className="preview-page-number">Page {page.pageNumber}</div>}{page.chunks.map((chunk) => <article className="preview-section" key={chunk.id}>{chunk.displayHeading ? <h3>{chunk.displayHeading}</h3> : !chunk.heading && <h3>Document content</h3>}<div className="preview-text"><HighlightedPassage text={chunk.displayText} terms={[]} /></div></article>)}</section>)}</div> : <StructuredDocumentContent text={document.structured_content ?? ''} /> : <div className="empty-preview"><FolderOpen size={22} /><p>{document.status === 'failed' ? document.error_message || 'Ingestion failed. Retry processing from the document actions.' : 'No extracted content is available yet.'}</p></div>}</aside></>
}

type StructuredBlock =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'list'; ordered: boolean; start: number; items: { text: string; level: number }[] }
  | { kind: 'table'; rows: string[][] }

function parseStructuredBlocks(text: string): StructuredBlock[] {
  const lines = text.replace(/\r/g, '').split('\n')
  const blocks: StructuredBlock[] = []
  let paragraph: string[] = []
  let listItems: { text: string; level: number }[] = []
  let listOrdered: boolean | null = null
  let listStart = 1
  let tableRows: string[][] = []
  let codeLines: string[] = []
  let inCode = false
  const flushParagraph = () => { if (paragraph.length) blocks.push({ kind: 'paragraph', text: paragraph.join(' ') }); paragraph = [] }
  const flushList = () => { if (listItems.length) blocks.push({ kind: 'list', ordered: Boolean(listOrdered), start: listStart, items: listItems }); listItems = []; listOrdered = null; listStart = 1 }
  const flushTable = () => { if (tableRows.length) blocks.push({ kind: 'table', rows: tableRows }); tableRows = [] }
  const isTableSeparator = (line: string) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/u.test(line)
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const raw = lines[lineIndex]
    const line = raw.trim()
    if (line.startsWith('```')) {
      flushParagraph(); flushList(); flushTable()
      if (inCode) { blocks.push({ kind: 'code', text: codeLines.join('\n') }); codeLines = [] }
      inCode = !inCode
      continue
    }
    if (inCode) { codeLines.push(raw); continue }
    const heading = line.match(/^(#{1,6})\s+(.+)$/u)
    if (heading) { flushParagraph(); flushList(); flushTable(); blocks.push({ kind: 'heading', level: heading[1].length, text: heading[2] }); continue }
    if (!line) { flushParagraph(); flushList(); flushTable(); continue }
    if (line.includes('|') && (tableRows.length > 0 || isTableSeparator(lines[lineIndex + 1] ?? ''))) {
      flushParagraph(); flushList()
      const cells = line.replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim())
      tableRows.push(cells)
      if (isTableSeparator(lines[lineIndex + 1] ?? '')) lineIndex += 1
      continue
    }
    flushTable()
    const listMatch = line.match(/^(?:([-*+])\s+|(\d+)[.)]\s+)(.*)$/u)
    if (listMatch) {
      flushParagraph()
      const ordered = Boolean(listMatch[2])
      if (listOrdered !== null && listOrdered !== ordered) flushList()
      if (!listItems.length && ordered) listStart = Number(listMatch[2]) || 1
      listOrdered = ordered
      const indentation = raw.match(/^\s*/u)?.[0].replace(/\t/g, '  ').length ?? 0
      listItems.push({ text: listMatch[3], level: Math.floor(indentation / 2) })
      continue
    }
    flushList()
    paragraph.push(line)
  }
  flushParagraph(); flushList(); flushTable()
  if (codeLines.length) blocks.push({ kind: 'code', text: codeLines.join('\n') })
  return blocks
}

function inlineMarkdown(text: string) {
  return text.split(/(\[[^\]]+\]\(#[^)]+\)|\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/gu).map((part, index) => {
    const anchor = part.match(/^\[([^\]]+)\]\((#[^)]+)\)$/u)
    if (anchor) return <a href={anchor[2]} key={index}>{anchor[1]}</a>
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={index}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index}>{part.slice(1, -1)}</code>
    if (part.startsWith('*') && part.endsWith('*')) return <em key={index}>{part.slice(1, -1)}</em>
    return <span key={index}>{part}</span>
  })
}

function StructuredDocumentContent({ text }: { text: string }) {
  return <div className="structured-document">{parseStructuredBlocks(text).map((block, index) => {
    if (block.kind === 'heading') {
      const Tag = (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] as const)[block.level - 1] ?? 'h6'
      return <Tag key={index}>{inlineMarkdown(block.text)}</Tag>
    }
    if (block.kind === 'paragraph') return <p key={index}>{inlineMarkdown(block.text)}</p>
    if (block.kind === 'code') return <pre key={index}><code>{block.text}</code></pre>
    if (block.kind === 'list') {
      const items = block.items.map((item, itemIndex) => <li key={itemIndex} style={{ marginLeft: item.level * 18 }}>{inlineMarkdown(item.text)}</li>)
      return block.ordered ? <ol key={index} start={block.start}>{items}</ol> : <ul key={index}>{items}</ul>
    }
    const [header, ...rows] = block.rows
    return <div className="structured-table-wrap" key={index}><table><thead><tr>{header.map((cell, cellIndex) => <th key={cellIndex}>{inlineMarkdown(cell)}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{inlineMarkdown(cell)}</td>)}</tr>)}</tbody></table></div>
  })}</div>
}

function AccountPopover({ user, accounts, onSwitchAccount }: { user: AuthUser | null; accounts: RememberedAccount[]; onSwitchAccount: (email?: string) => void }) {
  const accountList = user && !accounts.some((account) => account.email === user.email)
    ? [{ email: user.email, display_name: user.display_name || user.email }, ...accounts]
    : accounts
  const otherAccounts = accountList.filter((account) => account.email !== user?.email)
  return <section className="account-popover" role="dialog" aria-label="User accounts"><div className="account-popover-current"><span className="account-mini-avatar">{user?.display_name?.slice(0, 1).toUpperCase() || 'A'}</span><div><strong>{user?.display_name || 'Signed-in account'}</strong><small>{user?.email || 'Current account'}</small></div><CheckCircle2 size={15} /></div><div className="account-popover-heading"><span>Switch account</span><strong>{accountList.length}</strong></div><div className="account-popover-list">{otherAccounts.length ? otherAccounts.map((account) => <button className="account-switch-row" key={account.email} onClick={() => onSwitchAccount(account.email)} title={`Switch to ${account.email}`}><span className="account-mini-avatar">{account.display_name.slice(0, 1).toUpperCase()}</span><span className="account-switch-copy"><strong>{account.display_name || account.email}</strong><small>{account.email}</small></span><ArrowRight size={14} /></button>) : <p className="account-switch-empty">No other accounts have signed in here yet.</p>}</div><button className="account-add-row" onClick={() => onSwitchAccount()}><Plus size={15} /><span>Add account</span></button></section>
}

