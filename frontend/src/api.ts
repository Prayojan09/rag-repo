export type DocumentRecord = {
  id: string
  filename: string
  media_type: string
  size_bytes: number
  version: number
  status: 'queued' | 'indexing' | 'ready' | 'failed' | string
  retry_count: number
  error_message: string | null
  created_at: string
  updated_at: string
}

export type DocumentPage = { items: DocumentRecord[]; total: number; limit: number; offset: number }
export type DocumentDetail = DocumentRecord & {
  structured_content: string | null
  chunks: { id: string; ordinal: number; text: string; page_number: number | null; heading: string | null }[]
}

export type SearchResult = {
  chunk_id: string
  document_id: string
  filename: string
  version: number
  ordinal: number
  page_number: number | null
  heading: string | null
  text: string
  score: number
  dense_score: number | null
  keyword_score: number | null
  rerank_score: number | null
}

export type SearchResponse = {
  query: string
  mode: string
  embedding_model: string | null
  reranker_model: string | null
  results: SearchResult[]
}

export type Citation = {
  chunk_id: string
  document_id: string
  filename: string
  version: number
  ordinal: number
  page_number: number | null
  heading: string | null
  text: string
}

export type Claim = { text: string; source_chunk_ids: string[] }

export type ChatResponse = {
  conversation_id: string | null
  answer_id: string
  question: string
  answer: string
  abstained: boolean
  abstention_reason: string | null
  claims: Claim[]
  conflicts: { description: string; source_chunk_ids: string[] }[]
  citations: Citation[]
  evidence_verified: boolean
  verification_model: string | null
  unverified_claim_count: number
  llm_prompt_tokens: number | null
  llm_completion_tokens: number | null
  retrieval_mode: string
  reranker_model: string | null
  llm_provider: string
  llm_model: string | null
}

export type ConversationSummary = { id: string; title: string; created_at: string; updated_at: string }
export type ConversationMessage = { id: string; role: 'user' | 'assistant'; content: string; response_data: ChatResponse | null; created_at: string }
export type ConversationDetail = ConversationSummary & { messages: ConversationMessage[] }

export type CitationDetail = Citation
export type ApiConfig = { baseUrl: string }
export type AuthUser = { email: string; display_name: string; picture_url: string | null }

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
// Remove credentials and endpoint overrides saved by older versions of the UI.
localStorage.removeItem('atlas-api-token')
localStorage.removeItem('atlas-api-base')
export const initialApiConfig: ApiConfig = { baseUrl: API_BASE }
let csrfToken = ''
export function setCsrfToken(value: string): void { csrfToken = value }

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(config: ApiConfig, path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  if (!['GET', 'HEAD', 'OPTIONS'].includes((init.method ?? 'GET').toUpperCase()) && csrfToken) {
    headers.set('X-CSRF-Token', csrfToken)
  }
  let response: Response
  try {
    response = await fetch(`${config.baseUrl}${path}`, { ...init, headers, credentials: 'include' })
  } catch {
    throw new ApiError('Could not reach the backend. Check that FastAPI is running and the API URL is correct.')
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      const value = body.detail
      detail = typeof value === 'string' ? value : value?.message ?? detail
    } catch { /* retain generic HTTP detail */ }
    if (response.status === 401) detail = 'Your sign-in session has expired. Please sign in again.'
    throw new ApiError(detail, response.status)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  authConfig: (c: ApiConfig) => request<{
    authentication_required: boolean
    google_enabled: boolean
    google_login_url: string | null
  }>(c, '/api/v1/auth/config'),
  me: (c: ApiConfig) => request<AuthUser & { csrf_token: string }>(c, '/api/v1/auth/me'),
  googleLoginUrl: (c: ApiConfig) => `${c.baseUrl}/api/v1/auth/google/start`,
  logout: (c: ApiConfig) => request<void>(c, '/api/v1/auth/logout', { method: 'POST' }),
  health: (c: ApiConfig) => request<{ status: string }>(c, '/health'),
  ready: (c: ApiConfig) => request<{ status: string; checks: Record<string, string> }>(c, '/ready'),
  documents: (c: ApiConfig, offset = 0) => request<DocumentPage>(c, `/api/v1/documents?limit=50&offset=${offset}`),
  document: (c: ApiConfig, id: string) => request<DocumentDetail>(c, `/api/v1/documents/${id}`),
  upload: (c: ApiConfig, file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<DocumentRecord>(c, '/api/v1/documents', { method: 'POST', body })
  },
  remove: (c: ApiConfig, id: string) => request<void>(c, `/api/v1/documents/${id}`, { method: 'DELETE' }),
  retry: (c: ApiConfig, id: string) => request<DocumentRecord>(c, `/api/v1/documents/${id}/retry`, { method: 'POST' }),
  reindex: (c: ApiConfig, id: string) => request<DocumentRecord>(c, `/api/v1/documents/${id}/reindex`, { method: 'POST' }),
  search: (c: ApiConfig, payload: { query: string; mode: string; top_k: number; rerank: boolean }) =>
    request<SearchResponse>(c, '/api/v1/search', { method: 'POST', body: JSON.stringify(payload) }),
  chat: (c: ApiConfig, payload: { question: string; mode: string; top_k: number; rerank: boolean; conversation_id?: string }) =>
    request<ChatResponse>(c, '/api/v1/chat', { method: 'POST', body: JSON.stringify(payload) }),
  conversations: (c: ApiConfig) => request<{ items: ConversationSummary[]; total: number }>(c, '/api/v1/conversations'),
  createConversation: (c: ApiConfig, title: string) => request<ConversationSummary>(c, `/api/v1/conversations?title=${encodeURIComponent(title)}`, { method: 'POST' }),
  conversation: (c: ApiConfig, id: string) => request<ConversationDetail>(c, `/api/v1/conversations/${id}`),
  removeConversation: (c: ApiConfig, id: string) => request<void>(c, `/api/v1/conversations/${id}`, { method: 'DELETE' }),
  citation: (c: ApiConfig, id: string) => request<CitationDetail>(c, `/api/v1/citations/${id}`),
  feedback: (c: ApiConfig, answerId: string, rating: 'helpful' | 'not_helpful') =>
    request(c, '/api/v1/feedback', { method: 'POST', body: JSON.stringify({ answer_id: answerId, rating }) }),
}
