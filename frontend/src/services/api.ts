/**
 * Centralised API service layer.
 * All fetch calls go through here so error handling, auth refresh,
 * and base URL changes happen in one place.
 */

// In dev: empty string (Vite proxy handles /v1 → localhost:8000)
// In prod (Render/Docker): set VITE_API_URL=https://your-api.onrender.com
const BASE: string = import.meta.env.VITE_API_URL ?? "";

// ── In-memory access token (never stored in cookie or localStorage) ──────────
let _accessToken: string | null = null;

export function setAccessToken(token: string | null): void {
  _accessToken = token;
}

export function getAccessToken(): string | null {
  return _accessToken;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ApiUser {
  id: string;
  email: string;
  name: string;
  role: "user" | "clinician" | "admin";
  is_active: boolean;
}

export interface AuthResponse {
  user: ApiUser;
  access_token: string;   // returned in body; stored in JS memory only
  message?: string;
}

// ── Core fetch wrapper ────────────────────────────────────────────────────────

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function authHeaders(extra?: HeadersInit): Record<string, string> {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
    ...(extra as Record<string, string>),
  };
  if (_accessToken) h["Authorization"] = `Bearer ${_accessToken}`;
  return h;
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    credentials: "include",           // send HttpOnly refresh cookie when needed
    headers: authHeaders(options.headers as HeadersInit),
    ...options,
  });

  // Attempt silent token refresh on 401 (once, not for auth endpoints)
  if (res.status === 401 && !path.includes("/auth/")) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      const retry = await fetch(`${BASE}${path}`, {
        credentials: "include",
        headers: authHeaders(options.headers as HeadersInit),
        ...options,
      });
      if (retry.ok) return retry.json() as Promise<T>;
    }
    throw new ApiError(401, "Session expired — please log in again");
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch (_) { /* ignore */ }
    throw new ApiError(res.status, detail);
  }

  // 204 No Content
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

async function tryRefresh(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/v1/auth/refresh`, {
      method: "POST",
      credentials: "include",   // HttpOnly refresh cookie sent automatically
    });
    if (res.ok) {
      const data = await res.json() as { access_token?: string };
      if (data.access_token) setAccessToken(data.access_token);
      return true;
    }
    return false;
  } catch (_) {
    return false;
  }
}

// ── Auth endpoints ────────────────────────────────────────────────────────────

export const authApi = {
  login: (email: string, password: string) =>
    request<AuthResponse>("/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  logout: () =>
    request<{ message: string }>("/v1/auth/logout", { method: "POST" }),

  /** Uses the HttpOnly refresh cookie to silently restore the access token. */
  refresh: () =>
    fetch(`${BASE}/v1/auth/refresh`, { method: "POST", credentials: "include" })
      .then(async (res) => {
        if (!res.ok) throw new ApiError(res.status, "Refresh failed");
        return res.json() as Promise<AuthResponse>;
      }),

  me: () =>
    request<AuthResponse>("/v1/auth/me"),

  register: (email: string, password: string, name?: string, role?: string) =>
    request<AuthResponse>("/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, name: name ?? "", role: role ?? "user" }),
    }),

  listUsers: () =>
    request<ApiUser[]>("/v1/auth/users"),
};

// ── Prediction endpoints ──────────────────────────────────────────────────────

function _inferenceHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  if (_accessToken) h["Authorization"] = `Bearer ${_accessToken}`;
  return h;  // no Content-Type — browser sets multipart boundary automatically
}

async function _inferencePost(url: string, file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    credentials: "include",
    headers: _inferenceHeaders(),
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new ApiError(res.status, (err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const predictApi = {
  predict: (file: File) => _inferencePost("/v1/predict", file),
  explain: (file: File) => _inferencePost("/v1/explain", file),
};

// ── Chat endpoint ─────────────────────────────────────────────────────────────

export const chatApi = {
  send: (messages: { role: string; content: string }[], context: object | null) =>
    request<{ reply: string; model_used: string }>("/v1/chat", {
      method: "POST",
      body: JSON.stringify({ messages, context }),
    }),
};

// ── Patient records / feedback ────────────────────────────────────────────────

export interface PredictionRecord {
  id: string;
  request_id: string;
  patient_name: string;
  patient_id: string;
  patient_age: number | null;
  patient_gender: string;
  notes: string;
  model_version: string;
  primary_label: string;
  confidence: number;
  decision: string;
  feedback: "pending" | "correct" | "wrong";
  true_label: string | null;
  created_at: string;
  full_result: Record<string, unknown> | null;
}

export const recordsApi = {
  list: (params?: { limit?: number; offset?: number; all_users?: boolean }) => {
    const qs = new URLSearchParams();
    if (params?.limit)     qs.set("limit",     String(params.limit));
    if (params?.offset)    qs.set("offset",    String(params.offset));
    if (params?.all_users) qs.set("all_users", "true");
    return request<PredictionRecord[]>(`/v1/records?${qs}`);
  },

  get: (id: string) =>
    request<PredictionRecord>(`/v1/records/${id}`),

  submitFeedback: (id: string, feedback: "correct" | "wrong", true_label?: string) =>
    request<{ message: string; retraining_triggered: boolean }>(
      `/v1/records/${id}/feedback`,
      { method: "POST", body: JSON.stringify({ feedback, true_label }) },
    ),

  updatePatient: (
    id: string,
    info: {
      patient_name?: string;
      patient_id?: string;
      patient_age?: number | null;
      patient_gender?: string;
      notes?: string;
    },
  ) =>
    request<{ message: string }>(`/v1/records/${id}/patient`, {
      method: "PATCH",
      body: JSON.stringify(info),
    }),
};

export { ApiError };
