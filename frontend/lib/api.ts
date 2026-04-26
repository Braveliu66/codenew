import type { AlgorithmEntry, Artifact, AuthResponse, MediaAsset, Project, Task, User, ViewerConfig } from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "three_dgs_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(TOKEN_KEY);
  }
}

export function isPublicPath(pathname: string): boolean {
  return pathname === "/login" || pathname === "/about";
}

async function request<T>(path: string, init?: RequestInit & { auth?: boolean }): Promise<T> {
  const auth = init?.auth ?? true;
  const token = auth ? getToken() : null;
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers
    },
    cache: "no-store"
  });
  if (response.status === 401) {
    clearToken();
    if (typeof window !== "undefined" && !isPublicPath(window.location.pathname)) {
      window.location.assign("/login");
    }
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(readErrorMessage(text, `${response.status} ${response.statusText}`));
  }
  return (await response.json()) as T;
}

function readErrorMessage(text: string, fallback: string): string {
  if (!text) return fallback;
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    return text;
  }
  return text;
}

export const api = {
  register: (payload: { username: string; password: string; email?: string }) =>
    request<AuthResponse>("/api/auth/register", { method: "POST", body: JSON.stringify(payload), auth: false }),
  login: (payload: { username: string; password: string }) =>
    request<AuthResponse>("/api/auth/login", { method: "POST", body: JSON.stringify(payload), auth: false }),
  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  me: () => request<User>("/api/me"),
  resources: () => request<{ cpu: Record<string, unknown>; gpu: Record<string, unknown>; workers?: Record<string, unknown> }>("/api/admin/system/resources"),
  algorithms: () => request<{ algorithms: AlgorithmEntry[] }>("/api/algorithms", { auth: false }),
  adminAlgorithms: () => request<{ algorithms: AlgorithmEntry[] }>("/api/admin/algorithms"),
  adminTasks: () => request<{ tasks: Task[] }>("/api/admin/tasks"),
  workers: () => request<{ workers: unknown[]; message?: string }>("/api/admin/workers"),
  projectSummary: () => request<Record<string, number>>("/api/projects/summary"),
  projects: () => request<{ projects: Project[] }>("/api/projects"),
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  createProject: (payload: { name: string; input_type: Project["input_type"]; tags: string[] }) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify(payload) }),
  uploadMedia: (projectId: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<MediaAsset>(`/api/projects/${projectId}/media`, { method: "POST", body });
  },
  mediaStats: (projectId: string) => request<Record<string, unknown>>(`/api/projects/${projectId}/media/stats`),
  startPreview: (projectId: string) =>
    request<Task>(`/api/projects/${projectId}/tasks/preview`, { method: "POST", body: JSON.stringify({ options: {} }) }),
  task: (id: string) => request<Task>(`/api/tasks/${id}`),
  artifacts: (projectId: string) => request<{ artifacts: Artifact[] }>(`/api/projects/${projectId}/artifacts`),
  viewerConfig: (projectId: string) => request<ViewerConfig>(`/api/projects/${projectId}/viewer-config`),
  feedback: (payload: { title: string; content: string; project_id?: string }) =>
    request<Record<string, unknown>>("/api/feedback", { method: "POST", body: JSON.stringify(payload) })
};

export function artifactUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

export function formatBytes(value: number | undefined | null): string {
  const bytes = value ?? 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}
