import { api, apiBase } from './client';

export const getRun = (runId: string) =>
  api.get(`/api/runs/${runId}`).then((r) => r.data);

export const getSession = (sessionId: string) =>
  api.get(`/api/sessions/${sessionId}`).then((r) => r.data);

export const getUserMemory = (userId: string) =>
  api.get(`/api/users/${userId}/memory`).then((r) => r.data);

export const patchUserMemory = (
  userId: string,
  payload: Record<string, Record<string, string>>
) => api.patch(`/api/users/${userId}/memory`, payload).then((r) => r.data);

export const deleteUserMemory = (userId: string, group: string, key: string) =>
  api.delete(`/api/users/${userId}/memory/${group}/${encodeURIComponent(key)}`).then((r) => r.data);

export const knowledgeSearch = (payload: {
  query: string;
  top_k?: number;
  favorite_folder_ids?: string[];
  video_ids?: string[];
}) => api.post('/api/knowledge/search', payload).then((r) => r.data);

// ── Streaming helpers ────────────────────────────────────────────────────────

export type StreamChunk =
  | { type: 'token'; content: string }
  | { type: 'node'; node: string; data: Record<string, unknown> }
  | { type: 'interrupt'; data: Record<string, unknown> }
  | { type: 'done'; run_id: string; status: string; reply: string; intent?: string; route?: string; requires_confirmation: boolean; approval_status?: string; execution_plan?: unknown; pending_actions: unknown[] }
  | { type: 'error'; detail: string };

/**
 * Stream a chat turn via POST /api/chat/stream.
 * Calls `onChunk` for every SSE chunk. Returns the final `done` chunk.
 */
export async function streamChat(
  payload: { session_id?: string; user_id?: string; message: string },
  onChunk: (chunk: StreamChunk) => void,
  signal?: AbortSignal,
): Promise<StreamChunk & { type: 'done' }> {
  const biliCookie = localStorage.getItem('biliCookie');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (biliCookie) headers['X-Bilibili-Cookie'] = biliCookie;

  const resp = await fetch(`${apiBase}/api/chat/stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal,
  });

  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`Stream request failed: ${resp.status} ${detail}`);
  }

  const runId = resp.headers.get('X-Run-Id') ?? '';
  return _consumeStream(resp, runId, onChunk);
}

/**
 * Resume an interrupted run via POST /api/runs/{runId}/confirm/stream.
 */
export async function streamConfirm(
  runId: string,
  approved: boolean,
  onChunk: (chunk: StreamChunk) => void,
  signal?: AbortSignal,
): Promise<StreamChunk & { type: 'done' }> {
  const biliCookie = localStorage.getItem('biliCookie');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (biliCookie) headers['X-Bilibili-Cookie'] = biliCookie;

  const resp = await fetch(`${apiBase}/api/runs/${runId}/confirm/stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ approved }),
    signal,
  });

  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`Confirm stream failed: ${resp.status} ${detail}`);
  }

  return _consumeStream(resp, runId, onChunk);
}

async function _consumeStream(
  resp: Response,
  fallbackRunId: string,
  onChunk: (chunk: StreamChunk) => void,
): Promise<StreamChunk & { type: 'done' }> {
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let doneChunk: (StreamChunk & { type: 'done' }) | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (raw === '[DONE]') break;
      try {
        const chunk = JSON.parse(raw) as StreamChunk;
        onChunk(chunk);
        if (chunk.type === 'done') doneChunk = chunk;
      } catch {
        // ignore malformed lines
      }
    }
  }

  if (!doneChunk) {
    // Synthesize a minimal done if stream ended without one
    doneChunk = {
      type: 'done',
      run_id: fallbackRunId,
      status: 'completed',
      reply: '',
      requires_confirmation: false,
      pending_actions: [],
    };
  }
  return doneChunk;
}
