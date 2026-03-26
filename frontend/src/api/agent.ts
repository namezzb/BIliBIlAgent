import { api, apiBase } from './client';

export const sendChat = (payload: {
  session_id?: string;
  user_id?: string;
  message: string;
}) => api.post('/api/chat', payload).then((r) => r.data);

export const confirmRun = (runId: string, approved: boolean) =>
  api.post(`/api/runs/${runId}/confirm`, { approved }).then((r) => r.data);

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

export function createRunEventSource(runId: string, follow = true): EventSource {
  return new EventSource(`${apiBase}/api/runs/${runId}/events?follow=${follow}`);
}
