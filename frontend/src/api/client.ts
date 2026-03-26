import axios from 'axios';

// Use relative base so all requests go through Vite proxy -> no CORS issues
const BASE = import.meta.env.VITE_API_BASE ?? '';

export const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
});

// Attach Bilibili Cookie header when available
api.interceptors.request.use((config) => {
  const cookie = localStorage.getItem('biliCookie');
  if (cookie && config.headers) {
    config.headers['X-Bilibili-Cookie'] = cookie;
  }
  return config;
});

// Auto-redirect on 401
api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('biliCookie');
      localStorage.removeItem('biliAccount');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

// For SSE EventSource (must be absolute URL)
export const apiBase = 'http://localhost:8000';
