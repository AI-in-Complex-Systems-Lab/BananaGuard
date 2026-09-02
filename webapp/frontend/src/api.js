export const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8081';

export const WEBSOCKET_URL =
  import.meta.env.VITE_WS_URL || 'ws://localhost:8081/ws';


export async function authFetch(token, path, options = {}) {
  const headers = new Headers(options.headers || {});

  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  return fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });
}


export function mediaUrl(path, token) {
  const separator = path.includes('?') ? '&' : '?';

  const tokenParam = token
    ? `${separator}token=${encodeURIComponent(token)}`
    : '';

  return `${API_BASE_URL}${path}${tokenParam}`;
}


export function websocketUrl(token) {
  const separator = WEBSOCKET_URL.includes('?')
    ? '&'
    : '?';

  return `${WEBSOCKET_URL}${separator}token=${encodeURIComponent(token)}`;
}
