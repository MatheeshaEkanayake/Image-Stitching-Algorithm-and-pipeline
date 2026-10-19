import type { Capture, Comment, Post, Profile, Session } from './types';

/** Error with the server's message (the server answers errors as {"error": "..."}). */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
  }
}

export type Api = ReturnType<typeof createApi>;

export function normaliseServerUrl(url: string): string {
  let value = url.trim().replace(/\/+$/, '');
  if (value && !/^https?:\/\//i.test(value)) value = `http://${value}`;
  return value;
}

/** Typed client for the stitching server. `token` may be null before sign-in. */
export function createApi(baseUrl: string, token: string | null) {
  const authHeaders = (): Record<string, string> =>
    token ? { Authorization: `Bearer ${token}` } : {};

  async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
    let response: Response;
    try {
      response = await fetch(`${baseUrl}${path}`, {
        method,
        headers: {
          ...authHeaders(),
          ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch {
      throw new ApiError(`Cannot reach the server at ${baseUrl}. Check the address and Wi-Fi.`, 0);
    }
    if (response.status === 204) return undefined as T;
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const message = (data && typeof data.error === 'string' && data.error) || `Request failed (${response.status})`;
      throw new ApiError(message, response.status);
    }
    return data as T;
  }

  return {
    baseUrl,
    token,
    authHeaders,

    health: () => request<{ status: string }>('GET', '/health'),
    register: (username: string, password: string) =>
      request<Session>('POST', '/auth/register', { username, password }),
    login: (username: string, password: string) =>
      request<Session>('POST', '/auth/login', { username, password }),
    logout: () => request<void>('POST', '/auth/logout'),

    library: () => request<Capture[]>('GET', '/captures'),
    capture: (id: string) => request<Capture>('GET', `/captures/${id}`),
    createCapture: (title = '') => request<Capture>('POST', '/captures', { title }),
    finishCapture: (id: string) => request<Capture>('POST', `/captures/${id}/finish`, {}),
    updateCapture: (id: string, changes: { title?: string; public?: boolean }) =>
      request<Capture>('PATCH', `/captures/${id}`, changes),
    deleteCapture: (id: string) => request<void>('DELETE', `/captures/${id}`),

    /** Upload one photo the moment it is taken. Re-sending the same index replaces it. */
    async uploadPhoto(id: string, index: number, uri: string): Promise<{ photo_count: number }> {
      const form = new FormData();
      form.append('index', String(index));
      // React Native's FormData accepts a {uri, name, type} file descriptor.
      form.append('file', { uri, name: `photo-${index}.jpg`, type: 'image/jpeg' } as unknown as Blob);
      let response: Response;
      try {
        response = await fetch(`${baseUrl}/captures/${id}/photos`, {
          method: 'POST',
          headers: authHeaders(),
          body: form,
        });
      } catch {
        throw new ApiError('Upload failed: no connection to the server.', 0);
      }
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new ApiError(data?.error ?? `Upload failed (${response.status})`, response.status);
      return data;
    },

    thumbUrl: (id: string) => `${baseUrl}/captures/${id}/thumb.jpg`,
    /** Viewer page; the token travels in the URL fragment, which is never sent to the server. */
    viewerUrl: (id: string) => `${baseUrl}/captures/${id}/view#token=${encodeURIComponent(token ?? '')}`,

    feed: (before?: number) =>
      request<Post[]>('GET', `/posts?limit=20${before ? `&before=${before}` : ''}`),
    post: (id: number) => request<Post>('GET', `/posts/${id}`),
    createPost: (body: string, captureId?: string | null) =>
      request<Post>('POST', '/posts', { body, capture_id: captureId ?? null }),
    deletePost: (id: number) => request<void>('DELETE', `/posts/${id}`),
    like: (id: number, liked: boolean) => request<Post>(liked ? 'PUT' : 'DELETE', `/posts/${id}/like`),
    comments: (postId: number) => request<Comment[]>('GET', `/posts/${postId}/comments`),
    addComment: (postId: number, body: string) =>
      request<Comment>('POST', `/posts/${postId}/comments`, { body }),
    deleteComment: (id: number) => request<void>('DELETE', `/comments/${id}`),
    profile: (username: string) => request<Profile>('GET', `/users/${encodeURIComponent(username)}`),
  };
}
