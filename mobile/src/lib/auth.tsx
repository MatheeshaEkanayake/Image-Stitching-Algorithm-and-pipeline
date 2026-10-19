import * as SecureStore from 'expo-secure-store';
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

import { ApiError, createApi, normaliseServerUrl, type Api } from './api';
import type { UserRef } from './types';

const KEYS = { server: 'server_url', token: 'session_token', user: 'session_user' } as const;

type AuthState = {
  ready: boolean; // false while restoring the saved session
  serverUrl: string;
  user: UserRef | null;
  api: Api;
  signIn: (serverUrl: string, username: string, password: string, register: boolean) => Promise<void>;
  signOut: () => Promise<void>;
  /** Call when the server says the session is no longer valid (401). */
  handleUnauthorized: (error: unknown) => boolean;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [serverUrl, setServerUrl] = useState('');
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<UserRef | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [savedServer, savedToken, savedUser] = await Promise.all([
          SecureStore.getItemAsync(KEYS.server),
          SecureStore.getItemAsync(KEYS.token),
          SecureStore.getItemAsync(KEYS.user),
        ]);
        if (savedServer) setServerUrl(savedServer);
        if (savedToken && savedUser) {
          setToken(savedToken);
          setUser(JSON.parse(savedUser) as UserRef);
        }
      } catch {
        // Unreadable secure storage: start signed out.
      } finally {
        setReady(true);
      }
    })();
  }, []);

  const clearSession = useCallback(async () => {
    setToken(null);
    setUser(null);
    await Promise.all([SecureStore.deleteItemAsync(KEYS.token), SecureStore.deleteItemAsync(KEYS.user)]).catch(
      () => undefined,
    );
  }, []);

  const signIn = useCallback(async (url: string, username: string, password: string, register: boolean) => {
    const base = normaliseServerUrl(url);
    if (!base) throw new ApiError('Enter the server address.', 0);
    const client = createApi(base, null);
    const session = register ? await client.register(username.trim(), password) : await client.login(username.trim(), password);
    await Promise.all([
      SecureStore.setItemAsync(KEYS.server, base),
      SecureStore.setItemAsync(KEYS.token, session.token),
      SecureStore.setItemAsync(KEYS.user, JSON.stringify(session.user)),
    ]);
    setServerUrl(base);
    setToken(session.token);
    setUser(session.user);
  }, []);

  const api = useMemo(() => createApi(serverUrl, token), [serverUrl, token]);

  const signOut = useCallback(async () => {
    await api.logout().catch(() => undefined); // still sign out locally if offline
    await clearSession();
  }, [api, clearSession]);

  const handleUnauthorized = useCallback(
    (error: unknown) => {
      if (error instanceof ApiError && error.status === 401) {
        void clearSession();
        return true;
      }
      return false;
    },
    [clearSession],
  );

  const value = useMemo(
    () => ({ ready, serverUrl, user, api, signIn, signOut, handleUnauthorized }),
    [ready, serverUrl, user, api, signIn, signOut, handleUnauthorized],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const value = useContext(AuthContext);
  if (!value) throw new Error('useAuth must be used inside <AuthProvider>');
  return value;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong.';
}
