import {
  createContext,
  useContext,
  useEffect,
  useState,
} from 'react';
import { API_BASE_URL, authFetch } from './api';


const AuthContext = createContext(null);

const STORAGE_KEY = 'bananaguard_auth';

function readStoredSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeStoredSession(session) {
  if (session) {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(session)
    );
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}


export function AuthProvider({ children }) {
  const storedSession = readStoredSession();

  const [token, setToken] = useState(
    storedSession?.token || null
  );

  const [user, setUser] = useState(
    storedSession?.user || null
  );

  const [initializing, setInitializing] =
    useState(true);

  useEffect(() => {
    let cancelled = false;

    async function verifySession() {
      if (!token) {
        setInitializing(false);
        return;
      }

      try {
        const response = await authFetch(
          token,
          '/api/auth/me'
        );

        if (!response.ok) {
          throw new Error('Session is no longer valid');
        }

        const data = await response.json();

        if (!cancelled) {
          setUser(data);
        }
      } catch {
        if (!cancelled) {
          setToken(null);
          setUser(null);
          writeStoredSession(null);
        }
      } finally {
        if (!cancelled) {
          setInitializing(false);
        }
      }
    }

    verifySession();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function login(username, password) {
    const response = await fetch(
      `${API_BASE_URL}/api/auth/login`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || 'Unable to sign in'
      );
    }

    setToken(data.access_token);
    setUser(data.user);

    writeStoredSession({
      token: data.access_token,
      user: data.user,
    });
  }

  function logout() {
    setToken(null);
    setUser(null);
    writeStoredSession(null);
  }

  return (
    <AuthContext.Provider
      value={{
        token,
        user,
        login,
        logout,
        initializing,
        isAuthenticated: Boolean(token && user),
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}


// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error(
      'useAuth must be used within an AuthProvider'
    );
  }

  return context;
}
