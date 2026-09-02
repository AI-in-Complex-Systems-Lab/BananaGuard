import { useEffect, useState } from 'react';
import { API_BASE_URL } from './api';
import { useAuth } from './AuthContext';


function LoginPage() {
  const { login } = useAuth();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [bootstrapHint, setBootstrapHint] =
    useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadBootstrapHint() {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/auth/bootstrap-hint`
        );

        const data = await response.json();

        if (!cancelled && data.available) {
          setBootstrapHint(data);
        }
      } catch {
        // No hint available; the login form still works normally.
      }
    }

    loadBootstrapHint();

    return () => {
      cancelled = true;
    };
  }, []);

  function fillBootstrapCredentials() {
    if (!bootstrapHint) return;

    setUsername(bootstrapHint.username);
    setPassword(bootstrapHint.password);
    setError('');
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!username.trim() || !password) {
      setError('Enter your username and password');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      await login(username.trim(), password);
    } catch (loginError) {
      setError(loginError.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="login-screen">
      <div className="login-card card card-padded">
        <div className="login-header">
          <div className="login-mark">BG</div>

          <div>
            <h1 className="login-title">BananaGuard</h1>

            <p className="login-subtitle">
              Firearm Detection Platform
            </p>
          </div>
        </div>

        {bootstrapHint && (
          <div
            className="info-banner"
            style={{ marginBottom: 16 }}
          >
            <strong>First-time setup:</strong> no
            accounts have had their password changed
            yet. Default administrator login is{' '}
            <code>{bootstrapHint.username}</code> /{' '}
            <code>{bootstrapHint.password}</code>.
            <div style={{ marginTop: 10 }}>
              <button
                type="button"
                className="btn btn-sm"
                onClick={fillBootstrapCredentials}
              >
                Fill in credentials
              </button>
            </div>
            <div
              style={{
                marginTop: 8,
                fontSize: 12,
                opacity: 0.85,
              }}
            >
              This notice disappears automatically once
              the administrator password is changed.
            </div>
          </div>
        )}

        {error && (
          <div
            className="error-banner"
            style={{ marginBottom: 16 }}
          >
            {error}
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          className="form-stack"
        >
          <div>
            <label className="field-label">
              Username
            </label>

            <input
              type="text"
              className="text-input"
              value={username}
              autoComplete="username"
              autoFocus
              onChange={(event) =>
                setUsername(event.target.value)
              }
            />
          </div>

          <div>
            <label className="field-label">
              Password
            </label>

            <input
              type="password"
              className="text-input"
              value={password}
              autoComplete="current-password"
              onChange={(event) =>
                setPassword(event.target.value)
              }
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={submitting}
            style={{ width: '100%', padding: '12px' }}
          >
            {submitting ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p className="login-footer-note">
          Authorized personnel only. All access is logged.
        </p>
      </div>
    </div>
  );
}


export default LoginPage;
