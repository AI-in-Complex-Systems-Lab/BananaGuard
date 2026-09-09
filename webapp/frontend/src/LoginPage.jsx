import { useState } from 'react';
import { useAuth } from './AuthContext';


function LoginPage() {
  const { login } = useAuth();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

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
