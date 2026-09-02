import { useEffect, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';


function formatTimestamp(secondsSinceEpoch) {
  if (!secondsSinceEpoch) return 'Unknown';

  return new Date(
    secondsSinceEpoch * 1000
  ).toLocaleDateString();
}


function UsersAdminPage() {
  const { token, user: currentUser } = useAuth();

  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [formError, setFormError] = useState('');
  const [creating, setCreating] = useState(false);

  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('officer');

  async function loadUsers() {
    setLoading(true);
    setError('');

    try {
      const response = await authFetch(
        token,
        '/api/auth/users'
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to load users'
        );
      }

      setUsers(data);
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadUsers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleCreate(event) {
    event.preventDefault();
    setFormError('');
    setCreating(true);

    try {
      const response = await authFetch(
        token,
        '/api/auth/users',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            username,
            password,
            display_name: displayName,
            role,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to create user'
        );
      }

      setUsername('');
      setDisplayName('');
      setPassword('');
      setRole('officer');

      await loadUsers();
    } catch (createError) {
      setFormError(createError.message);
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(targetUsername) {
    const confirmed = window.confirm(
      `Remove access for "${targetUsername}"? This cannot be undone.`
    );

    if (!confirmed) return;

    try {
      const response = await authFetch(
        token,
        `/api/auth/users/${targetUsername}`,
        { method: 'DELETE' }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to remove user'
        );
      }

      await loadUsers();
    } catch (deleteError) {
      setError(deleteError.message);
    }
  }

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">User Management</h2>

        <p className="page-subtitle">
          Create and manage accounts for officers and
          analysts who access this platform.
        </p>
      </div>

      <div
        className="card card-padded"
        style={{ marginBottom: 22 }}
      >
        <h3 style={{ marginTop: 0 }}>Add User</h3>

        {formError && (
          <div
            className="error-banner"
            style={{ marginBottom: 16 }}
          >
            {formError}
          </div>
        )}

        <form
          onSubmit={handleCreate}
          style={{
            display: 'grid',
            gridTemplateColumns:
              'repeat(auto-fit, minmax(180px, 1fr))',
            gap: 14,
            alignItems: 'end',
          }}
        >
          <div>
            <label className="field-label">
              Username
            </label>

            <input
              type="text"
              className="text-input"
              value={username}
              onChange={(event) =>
                setUsername(event.target.value)
              }
              required
            />
          </div>

          <div>
            <label className="field-label">
              Display Name
            </label>

            <input
              type="text"
              className="text-input"
              value={displayName}
              onChange={(event) =>
                setDisplayName(event.target.value)
              }
              placeholder="Optional"
            />
          </div>

          <div>
            <label className="field-label">
              Temporary Password
            </label>

            <input
              type="password"
              className="text-input"
              value={password}
              onChange={(event) =>
                setPassword(event.target.value)
              }
              minLength={10}
              required
            />
          </div>

          <div>
            <label className="field-label">Role</label>

            <select
              className="text-input"
              value={role}
              onChange={(event) =>
                setRole(event.target.value)
              }
            >
              <option value="officer">Officer</option>
              <option value="admin">Admin</option>
            </select>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={creating}
          >
            {creating ? 'Adding...' : 'Add User'}
          </button>
        </form>
      </div>

      <div className="card card-padded">
        <h3 style={{ marginTop: 0 }}>All Users</h3>

        {error && (
          <div
            className="error-banner"
            style={{ marginBottom: 16 }}
          >
            {error}
          </div>
        )}

        {loading ? (
          <div className="empty-state">
            Loading users...
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Username</th>
                  <th>Role</th>
                  <th>Created</th>
                  <th></th>
                </tr>
              </thead>

              <tbody>
                {users.map((user) => (
                  <tr key={user.username}>
                    <td>{user.display_name}</td>
                    <td>{user.username}</td>
                    <td>
                      <span
                        className={`badge ${
                          user.role === 'admin'
                            ? 'badge-amber'
                            : 'badge-info'
                        }`}
                      >
                        {user.role}
                      </span>
                    </td>
                    <td>
                      {formatTimestamp(user.created_at)}
                    </td>
                    <td>
                      {user.username !==
                        currentUser.username && (
                        <button
                          type="button"
                          className="btn btn-danger btn-sm"
                          onClick={() =>
                            handleDelete(user.username)
                          }
                        >
                          Remove
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}


export default UsersAdminPage;
