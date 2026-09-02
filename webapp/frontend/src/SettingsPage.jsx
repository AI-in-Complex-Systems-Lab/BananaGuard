import { useEffect, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';


function SettingsPage() {
  const { token } = useAuth();

  const [settingsData, setSettingsData] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [savedMessage, setSavedMessage] = useState('');

  useEffect(() => {
    let cancelled = false;

    async function loadSettings() {
      try {
        const response = await authFetch(
          token,
          '/api/settings'
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load settings'
          );
        }

        if (!cancelled) {
          setSettingsData(data);
          setThreshold(data.confidence_threshold);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadSettings();

    return () => {
      cancelled = true;
    };
  }, [token]);

  async function handleSave(event) {
    event.preventDefault();
    setSaving(true);
    setError('');
    setSavedMessage('');

    try {
      const response = await authFetch(
        token,
        '/api/settings',
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            confidence_threshold: Number(threshold),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to save settings'
        );
      }

      setThreshold(data.confidence_threshold);
      setSavedMessage(
        'Detection sensitivity updated for all future frames.'
      );
    } catch (saveError) {
      setError(saveError.message);
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="empty-state">
        Loading settings...
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Settings</h2>

        <p className="page-subtitle">
          Platform-wide detection configuration. Changes
          apply immediately to new video uploads and the
          live camera feed.
        </p>
      </div>

      <div
        className="card card-padded"
        style={{ maxWidth: 520 }}
      >
        <h3 style={{ marginTop: 0 }}>
          Detection Confidence Threshold
        </h3>

        <p style={{ color: 'var(--text-secondary)' }}>
          Lower values surface more detections (including
          more false positives). Higher values are
          stricter and may miss weak signatures. Current
          range:{' '}
          {settingsData?.minimum_confidence_threshold} to{' '}
          {settingsData?.maximum_confidence_threshold}.
        </p>

        {error && (
          <div
            className="error-banner"
            style={{ marginBottom: 16 }}
          >
            {error}
          </div>
        )}

        {savedMessage && (
          <div
            className="info-banner"
            style={{ marginBottom: 16 }}
          >
            {savedMessage}
          </div>
        )}

        <form onSubmit={handleSave}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 16,
              marginBottom: 20,
            }}
          >
            <input
              type="range"
              min={settingsData?.minimum_confidence_threshold}
              max={settingsData?.maximum_confidence_threshold}
              step={0.01}
              value={threshold}
              onChange={(event) =>
                setThreshold(
                  Number(event.target.value)
                )
              }
              style={{ flex: 1 }}
            />

            <span
              style={{
                fontWeight: 700,
                fontSize: 18,
                minWidth: 60,
                textAlign: 'right',
              }}
            >
              {Number(threshold).toFixed(2)}
            </span>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={saving}
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </form>
      </div>
    </div>
  );
}


export default SettingsPage;
