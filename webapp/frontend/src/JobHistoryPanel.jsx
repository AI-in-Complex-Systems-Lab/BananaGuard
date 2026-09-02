import { useEffect, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';
import CompletedJobDetails from './CompletedJobDetails';


function formatTimestamp(secondsSinceEpoch) {
  if (!secondsSinceEpoch) return 'Unknown time';

  return new Date(
    secondsSinceEpoch * 1000
  ).toLocaleString();
}

function statusBadgeClass(status) {
  if (status === 'completed') return 'badge-success';
  if (status === 'failed') return 'badge-danger';
  if (status === 'processing' || status === 'queued') {
    return 'badge-warning';
  }

  return 'badge-neutral';
}


function JobHistoryPanel({ initialSelectedJobId }) {
  const { token } = useAuth();

  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedJobId, setSelectedJobId] = useState(
    initialSelectedJobId || null
  );

  async function loadJobs() {
    setLoading(true);
    setError('');

    try {
      const response = await authFetch(
        token,
        '/api/jobs'
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to load job history'
        );
      }

      setJobs(data);
    } catch (loadError) {
      console.error(loadError);
      setError(loadError.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (initialSelectedJobId) {
      setSelectedJobId(initialSelectedJobId);
    }
  }, [initialSelectedJobId]);

  const selectedJob = jobs.find(
    (job) => job.job_id === selectedJobId
  );

  return (
    <div>
      <div
        className="page-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 20,
          flexWrap: 'wrap',
        }}
      >
        <div>
          <h2 className="page-title">Job History</h2>

          <p className="page-subtitle">
            Every processed job is persisted, so past
            uploads and their reviews stay reachable
            even after a server restart.
          </p>
        </div>

        <button
          type="button"
          onClick={loadJobs}
          className="btn"
        >
          Refresh
        </button>
      </div>

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
          Loading jobs...
        </div>
      ) : jobs.length === 0 ? (
        <div className="empty-state">No jobs yet.</div>
      ) : (
        <div className="card" style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Uploaded By</th>
                <th>Date</th>
                <th>Status</th>
              </tr>
            </thead>

            <tbody>
              {jobs.map((job) => (
                <tr
                  key={job.job_id}
                  className="clickable"
                  onClick={() =>
                    setSelectedJobId(
                      job.job_id === selectedJobId
                        ? null
                        : job.job_id
                    )
                  }
                  style={
                    job.job_id === selectedJobId
                      ? {
                          background:
                            'var(--bg-hover)',
                        }
                      : undefined
                  }
                >
                  <td>{job.filename}</td>
                  <td>{job.uploaded_by || '—'}</td>
                  <td>
                    {formatTimestamp(job.created_at)}
                  </td>
                  <td>
                    <span
                      className={`badge ${statusBadgeClass(
                        job.status
                      )}`}
                    >
                      {job.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedJob && (
        <div
          className="card"
          style={{ marginTop: 24, padding: 22 }}
        >
          <h3 style={{ marginTop: 0 }}>
            {selectedJob.filename}
          </h3>

          <p>
            <strong>Status:</strong>{' '}
            <span
              className={`badge ${statusBadgeClass(
                selectedJob.status
              )}`}
            >
              {selectedJob.status}
            </span>
          </p>

          {selectedJob.status === 'completed' ? (
            <CompletedJobDetails job={selectedJob} />
          ) : (
            <p style={{ color: 'var(--text-secondary)' }}>
              {selectedJob.message}
            </p>
          )}
        </div>
      )}
    </div>
  );
}


export default JobHistoryPanel;
