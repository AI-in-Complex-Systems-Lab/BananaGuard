import { useEffect, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';


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


function DashboardPage({ onOpenJob }) {
  const { token } = useAuth();

  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setLoading(true);
      setError('');

      try {
        const response = await authFetch(
          token,
          '/api/dashboard'
        );

        const payload = await response.json();

        if (!response.ok) {
          throw new Error(
            payload.detail ||
              'Unable to load dashboard data'
          );
        }

        if (!cancelled) {
          setData(payload);
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

    loadDashboard();

    return () => {
      cancelled = true;
    };
  }, [token]);

  if (loading) {
    return (
      <div className="empty-state">
        Loading dashboard...
      </div>
    );
  }

  if (error) {
    return <div className="error-banner">{error}</div>;
  }

  const { totals, recent_jobs: recentJobs } = data;

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">
          Operations Overview
        </h2>

        <p className="page-subtitle">
          Aggregate detection and review activity across
          all processed footage.
        </p>
      </div>

      <div className="stat-grid">
        <div className="stat-card stat-accent-blue">
          <div className="stat-value">
            {totals.jobs}
          </div>

          <div className="stat-label">Total Jobs</div>
        </div>

        <div className="stat-card stat-accent-success">
          <div className="stat-value">
            {totals.completed_jobs}
          </div>

          <div className="stat-label">Completed</div>
        </div>

        <div className="stat-card stat-accent-warning">
          <div className="stat-value">
            {totals.processing_jobs}
          </div>

          <div className="stat-label">Processing</div>
        </div>

        <div className="stat-card stat-accent-danger">
          <div className="stat-value">
            {totals.failed_jobs}
          </div>

          <div className="stat-label">Failed</div>
        </div>

        <div className="stat-card stat-accent-blue">
          <div className="stat-value">
            {totals.total_detections}
          </div>

          <div className="stat-label">
            Total Detections
          </div>
        </div>

        <div className="stat-card stat-accent-warning">
          <div className="stat-value">
            {totals.pending_reviews}
          </div>

          <div className="stat-label">
            Pending Review
          </div>
        </div>

        <div className="stat-card stat-accent-success">
          <div className="stat-value">
            {totals.approved_reviews}
          </div>

          <div className="stat-label">Approved</div>
        </div>

        <div className="stat-card stat-accent-danger">
          <div className="stat-value">
            {totals.rejected_reviews +
              totals.corrected_reviews}
          </div>

          <div className="stat-label">
            Rejected / Corrected
          </div>
        </div>
      </div>

      <div className="card card-padded">
        <h3 style={{ marginTop: 0 }}>Recent Activity</h3>

        {recentJobs.length === 0 ? (
          <div className="empty-state">
            No jobs have been processed yet.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Uploaded By</th>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Detections</th>
                </tr>
              </thead>

              <tbody>
                {recentJobs.map((job) => (
                  <tr
                    key={job.job_id}
                    className="clickable"
                    onClick={() =>
                      onOpenJob(job.job_id)
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
                    <td>{job.total_detections}</td>
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


export default DashboardPage;
