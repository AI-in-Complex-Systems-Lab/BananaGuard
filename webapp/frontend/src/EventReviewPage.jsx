import { useEffect, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';
import ReviewPanel from './ReviewPanel';


function formatTimestamp(secondsSinceEpoch) {
  if (!secondsSinceEpoch) return 'Unknown time';

  return new Date(secondsSinceEpoch * 1000).toLocaleString();
}

/**
 * First-class front door for the review/correction workflow, which
 * previously only existed nested inside Job History's per-job detail
 * view. Reuses ReviewPanel unmodified; the Job History path into it
 * (via CompletedJobDetails) is left exactly as-is alongside this.
 */
function EventReviewPage({ initialJobId }) {
  const { token } = useAuth();

  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedJobId, setSelectedJobId] = useState(
    initialJobId || null
  );

  useEffect(() => {
    let cancelled = false;

    async function loadJobs() {
      setLoading(true);
      setError('');

      try {
        const response = await authFetch(token, '/api/jobs');
        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load videos for review'
          );
        }

        if (!cancelled) {
          setJobs(
            data.filter((job) => job.status === 'completed')
          );
        }
      } catch (loadError) {
        console.error(loadError);

        if (!cancelled) {
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadJobs();

    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    if (initialJobId) {
      setSelectedJobId(initialJobId);
    }
  }, [initialJobId]);

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Event Review</h2>

        <p className="page-subtitle">
          Pick a processed video, then approve, reject, or correct
          each detection it produced.
        </p>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="empty-state">Loading videos...</div>
      ) : jobs.length === 0 ? (
        <div className="empty-state">
          No processed videos are available to review yet.
        </div>
      ) : (
        <div className="card" style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Uploaded</th>
                <th>Pending</th>
                <th>Approved</th>
                <th>Rejected</th>
                <th>Corrected</th>
              </tr>
            </thead>

            <tbody>
              {jobs.map((job) => {
                const summary = job.review_summary || {};

                return (
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
                        ? { background: 'var(--bg-hover)' }
                        : undefined
                    }
                  >
                    <td>{job.filename}</td>
                    <td>{formatTimestamp(job.created_at)}</td>
                    <td>{summary.pending ?? 0}</td>
                    <td>{summary.approved ?? 0}</td>
                    <td>{summary.rejected ?? 0}</td>
                    <td>{summary.corrected ?? 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {selectedJobId && <ReviewPanel jobId={selectedJobId} />}
    </div>
  );
}


export default EventReviewPage;
