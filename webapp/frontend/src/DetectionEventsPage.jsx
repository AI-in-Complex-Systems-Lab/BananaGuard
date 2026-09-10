import { useEffect, useMemo, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';


const PAGE_SIZE = 25;


function formatTimestamp(secondsSinceEpoch) {
  if (!secondsSinceEpoch) return 'Unknown time';

  return new Date(secondsSinceEpoch * 1000).toLocaleString();
}

function formatVideoOffset(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);

  return `${minutes}:${String(remainder).padStart(2, '0')}`;
}

/**
 * Flattens every job's detection_events into individual event rows.
 * /api/jobs already returns the full detection_events array per job
 * (see review_store/dataset_export exploration), so no backend change
 * is needed at the platform's current job-count scale — this is a
 * pure client-side view over data that's already public.
 */
function flattenEvents(jobs) {
  const events = [];

  for (const job of jobs) {
    if (!job.detection_events) continue;

    for (const frameEvent of job.detection_events) {
      for (const detection of frameEvent.detections) {
        events.push({
          key: `${job.job_id}-${frameEvent.frame}-${events.length}`,
          job_id: job.job_id,
          filename: job.filename,
          created_at: job.created_at,
          frame: frameEvent.frame,
          timestamp_seconds: frameEvent.timestamp_seconds,
          label: detection.label,
          score: detection.score,
        });
      }
    }
  }

  events.sort((a, b) => {
    if (b.created_at !== a.created_at) {
      return b.created_at - a.created_at;
    }

    return b.timestamp_seconds - a.timestamp_seconds;
  });

  return events;
}


function DetectionEventsPage({ onOpenReview }) {
  const { token } = useAuth();

  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [labelFilter, setLabelFilter] = useState('all');
  const [page, setPage] = useState(1);

  async function loadJobs() {
    setLoading(true);
    setError('');

    try {
      const response = await authFetch(token, '/api/jobs');
      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to load detection events'
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

  const allEvents = useMemo(() => flattenEvents(jobs), [jobs]);

  const labels = useMemo(
    () =>
      Array.from(
        new Set(allEvents.map((event) => event.label))
      ).sort(),
    [allEvents]
  );

  const filteredEvents = useMemo(() => {
    if (labelFilter === 'all') return allEvents;

    return allEvents.filter(
      (event) => event.label === labelFilter
    );
  }, [allEvents, labelFilter]);

  const pageCount = Math.max(
    1,
    Math.ceil(filteredEvents.length / PAGE_SIZE)
  );

  const visibleEvents = filteredEvents.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE
  );

  function changeFilter(event) {
    setLabelFilter(event.target.value);
    setPage(1);
  }

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
          <h2 className="page-title">Detection Events</h2>

          <p className="page-subtitle">
            Every individual detection the system has flagged,
            across all processed videos.
          </p>
        </div>

        <div
          style={{
            display: 'flex',
            gap: 10,
            alignItems: 'center',
          }}
        >
          <select
            value={labelFilter}
            onChange={changeFilter}
            className="text-input"
            style={{ maxWidth: 200 }}
          >
            <option value="all">All weapon types</option>

            {labels.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={loadJobs}
            className="btn"
          >
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="empty-state">Loading detection events...</div>
      ) : visibleEvents.length === 0 ? (
        <div className="empty-state">
          No detection events match this filter.
        </div>
      ) : (
        <div className="card" style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Uploaded</th>
                <th>Source Video</th>
                <th>Time in Video</th>
                <th>Weapon Type</th>
                <th>Confidence</th>
                <th>Review</th>
              </tr>
            </thead>

            <tbody>
              {visibleEvents.map((event) => (
                <tr key={event.key}>
                  <td>{formatTimestamp(event.created_at)}</td>
                  <td>{event.filename}</td>
                  <td>
                    {formatVideoOffset(event.timestamp_seconds)}
                  </td>
                  <td>
                    <span className="badge badge-danger">
                      {event.label}
                    </span>
                  </td>
                  <td>{Math.round(event.score * 100)}%</td>
                  <td>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => onOpenReview?.(event.job_id)}
                    >
                      Review
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {filteredEvents.length > 0 && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 16,
            marginTop: 18,
          }}
        >
          <button
            type="button"
            disabled={page === 1}
            onClick={() =>
              setPage((current) => Math.max(1, current - 1))
            }
            className="btn btn-ghost btn-sm"
          >
            Previous
          </button>

          <span style={{ color: 'var(--text-secondary)' }}>
            Page {page} of {pageCount}
          </span>

          <button
            type="button"
            disabled={page === pageCount}
            onClick={() =>
              setPage((current) => Math.min(pageCount, current + 1))
            }
            className="btn btn-ghost btn-sm"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}


export default DetectionEventsPage;
