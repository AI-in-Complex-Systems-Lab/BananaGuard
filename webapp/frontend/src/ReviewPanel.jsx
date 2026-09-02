import { useEffect, useMemo, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';
import BoxCorrectionModal from './BoxCorrectionModal';


const PAGE_SIZE = 20;


function statusBadgeClass(status) {
  if (status === 'approved') return 'badge-success';
  if (status === 'rejected') return 'badge-danger';
  if (status === 'corrected') return 'badge-info';
  return 'badge-warning';
}


function ReviewPanel({ jobId }) {
  const { token } = useAuth();

  const [reviewData, setReviewData] = useState(null);
  const [filter, setFilter] = useState('all');
  const [page, setPage] = useState(1);
  const [updatingId, setUpdatingId] = useState(null);
  const [error, setError] = useState('');
  const [correctingDetection, setCorrectingDetection] =
    useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadReviews() {
      try {
        const response = await authFetch(
          token,
          `/api/jobs/${jobId}/reviews`
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load reviews'
          );
        }

        if (!cancelled) {
          setReviewData(data);
        }
      } catch (loadError) {
        console.error(loadError);

        if (!cancelled) {
          setError(loadError.message);
        }
      }
    }

    loadReviews();

    return () => {
      cancelled = true;
    };
  }, [jobId, token]);

  const filteredDetections = useMemo(() => {
    if (!reviewData) return [];

    if (filter === 'all') {
      return reviewData.detections;
    }

    return reviewData.detections.filter(
      (detection) => detection.status === filter
    );
  }, [filter, reviewData]);

  const pageCount = Math.max(
    1,
    Math.ceil(filteredDetections.length / PAGE_SIZE)
  );

  const visibleDetections = filteredDetections.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE
  );

  async function updateReview(
    detection,
    status,
    extraValues = {}
  ) {
    setUpdatingId(detection.detection_id);
    setError('');

    try {
      const response = await authFetch(
        token,
        `/api/jobs/${jobId}/reviews/${detection.detection_id}`,
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            status,
            ...extraValues,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to save review'
        );
      }

      setReviewData((currentData) => ({
        ...currentData,
        summary: data.summary,
        detections: currentData.detections.map(
          (currentDetection) =>
            currentDetection.detection_id ===
            data.detection.detection_id
              ? data.detection
              : currentDetection
        ),
      }));
    } catch (updateError) {
      console.error(updateError);
      setError(updateError.message);
    } finally {
      setUpdatingId(null);
    }
  }

  function openCorrectionModal(detection) {
    setCorrectingDetection(detection);
  }

  function saveCorrection(label, box) {
    const detection = correctingDetection;
    setCorrectingDetection(null);

    updateReview(detection, 'corrected', {
      label,
      box,
    });
  }

  function changeFilter(event) {
    setFilter(event.target.value);
    setPage(1);
  }

  if (error && !reviewData) {
    return (
      <div className="error-banner">
        Unable to load review data: {error}
      </div>
    );
  }

  if (!reviewData) {
    return (
      <div className="empty-state">
        Loading detection reviews...
      </div>
    );
  }

  const { summary } = reviewData;

  return (
    <section style={styles.panel}>
      <div style={styles.headingRow}>
        <div>
          <h3 style={{ margin: 0 }}>
            Human Detection Review
          </h3>

          <p
            style={{
              color: 'var(--text-secondary)',
              marginTop: 4,
            }}
          >
            Approve correct detections, reject false
            positives, or correct inaccurate labels and
            boxes.
          </p>
        </div>

        <select
          value={filter}
          onChange={changeFilter}
          className="text-input"
          style={{ maxWidth: 200 }}
        >
          <option value="all">All detections</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="corrected">Corrected</option>
        </select>
      </div>

      <div className="stat-grid">
        <SummaryCard
          label="Total"
          value={summary.total}
        />

        <SummaryCard
          label="Pending"
          value={summary.pending}
          accent="warning"
        />

        <SummaryCard
          label="Approved"
          value={summary.approved}
          accent="success"
        />

        <SummaryCard
          label="Rejected"
          value={summary.rejected}
          accent="danger"
        />

        <SummaryCard
          label="Corrected"
          value={summary.corrected}
          accent="blue"
        />
      </div>

      {error && (
        <div
          className="error-banner"
          style={{ marginBottom: 16 }}
        >
          {error}
        </div>
      )}

      {visibleDetections.length === 0 ? (
        <div className="empty-state">
          No detections match this filter.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Frame</th>
                <th>Label</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Reviewed By</th>
                <th>Review</th>
              </tr>
            </thead>

            <tbody>
              {visibleDetections.map((detection) => {
                const updating =
                  updatingId ===
                  detection.detection_id;

                return (
                  <tr key={detection.detection_id}>
                    <td>
                      {detection.timestamp_seconds}s
                    </td>

                    <td>{detection.frame}</td>

                    <td>{detection.label}</td>

                    <td>
                      {Math.round(
                        detection.score * 100
                      )}
                      %
                    </td>

                    <td>
                      <span
                        className={`badge ${statusBadgeClass(
                          detection.status
                        )}`}
                      >
                        {detection.status}
                      </span>
                    </td>

                    <td
                      style={{
                        color: 'var(--text-muted)',
                      }}
                    >
                      {detection.reviewed_by || '—'}
                    </td>

                    <td>
                      <div style={styles.actions}>
                        <button
                          type="button"
                          disabled={updating}
                          onClick={() =>
                            updateReview(
                              detection,
                              'approved'
                            )
                          }
                          className="btn btn-success btn-sm"
                        >
                          Approve
                        </button>

                        <button
                          type="button"
                          disabled={updating}
                          onClick={() =>
                            updateReview(
                              detection,
                              'rejected'
                            )
                          }
                          className="btn btn-danger btn-sm"
                        >
                          Reject
                        </button>

                        <button
                          type="button"
                          disabled={updating}
                          onClick={() =>
                            openCorrectionModal(
                              detection
                            )
                          }
                          className="btn btn-sm"
                        >
                          Correct
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div style={styles.pagination}>
        <button
          type="button"
          disabled={page === 1}
          onClick={() =>
            setPage((currentPage) =>
              Math.max(1, currentPage - 1)
            )
          }
          className="btn btn-ghost btn-sm"
        >
          Previous
        </button>

        <span
          style={{ color: 'var(--text-secondary)' }}
        >
          Page {page} of {pageCount}
        </span>

        <button
          type="button"
          disabled={page === pageCount}
          onClick={() =>
            setPage((currentPage) =>
              Math.min(
                pageCount,
                currentPage + 1
              )
            )
          }
          className="btn btn-ghost btn-sm"
        >
          Next
        </button>
      </div>

      {correctingDetection && (
        <BoxCorrectionModal
          jobId={jobId}
          detection={correctingDetection}
          onCancel={() => setCorrectingDetection(null)}
          onSave={saveCorrection}
        />
      )}
    </section>
  );
}


function SummaryCard({ label, value, accent }) {
  return (
    <div
      className={`stat-card${
        accent ? ` stat-accent-${accent}` : ''
      }`}
    >
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}


const styles = {
  panel: {
    marginTop: '32px',
    paddingTop: '24px',
    borderTop: '1px solid var(--border)',
  },

  headingRow: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '20px',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
    marginBottom: '18px',
  },

  actions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  },

  pagination: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '16px',
    marginTop: '18px',
  },
};


export default ReviewPanel;
