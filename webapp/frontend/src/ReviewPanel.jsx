import { useEffect, useMemo, useState } from 'react';
import BoxCorrectionModal from './BoxCorrectionModal';


const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8081';

const PAGE_SIZE = 20;


function ReviewPanel({ jobId }) {
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
        const response = await fetch(
          `${API_BASE_URL}/api/jobs/${jobId}/reviews`
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
  }, [jobId]);

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
      const response = await fetch(
        `${API_BASE_URL}/api/jobs/${jobId}/reviews/${detection.detection_id}`,
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
      <div style={styles.errorBox}>
        Unable to load review data: {error}
      </div>
    );
  }

  if (!reviewData) {
    return (
      <div style={styles.loadingBox}>
        Loading detection reviews...
      </div>
    );
  }

  const { summary } = reviewData;

  return (
    <section style={styles.panel}>
      <div style={styles.headingRow}>
        <div>
          <h3 style={styles.heading}>
            Human Detection Review
          </h3>

          <p style={styles.description}>
            Approve correct detections, reject false
            positives, or correct inaccurate labels.
          </p>
        </div>

        <select
          value={filter}
          onChange={changeFilter}
          style={styles.select}
        >
          <option value="all">All detections</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="corrected">Corrected</option>
        </select>
      </div>

      <div style={styles.summaryGrid}>
        <SummaryCard
          label="Total"
          value={summary.total}
        />

        <SummaryCard
          label="Pending"
          value={summary.pending}
        />

        <SummaryCard
          label="Approved"
          value={summary.approved}
        />

        <SummaryCard
          label="Rejected"
          value={summary.rejected}
        />

        <SummaryCard
          label="Corrected"
          value={summary.corrected}
        />
      </div>

      {error && (
        <div style={styles.errorBox}>
          {error}
        </div>
      )}

      {visibleDetections.length === 0 ? (
        <div style={styles.emptyBox}>
          No detections match this filter.
        </div>
      ) : (
        <div style={styles.tableWrapper}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.headerCell}>
                  Time
                </th>

                <th style={styles.headerCell}>
                  Frame
                </th>

                <th style={styles.headerCell}>
                  Label
                </th>

                <th style={styles.headerCell}>
                  Confidence
                </th>

                <th style={styles.headerCell}>
                  Status
                </th>

                <th style={styles.headerCell}>
                  Review
                </th>
              </tr>
            </thead>

            <tbody>
              {visibleDetections.map((detection) => {
                const updating =
                  updatingId ===
                  detection.detection_id;

                return (
                  <tr key={detection.detection_id}>
                    <td style={styles.cell}>
                      {detection.timestamp_seconds}s
                    </td>

                    <td style={styles.cell}>
                      {detection.frame}
                    </td>

                    <td style={styles.cell}>
                      {detection.label}
                    </td>

                    <td style={styles.cell}>
                      {Math.round(
                        detection.score * 100
                      )}
                      %
                    </td>

                    <td style={styles.cell}>
                      <span
                        style={{
                          ...styles.statusBadge,
                          ...getStatusStyle(
                            detection.status
                          ),
                        }}
                      >
                        {detection.status}
                      </span>
                    </td>

                    <td style={styles.cell}>
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
                          style={styles.approveButton}
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
                          style={styles.rejectButton}
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
                          style={styles.correctButton}
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
          style={styles.pageButton}
        >
          Previous
        </button>

        <span>
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
          style={styles.pageButton}
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


function SummaryCard({ label, value }) {
  return (
    <div style={styles.summaryCard}>
      <strong style={styles.summaryValue}>
        {value}
      </strong>

      <span style={styles.summaryLabel}>
        {label}
      </span>
    </div>
  );
}


function getStatusStyle(status) {
  const colors = {
    pending: {
      background: '#fff4cc',
      color: '#765b00',
    },
    approved: {
      background: '#dcfce7',
      color: '#166534',
    },
    rejected: {
      background: '#fee2e2',
      color: '#991b1b',
    },
    corrected: {
      background: '#dbeafe',
      color: '#1e40af',
    },
  };

  return colors[status] || colors.pending;
}


const styles = {
  panel: {
    marginTop: '32px',
    paddingTop: '24px',
    borderTop: '1px solid #d9deea',
  },

  headingRow: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '20px',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
  },

  heading: {
    margin: 0,
  },

  description: {
    color: '#5d6678',
  },

  select: {
    padding: '10px 12px',
    border: '1px solid #b9c1d1',
    borderRadius: '8px',
  },

  summaryGrid: {
    display: 'grid',
    gridTemplateColumns:
      'repeat(auto-fit, minmax(110px, 1fr))',
    gap: '12px',
    margin: '20px 0',
  },

  summaryCard: {
    display: 'flex',
    flexDirection: 'column',
    padding: '14px',
    background: '#f5f7fb',
    borderRadius: '8px',
  },

  summaryValue: {
    fontSize: '24px',
  },

  summaryLabel: {
    marginTop: '4px',
    color: '#5d6678',
  },

  tableWrapper: {
    overflowX: 'auto',
    border: '1px solid #d9deea',
    borderRadius: '8px',
  },

  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },

  headerCell: {
    padding: '12px',
    textAlign: 'left',
    background: '#f5f7fb',
    borderBottom: '1px solid #d9deea',
  },

  cell: {
    padding: '12px',
    borderBottom: '1px solid #edf0f5',
  },

  statusBadge: {
    display: 'inline-block',
    padding: '5px 9px',
    borderRadius: '999px',
    fontSize: '13px',
    textTransform: 'capitalize',
  },

  actions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
  },

  approveButton: {
    padding: '7px 10px',
    border: 0,
    borderRadius: '6px',
    background: '#15803d',
    color: '#ffffff',
    cursor: 'pointer',
  },

  rejectButton: {
    padding: '7px 10px',
    border: 0,
    borderRadius: '6px',
    background: '#b91c1c',
    color: '#ffffff',
    cursor: 'pointer',
  },

  correctButton: {
    padding: '7px 10px',
    border: '1px solid #175cd3',
    borderRadius: '6px',
    background: '#ffffff',
    color: '#175cd3',
    cursor: 'pointer',
  },

  pagination: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '16px',
    marginTop: '18px',
  },

  pageButton: {
    padding: '8px 12px',
    border: '1px solid #b9c1d1',
    borderRadius: '6px',
    background: '#ffffff',
    cursor: 'pointer',
  },

  loadingBox: {
    marginTop: '24px',
    padding: '18px',
    background: '#f5f7fb',
    borderRadius: '8px',
  },

  emptyBox: {
    padding: '20px',
    background: '#f5f7fb',
    borderRadius: '8px',
  },

  errorBox: {
    marginTop: '16px',
    padding: '12px',
    color: '#991b1b',
    background: '#fee2e2',
    borderRadius: '8px',
  },
};


export default ReviewPanel;