import { useEffect, useState } from 'react';
import CompletedJobDetails from './CompletedJobDetails';


const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8081';


function formatTimestamp(secondsSinceEpoch) {
  if (!secondsSinceEpoch) return 'Unknown time';

  return new Date(
    secondsSinceEpoch * 1000
  ).toLocaleString();
}


function JobHistoryPanel() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedJobId, setSelectedJobId] =
    useState(null);

  async function loadJobs() {
    setLoading(true);
    setError('');

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/jobs`
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
  }, []);

  const selectedJob = jobs.find(
    (job) => job.job_id === selectedJobId
  );

  return (
    <section style={styles.panel}>
      <div style={styles.headingRow}>
        <div>
          <h2 style={styles.heading}>Job History</h2>

          <p style={styles.description}>
            Jobs persist across backend restarts, so
            past uploads and their reviews stay
            reachable here.
          </p>
        </div>

        <button
          type="button"
          onClick={loadJobs}
          style={styles.refreshButton}
        >
          Refresh
        </button>
      </div>

      {error && (
        <div style={styles.errorBox}>{error}</div>
      )}

      {loading ? (
        <div style={styles.loadingBox}>
          Loading jobs...
        </div>
      ) : jobs.length === 0 ? (
        <div style={styles.emptyBox}>No jobs yet.</div>
      ) : (
        <div style={styles.list}>
          {jobs.map((job) => (
            <button
              type="button"
              key={job.job_id}
              onClick={() =>
                setSelectedJobId(
                  job.job_id === selectedJobId
                    ? null
                    : job.job_id
                )
              }
              style={{
                ...styles.listItem,
                ...(job.job_id === selectedJobId
                  ? styles.listItemActive
                  : {}),
              }}
            >
              <span style={styles.listItemFile}>
                {job.filename}
              </span>

              <span style={styles.listItemMeta}>
                {formatTimestamp(job.created_at)}
                {' — '}
                {job.status}
              </span>
            </button>
          ))}
        </div>
      )}

      {selectedJob && (
        <div style={styles.detailBox}>
          <h3>{selectedJob.filename}</h3>

          <p>
            <strong>Status:</strong>{' '}
            {selectedJob.status}
          </p>

          {selectedJob.status === 'completed' ? (
            <CompletedJobDetails
              job={selectedJob}
              apiBaseUrl={API_BASE_URL}
            />
          ) : (
            <p>{selectedJob.message}</p>
          )}
        </div>
      )}
    </section>
  );
}


const styles = {
  panel: {
    maxWidth: '900px',
  },

  headingRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '20px',
    flexWrap: 'wrap',
  },

  heading: {
    margin: 0,
  },

  description: {
    color: '#5d6678',
  },

  refreshButton: {
    border: '1px solid #175cd3',
    background: '#ffffff',
    color: '#175cd3',
    padding: '8px 14px',
    borderRadius: '8px',
    cursor: 'pointer',
  },

  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    marginTop: '16px',
  },

  listItem: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '12px',
    padding: '12px 16px',
    border: '1px solid #d9deea',
    borderRadius: '8px',
    background: '#ffffff',
    cursor: 'pointer',
    textAlign: 'left',
  },

  listItemActive: {
    border: '1px solid #175cd3',
    background: '#eef4ff',
  },

  listItemFile: {
    fontWeight: 600,
  },

  listItemMeta: {
    color: '#5d6678',
  },

  detailBox: {
    marginTop: '24px',
    padding: '22px',
    border: '1px solid #d9deea',
    borderRadius: '10px',
  },

  loadingBox: {
    marginTop: '18px',
    padding: '18px',
    background: '#f5f7fb',
    borderRadius: '8px',
  },

  emptyBox: {
    marginTop: '18px',
    padding: '18px',
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


export default JobHistoryPanel;
