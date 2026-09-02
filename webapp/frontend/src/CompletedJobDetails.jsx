import { mediaUrl } from './api';
import { useAuth } from './AuthContext';
import ReviewPanel from './ReviewPanel';


function CompletedJobDetails({ job }) {
  const { token } = useAuth();

  const resultUrl = job?.result_url
    ? mediaUrl(job.result_url, token)
    : null;

  return (
    <div className="card-padded" style={styles.resultsBox}>
      <h3>Detection Summary</h3>

      <div className="stat-grid">
        <div className="stat-card stat-accent-blue">
          <div className="stat-value">
            {job.processed_frames}
          </div>

          <div className="stat-label">
            Processed Frames
          </div>
        </div>

        <div className="stat-card stat-accent-warning">
          <div className="stat-value">
            {job.frames_with_detections}
          </div>

          <div className="stat-label">
            Frames with Detections
          </div>
        </div>

        <div className="stat-card stat-accent-danger">
          <div className="stat-value">
            {job.total_detections}
          </div>

          <div className="stat-label">
            Total Detections
          </div>
        </div>
      </div>

      {resultUrl && (
        <>
          <video
            controls
            src={resultUrl}
            style={styles.resultVideo}
          />

          <p>
            <a
              href={resultUrl}
              download
              style={styles.downloadLink}
            >
              Download annotated video
            </a>
          </p>
        </>
      )}

      {job.detection_events?.length > 0 && (
        <div>
          <h3>Detection Timeline</h3>

          <div style={styles.timeline}>
            {job.detection_events
              .slice(0, 100)
              .map((event) => (
                <div
                  key={`${event.frame}-${event.timestamp_seconds}`}
                  style={styles.timelineItem}
                >
                  <strong>
                    {event.timestamp_seconds}s
                  </strong>

                  {' — '}

                  {event.detections.length} detection(s)
                </div>
              ))}
          </div>
        </div>
      )}

      <ReviewPanel jobId={job.job_id} />
    </div>
  );
}


const styles = {
  resultsBox: {
    marginTop: '24px',
    paddingTop: '18px',
    borderTop: '1px solid var(--border)',
  },

  resultVideo: {
    width: '100%',
    maxWidth: '720px',
    marginTop: '16px',
    background: '#000000',
    borderRadius: 'var(--radius-md)',
  },

  downloadLink: {
    color: 'var(--accent)',
    fontWeight: 600,
  },

  timeline: {
    maxHeight: '280px',
    overflowY: 'auto',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
  },

  timelineItem: {
    padding: '10px 12px',
    borderBottom: '1px solid var(--border)',
    color: 'var(--text-secondary)',
  },
};


export default CompletedJobDetails;
