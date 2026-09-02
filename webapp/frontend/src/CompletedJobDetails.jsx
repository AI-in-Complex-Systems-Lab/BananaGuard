import ReviewPanel from './ReviewPanel';


function CompletedJobDetails({ job, apiBaseUrl }) {
  const resultUrl = job?.result_url
    ? `${apiBaseUrl}${job.result_url}`
    : null;

  return (
    <div style={styles.resultsBox}>
      <h3>Detection Summary</h3>

      <p>
        <strong>Processed frames:</strong>{' '}
        {job.processed_frames}
      </p>

      <p>
        <strong>Frames containing detections:</strong>{' '}
        {job.frames_with_detections}
      </p>

      <p>
        <strong>Total detections:</strong>{' '}
        {job.total_detections}
      </p>

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
    borderTop: '1px solid #d9deea',
  },

  resultVideo: {
    width: '100%',
    maxWidth: '720px',
    marginTop: '16px',
    background: '#000000',
  },

  downloadLink: {
    color: '#175cd3',
    fontWeight: 600,
  },

  timeline: {
    maxHeight: '280px',
    overflowY: 'auto',
    border: '1px solid #d9deea',
    borderRadius: '8px',
  },

  timelineItem: {
    padding: '10px 12px',
    borderBottom: '1px solid #edf0f5',
  },
};


export default CompletedJobDetails;
