import { useEffect, useMemo, useState } from 'react';
import { authFetch, mediaUrl } from './api';
import { useAuth } from './AuthContext';
import DetectionSnapshot from './DetectionSnapshot';


function flattenEvents(job) {
  const rows = [];

  for (const event of job?.detection_events || []) {
    for (const detection of event.detections) {
      rows.push({
        frame: event.frame,
        timestamp_seconds: event.timestamp_seconds,
        label: detection.label,
        score: detection.score,
        box: detection.box,
      });
    }
  }

  return rows.sort(
    (a, b) => a.timestamp_seconds - b.timestamp_seconds
  );
}

function StatBlock({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-value">
        {value === null || value === undefined ? '—' : value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function JobMetadataCard({ jobView }) {
  const { metadata } = jobView;

  return (
    <div className="card card-padded model-compare-card">
      <div className="model-compare-card-header">
        <span className="badge badge-info">
          {metadata.detector_type}
        </span>
        <h4 style={{ margin: 0 }}>{jobView.filename}</h4>
      </div>

      <dl className="model-compare-meta-list">
        <dt>Text prompts</dt>
        <dd>
          {metadata.supports_text_prompts
            ? (metadata.prompts || []).join('  ') || '—'
            : 'Not used — this is a trained fixed-class detector, not a text-prompted one.'}
        </dd>

        <dt>Confidence threshold</dt>
        <dd>{metadata.confidence_threshold ?? '—'}</dd>

        <dt>Source video</dt>
        <dd>
          {metadata.source_fps ?? '—'} fps &middot;{' '}
          {metadata.source_duration_seconds ?? '—'}s
        </dd>

        <dt>Frames analyzed</dt>
        <dd>
          {metadata.analyzed_frames ?? '—'} of{' '}
          {metadata.total_frames ?? '—'} total
          {metadata.frame_stride > 1
            ? ` (every ${metadata.frame_stride}th frame)`
            : ' (every frame)'}
        </dd>

        <dt>Processing time</dt>
        <dd>
          {metadata.elapsed_processing_seconds ?? '—'}s &middot;{' '}
          {metadata.throughput_analyzed_fps ?? '—'} analyzed fps
        </dd>
      </dl>
    </div>
  );
}

function JobSummaryCard({ jobView }) {
  const { summary } = jobView;

  return (
    <div className="card card-padded model-compare-card">
      <div className="stat-grid">
        <StatBlock
          label="Total detections"
          value={summary.total_detections}
        />
        <StatBlock
          label="Frames with detections"
          value={summary.frames_with_detections}
        />
        <StatBlock
          label="Detections / analyzed frame"
          value={summary.detections_per_analyzed_frame}
        />
        <StatBlock
          label="Avg confidence"
          value={summary.average_confidence}
        />
        <StatBlock
          label="Max confidence"
          value={summary.max_confidence}
        />
      </div>

      <p className="model-compare-note">
        By category:{' '}
        {Object.entries(summary.detections_by_category).length > 0
          ? Object.entries(summary.detections_by_category)
              .map(([label, count]) => `${label}: ${count}`)
              .join(', ')
          : 'none'}
      </p>

      <p className="model-compare-note">
        First detection at{' '}
        {summary.first_detection_timestamp ?? '—'}s, last at{' '}
        {summary.last_detection_timestamp ?? '—'}s.
      </p>

      <p className="model-compare-caveat">
        Confidence scores are not directly comparable across model
        families — each reflects its own internal calibration, not a
        shared probability scale.
      </p>
    </div>
  );
}

function Timeline({ timeline }) {
  if (!timeline || timeline.bins.length === 0) {
    return (
      <div className="empty-state">
        Not enough data to build a timeline.
      </div>
    );
  }

  return (
    <div>
      <div className="model-compare-timeline">
        {timeline.bins.map((bin, index) => {
          let className = 'model-compare-timeline-bin';

          if (bin.both) className += ' both';
          else if (bin.job_a) className += ' job-a';
          else if (bin.job_b) className += ' job-b';

          return (
            <div
              key={index}
              className={className}
              title={`${bin.start_seconds}s – ${bin.end_seconds}s`}
            />
          );
        })}
      </div>

      <div className="model-compare-timeline-legend">
        <span>
          <span className="legend-swatch job-a" /> Job A only
        </span>
        <span>
          <span className="legend-swatch job-b" /> Job B only
        </span>
        <span>
          <span className="legend-swatch both" /> Both (same{' '}
          {timeline.window_seconds}s window — not necessarily the
          same physical object)
        </span>
      </div>
    </div>
  );
}

function EventColumn({ title, rows, jobId, selected, onSelect }) {
  return (
    <div className="model-compare-event-column">
      <h4 style={{ marginTop: 0 }}>{title}</h4>

      {rows.length === 0 ? (
        <div className="empty-state">No detections.</div>
      ) : (
        <div className="model-compare-event-list">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Label</th>
                <th>Score</th>
                <th></th>
              </tr>
            </thead>

            <tbody>
              {rows.map((row, index) => (
                <tr
                  key={`${row.frame}-${index}`}
                  className={
                    selected === row ? 'clickable' : undefined
                  }
                  style={
                    selected === row
                      ? { background: 'var(--bg-hover)' }
                      : undefined
                  }
                >
                  <td>{row.timestamp_seconds.toFixed(2)}s</td>
                  <td>{row.label}</td>
                  <td>{(row.score * 100).toFixed(0)}%</td>
                  <td>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => onSelect(row)}
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <DetectionSnapshot
          jobId={jobId}
          frame={selected.frame}
          box={selected.box}
          label={selected.label}
          score={selected.score}
        />
      )}
    </div>
  );
}


function ModelComparisonPage() {
  const { token } = useAuth();

  const [jobs, setJobs] = useState([]);
  const [loadingJobs, setLoadingJobs] = useState(true);
  const [jobAId, setJobAId] = useState('');
  const [jobBId, setJobBId] = useState('');

  const [comparison, setComparison] = useState(null);
  const [jobADetail, setJobADetail] = useState(null);
  const [jobBDetail, setJobBDetail] = useState(null);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [error, setError] = useState('');

  const [selectedA, setSelectedA] = useState(null);
  const [selectedB, setSelectedB] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadJobs() {
      setLoadingJobs(true);

      try {
        const response = await authFetch(token, '/api/jobs');
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Unable to load jobs');
        }

        if (!cancelled) {
          setJobs(
            data.filter((job) => job.status === 'completed')
          );
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      } finally {
        if (!cancelled) setLoadingJobs(false);
      }
    }

    loadJobs();

    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    setSelectedA(null);
    setSelectedB(null);

    if (!jobAId || !jobBId) {
      setComparison(null);
      setJobADetail(null);
      setJobBDetail(null);
      return;
    }

    let cancelled = false;

    async function loadComparison() {
      setLoadingComparison(true);
      setError('');

      try {
        const [compareResponse, aResponse, bResponse] =
          await Promise.all([
            authFetch(
              token,
              `/api/research/compare?job_a=${encodeURIComponent(
                jobAId
              )}&job_b=${encodeURIComponent(jobBId)}`
            ),
            authFetch(token, `/api/jobs/${jobAId}`),
            authFetch(token, `/api/jobs/${jobBId}`),
          ]);

        const compareData = await compareResponse.json();

        if (!compareResponse.ok) {
          throw new Error(
            compareData.detail || 'Unable to load comparison'
          );
        }

        const aData = await aResponse.json();
        const bData = await bResponse.json();

        if (!cancelled) {
          setComparison(compareData);
          setJobADetail(aData);
          setJobBDetail(bData);
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      } finally {
        if (!cancelled) setLoadingComparison(false);
      }
    }

    loadComparison();

    return () => {
      cancelled = true;
    };
  }, [jobAId, jobBId, token]);

  const eventsA = useMemo(
    () => flattenEvents(jobADetail),
    [jobADetail]
  );

  const eventsB = useMemo(
    () => flattenEvents(jobBDetail),
    [jobBDetail]
  );

  const analyzedA = comparison?.jobs?.job_a?.metadata?.analyzed_frames;
  const analyzedB = comparison?.jobs?.job_b?.metadata?.analyzed_frames;
  const samplingDiffers =
    analyzedA != null && analyzedB != null && analyzedA !== analyzedB;

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Model Comparison</h2>

        <p className="page-subtitle">
          Compare two completed video-analysis jobs side by side,
          using their real, already-computed results. Nothing here is
          simulated.
        </p>
      </div>

      <div className="card card-padded" style={{ marginBottom: 20 }}>
        <div className="model-compare-picker-row">
          <div>
            <label className="field-label">Job A</label>

            <select
              className="text-input"
              value={jobAId}
              onChange={(event) => setJobAId(event.target.value)}
              disabled={loadingJobs}
            >
              <option value="">Select a completed job…</option>
              {jobs.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {job.filename} ({job.detector_type || 'yolo'})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="field-label">Job B</label>

            <select
              className="text-input"
              value={jobBId}
              onChange={(event) => setJobBId(event.target.value)}
              disabled={loadingJobs}
            >
              <option value="">Select a completed job…</option>
              {jobs.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {job.filename} ({job.detector_type || 'yolo'})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 20 }}>
          {error}
        </div>
      )}

      {loadingComparison && (
        <div className="empty-state">Loading comparison…</div>
      )}

      {comparison && (
        <>
          {comparison.warning && (
            <div
              className="warning-banner"
              style={{ marginBottom: 20 }}
            >
              {comparison.warning}
            </div>
          )}

          {samplingDiffers && (
            <div
              className="warning-banner"
              style={{ marginBottom: 20 }}
            >
              Job A analyzed {analyzedA} frames; Job B analyzed{' '}
              {analyzedB} frames. Because these differ, raw detection
              counts are not directly comparable — see "detections
              per analyzed frame" below instead.
            </div>
          )}

          <div className="model-compare-columns">
            <JobMetadataCard jobView={comparison.jobs.job_a} />
            <JobMetadataCard jobView={comparison.jobs.job_b} />
          </div>

          <div className="model-compare-columns">
            <JobSummaryCard jobView={comparison.jobs.job_a} />
            <JobSummaryCard jobView={comparison.jobs.job_b} />
          </div>

          <div className="card card-padded" style={{ marginBottom: 20 }}>
            <h3 style={{ marginTop: 0 }}>Timeline</h3>
            <Timeline timeline={comparison.timeline} />
          </div>

          {comparison.qualitative_notes.length > 0 && (
            <div
              className="card card-padded"
              style={{ marginBottom: 20 }}
            >
              <h3 style={{ marginTop: 0 }}>
                Known Research Findings
              </h3>

              <p className="model-compare-note">
                Curated observations from manual review of real
                detections against their source frames — not
                automatically detected.
              </p>

              <ul className="model-compare-notes-list">
                {comparison.qualitative_notes.map((note, index) => (
                  <li key={index}>
                    {note.note}
                    {note.frame_url && (
                      <>
                        {' '}
                        <a
                          href={mediaUrl(note.frame_url, token)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View frame {note.frame}
                        </a>
                      </>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="card card-padded" style={{ marginBottom: 20 }}>
            <h3 style={{ marginTop: 0 }}>Ground-Truth Evaluation</h3>
            <p className="model-compare-note">
              {comparison.ground_truth_evaluation.message}
            </p>
          </div>

          <div className="card card-padded">
            <h3 style={{ marginTop: 0 }}>Event Inspection</h3>

            <p className="model-compare-note">
              Pick a detection from each side to view its frame and
              box together. This does not claim they are the same
              physical object — only that they happened around the
              same point in the video.
            </p>

            <div className="model-compare-event-columns">
              <EventColumn
                title="Job A detections"
                rows={eventsA}
                jobId={jobAId}
                selected={selectedA}
                onSelect={setSelectedA}
              />

              <EventColumn
                title="Job B detections"
                rows={eventsB}
                jobId={jobBId}
                selected={selectedB}
                onSelect={setSelectedB}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}


export default ModelComparisonPage;
