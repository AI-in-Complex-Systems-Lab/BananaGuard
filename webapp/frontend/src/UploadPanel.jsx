import { useEffect, useState } from 'react';
import { authFetch } from './api';
import { useAuth } from './AuthContext';
import CompletedJobDetails from './CompletedJobDetails';


function UploadPanel() {
  const { token } = useAuth();

  const [selectedFile, setSelectedFile] =
    useState(null);

  const [job, setJob] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [uploading, setUploading] =
    useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!jobId) return undefined;

    let cancelled = false;

    const intervalId = setInterval(async () => {
      try {
        const response = await authFetch(
          token,
          `/api/jobs/${jobId}`
        );

        if (!response.ok) {
          throw new Error(
            'Unable to retrieve processing status'
          );
        }

        const updatedJob = await response.json();

        if (!cancelled) {
          setJob(updatedJob);
        }

        if (
          updatedJob.status === 'completed' ||
          updatedJob.status === 'failed'
        ) {
          clearInterval(intervalId);
        }
      } catch (pollingError) {
        console.error(pollingError);

        if (!cancelled) {
          setError(pollingError.message);
        }

        clearInterval(intervalId);
      }
    }, 1000);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [jobId, token]);

  async function handleUpload(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError('Please select a video first');
      return;
    }

    setUploading(true);
    setError('');
    setJob(null);
    setJobId(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await authFetch(
        token,
        '/api/videos',
        {
          method: 'POST',
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || 'Video upload failed'
        );
      }

      setJob(data);
      setJobId(data.job_id);
    } catch (uploadError) {
      console.error(uploadError);
      setError(uploadError.message);
    } finally {
      setUploading(false);
    }
  }

  function resetUpload() {
    setSelectedFile(null);
    setJob(null);
    setJobId(null);
    setError('');
  }

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">
          Process a Video
        </h2>

        <p className="page-subtitle">
          Upload footage to detect firearms, monitor
          processing progress, and generate an
          annotated result.
        </p>
      </div>

      <div className="card card-padded">
        <form
          onSubmit={handleUpload}
          className="file-drop"
        >
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
            onChange={(event) => {
              setSelectedFile(
                event.target.files?.[0] || null
              );
              setError('');
            }}
          />

          <button
            type="submit"
            disabled={!selectedFile || uploading}
            className="btn btn-primary"
          >
            {uploading
              ? 'Uploading...'
              : 'Upload and Process'}
          </button>
        </form>

        {selectedFile && !job && (
          <p style={{ marginTop: 14 }}>
            Selected: <strong>{selectedFile.name}</strong>
          </p>
        )}

        {error && (
          <div
            className="error-banner"
            style={{ marginTop: 16 }}
          >
            {error}
          </div>
        )}

        {job && (
          <div style={{ marginTop: 24 }}>
            <h3>Processing Job</h3>

            <p>
              <strong>File:</strong> {job.filename}
            </p>

            <p>
              <strong>Status:</strong>{' '}
              <span className="badge badge-info">
                {job.status}
              </span>
            </p>

            <p>
              <strong>Message:</strong> {job.message}
            </p>

            <div className="progress-track">
              <div
                className="progress-bar"
                style={{
                  width: `${job.progress || 0}%`,
                }}
              />
            </div>

            <p>{job.progress || 0}% complete</p>

            {job.status === 'completed' && (
              <CompletedJobDetails job={job} />
            )}

            {job.status === 'failed' && (
              <div className="error-banner">
                Processing failed: {job.error}
              </div>
            )}

            <button
              type="button"
              onClick={resetUpload}
              className="btn btn-ghost"
              style={{ marginTop: 18 }}
            >
              Process Another Video
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


export default UploadPanel;
