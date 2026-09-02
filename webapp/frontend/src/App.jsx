import { useEffect, useRef, useState } from 'react';
import CompletedJobDetails from './CompletedJobDetails';
import JobHistoryPanel from './JobHistoryPanel';

const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8081';

const WEBSOCKET_URL =
  import.meta.env.VITE_WS_URL || 'ws://localhost:8081/ws';


function drawBoxes(canvas, detections) {
  if (!canvas) return;

  const context = canvas.getContext('2d');

  context.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

  detections.forEach((detection) => {
    const [x, y, width, height] = detection.box;

    context.strokeStyle = '#00ff66';
    context.lineWidth = 4;
    context.strokeRect(x, y, width, height);

    context.font = '20px Arial';
    context.fillStyle = '#00ff66';
    context.fillText(
      `${detection.label} (${detection.score})`,
      x,
      Math.max(y - 10, 20)
    );
  });
}


function WebcamPanel() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);

  const [status, setStatus] = useState(
    'Initializing camera...'
  );

  useEffect(() => {
    const websocket = new WebSocket(WEBSOCKET_URL);

    socketRef.current = websocket;

    websocket.onopen = () => {
      setStatus('Connected. Waiting for camera stream...');
    };

    websocket.onclose = () => {
      setStatus('Disconnected');
    };

    websocket.onerror = () => {
      setStatus('WebSocket error');
    };

    websocket.onmessage = (event) => {
      try {
        const detections = JSON.parse(event.data);

        if (detections.length > 0) {
          setStatus(
            `Alert: ${detections.length} gun detection(s)`
          );
        } else {
          setStatus('Camera active — no gun detected');
        }

        drawBoxes(canvasRef.current, detections);
      } catch (error) {
        console.error(
          'Unable to process detection response:',
          error
        );

        setStatus('Invalid response from backend');
      }
    };

    return () => {
      if (
        websocket.readyState === WebSocket.OPEN ||
        websocket.readyState === WebSocket.CONNECTING
      ) {
        websocket.close();
      }

      socketRef.current = null;
    };
  }, []);

  useEffect(() => {
    let cameraStream;

    async function startCamera() {
      try {
        cameraStream =
          await navigator.mediaDevices.getUserMedia({
            video: {
              width: 640,
              height: 480,
            },
            audio: false,
          });

        if (videoRef.current) {
          videoRef.current.srcObject = cameraStream;
        }
      } catch (error) {
        console.error(
          'Unable to start camera:',
          error
        );

        setStatus(`Camera error: ${error.message}`);
      }
    }

    startCamera();

    return () => {
      cameraStream
        ?.getTracks()
        .forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    const intervalId = setInterval(() => {
      const websocket = socketRef.current;
      const video = videoRef.current;

      if (
        !websocket ||
        websocket.readyState !== WebSocket.OPEN ||
        websocket.bufferedAmount > 0
      ) {
        return;
      }

      if (
        !video ||
        video.readyState !== 4 ||
        video.paused
      ) {
        return;
      }

      const temporaryCanvas =
        document.createElement('canvas');

      temporaryCanvas.width = 640;
      temporaryCanvas.height = 480;

      const context = temporaryCanvas.getContext(
        '2d',
        {
          willReadFrequently: true,
        }
      );

      context.drawImage(
        video,
        0,
        0,
        temporaryCanvas.width,
        temporaryCanvas.height
      );

      temporaryCanvas.toBlob(
        (blob) => {
          const activeSocket = socketRef.current;

          if (
            blob &&
            activeSocket?.readyState ===
              WebSocket.OPEN &&
            activeSocket.bufferedAmount === 0
          ) {
            activeSocket.send(blob);
          }
        },
        'image/jpeg',
        0.5
      );
    }, 100);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <section>
      <div style={styles.statusBox}>
        <strong>Status:</strong> {status}
      </div>

      <div style={styles.videoContainer}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={styles.video}
        />

        <canvas
          ref={canvasRef}
          width="640"
          height="480"
          style={styles.canvas}
        />
      </div>
    </section>
  );
}


function UploadPanel() {
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
        const response = await fetch(
          `${API_BASE_URL}/api/jobs/${jobId}`
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
  }, [jobId]);

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
      const response = await fetch(
        `${API_BASE_URL}/api/videos`,
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
    <section style={styles.uploadPanel}>
      <h2>Process a Video</h2>

      <p>
        Upload a video to detect guns, monitor
        processing progress and generate an annotated
        result.
      </p>

      <form
        onSubmit={handleUpload}
        style={styles.uploadForm}
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
          style={styles.primaryButton}
        >
          {uploading
            ? 'Uploading...'
            : 'Upload and Process'}
        </button>
      </form>

      {selectedFile && !job && (
        <p>
          Selected: <strong>{selectedFile.name}</strong>
        </p>
      )}

      {error && (
        <div style={styles.errorBox}>
          {error}
        </div>
      )}

      {job && (
        <div style={styles.jobCard}>
          <h3>Processing Job</h3>

          <p>
            <strong>File:</strong> {job.filename}
          </p>

          <p>
            <strong>Status:</strong> {job.status}
          </p>

          <p>
            <strong>Message:</strong> {job.message}
          </p>

          <div style={styles.progressTrack}>
            <div
              style={{
                ...styles.progressBar,
                width: `${job.progress || 0}%`,
              }}
            />
          </div>

          <p>{job.progress || 0}% complete</p>

          {job.status === 'completed' && (
            <CompletedJobDetails
              job={job}
              apiBaseUrl={API_BASE_URL}
            />
          )}

          {job.status === 'failed' && (
            <div style={styles.errorBox}>
              Processing failed: {job.error}
            </div>
          )}

          <button
            type="button"
            onClick={resetUpload}
            style={styles.secondaryButton}
          >
            Process Another Video
          </button>
        </div>
      )}
    </section>
  );
}


function App() {
  const [activeView, setActiveView] =
    useState('upload');

  return (
    <main style={styles.page}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>BananaGuard</h1>

          <p style={styles.subtitle}>
            AI-assisted firearm detection platform
          </p>
        </div>
      </header>

      <nav style={styles.navigation}>
        <button
          type="button"
          onClick={() => setActiveView('upload')}
          style={
            activeView === 'upload'
              ? styles.activeTab
              : styles.tab
          }
        >
          Upload Video
        </button>

        <button
          type="button"
          onClick={() => setActiveView('webcam')}
          style={
            activeView === 'webcam'
              ? styles.activeTab
              : styles.tab
          }
        >
          Live Camera
        </button>

        <button
          type="button"
          onClick={() => setActiveView('history')}
          style={
            activeView === 'history'
              ? styles.activeTab
              : styles.tab
          }
        >
          Job History
        </button>
      </nav>

      {activeView === 'upload' && <UploadPanel />}
      {activeView === 'webcam' && <WebcamPanel />}
      {activeView === 'history' && <JobHistoryPanel />}
    </main>
  );
}


const styles = {
  page: {
    maxWidth: '1100px',
    margin: '0 auto',
    padding: '32px 24px 80px',
    fontFamily:
      'Inter, Arial, Helvetica, sans-serif',
    color: '#172033',
  },

  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px',
  },

  title: {
    margin: 0,
    fontSize: '48px',
  },

  subtitle: {
    marginTop: '8px',
    color: '#5d6678',
  },

  navigation: {
    display: 'flex',
    gap: '12px',
    marginBottom: '28px',
    borderBottom: '1px solid #d9deea',
    paddingBottom: '14px',
  },

  tab: {
    border: '1px solid #b9c1d1',
    background: '#ffffff',
    padding: '10px 18px',
    borderRadius: '8px',
    cursor: 'pointer',
  },

  activeTab: {
    border: '1px solid #175cd3',
    background: '#175cd3',
    color: '#ffffff',
    padding: '10px 18px',
    borderRadius: '8px',
    cursor: 'pointer',
  },

  statusBox: {
    padding: '14px',
    marginBottom: '20px',
    background: '#eef2f7',
    borderRadius: '8px',
  },

  videoContainer: {
    position: 'relative',
    width: '640px',
    height: '480px',
    maxWidth: '100%',
    background: '#000000',
  },

  video: {
    position: 'absolute',
    inset: 0,
    width: '640px',
    height: '480px',
    maxWidth: '100%',
  },

  canvas: {
    position: 'absolute',
    inset: 0,
    zIndex: 10,
    width: '640px',
    height: '480px',
    maxWidth: '100%',
  },

  uploadPanel: {
    maxWidth: '800px',
  },

  uploadForm: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '16px',
    alignItems: 'center',
    padding: '22px',
    marginTop: '22px',
    background: '#f5f7fb',
    border: '1px solid #d9deea',
    borderRadius: '10px',
  },

  primaryButton: {
    border: 0,
    background: '#175cd3',
    color: '#ffffff',
    padding: '12px 20px',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: 600,
  },

  secondaryButton: {
    border: '1px solid #175cd3',
    background: '#ffffff',
    color: '#175cd3',
    padding: '10px 16px',
    borderRadius: '8px',
    cursor: 'pointer',
    marginTop: '18px',
  },

  errorBox: {
    marginTop: '18px',
    padding: '14px',
    color: '#9b1c1c',
    background: '#feecec',
    borderRadius: '8px',
  },

  jobCard: {
    marginTop: '24px',
    padding: '22px',
    border: '1px solid #d9deea',
    borderRadius: '10px',
  },

  progressTrack: {
    width: '100%',
    height: '14px',
    overflow: 'hidden',
    background: '#e1e6ef',
    borderRadius: '999px',
  },

  progressBar: {
    height: '100%',
    background: '#175cd3',
    transition: 'width 0.3s ease',
  },

};


export default App;