import { useEffect, useRef, useState } from 'react';
import { websocketUrl } from './api';
import { useAuth } from './AuthContext';


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
  const { token } = useAuth();

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);

  const [status, setStatus] = useState(
    'Initializing camera...'
  );

  useEffect(() => {
    const websocket = new WebSocket(
      websocketUrl(token)
    );

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
  }, [token]);

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
    <div>
      <div className="page-header">
        <h2 className="page-title">Live Camera</h2>

        <p className="page-subtitle">
          Real-time firearm detection from a connected
          camera feed.
        </p>
      </div>

      <div
        className="info-banner"
        style={{ marginBottom: 20 }}
      >
        <strong>Status:</strong> {status}
      </div>

      <div
        className="card"
        style={{
          padding: 16,
          width: 'fit-content',
        }}
      >
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
      </div>
    </div>
  );
}


const styles = {
  videoContainer: {
    position: 'relative',
    width: '640px',
    height: '480px',
    maxWidth: '100%',
    background: '#000000',
    borderRadius: 'var(--radius-md)',
    overflow: 'hidden',
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
};


export default WebcamPanel;
