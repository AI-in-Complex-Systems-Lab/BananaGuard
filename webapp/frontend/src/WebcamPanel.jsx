import { useEffect, useRef, useState } from 'react';
import { websocketUrl } from './api';
import { useAuth } from './AuthContext';


const RECONNECT_INITIAL_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 10000;


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


function bannerClassName(connectionState) {
  if (connectionState === 'connected') return 'info-banner';
  if (connectionState === 'error') return 'error-banner';

  if (
    connectionState === 'connecting' ||
    connectionState === 'reconnecting'
  ) {
    return 'warning-banner';
  }

  return 'info-banner';
}


function WebcamPanel() {
  const { token } = useAuth();

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectDelayRef = useRef(
    RECONNECT_INITIAL_DELAY_MS
  );
  const isRunningRef = useRef(false);
  const tokenRef = useRef(token);

  const [isRunning, setIsRunning] = useState(false);
  const [connectionState, setConnectionState] =
    useState('idle');
  const [status, setStatus] = useState(
    'Camera is stopped.'
  );

  tokenRef.current = token;

  function stopSendLoop() {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
  }

  function startSendLoop() {
    stopSendLoop();

    sendIntervalRef.current = setInterval(() => {
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
        { willReadFrequently: true }
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
  }

  function connectSocket() {
    const websocket = new WebSocket(
      websocketUrl(tokenRef.current)
    );

    socketRef.current = websocket;

    websocket.onopen = () => {
      reconnectDelayRef.current =
        RECONNECT_INITIAL_DELAY_MS;

      setConnectionState('connected');
      setStatus(
        'Connected. Watching for firearms...'
      );

      startSendLoop();
    };

    websocket.onmessage = (event) => {
      try {
        const detections = JSON.parse(event.data);

        if (detections.length > 0) {
          setStatus(
            `Alert: ${detections.length} gun detection(s)`
          );
        } else {
          setStatus(
            'Camera active — no gun detected'
          );
        }

        drawBoxes(canvasRef.current, detections);
      } catch (error) {
        console.error(
          'Unable to process detection response:',
          error
        );
      }
    };

    websocket.onclose = () => {
      stopSendLoop();

      if (socketRef.current === websocket) {
        socketRef.current = null;
      }

      if (!isRunningRef.current) {
        setConnectionState('idle');
        return;
      }

      const delay = reconnectDelayRef.current;

      reconnectDelayRef.current = Math.min(
        delay * 2,
        RECONNECT_MAX_DELAY_MS
      );

      setConnectionState('reconnecting');
      setStatus(
        `Connection lost. Reconnecting in ${Math.round(
          delay / 1000
        )}s...`
      );

      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectTimeoutRef.current = null;

        if (isRunningRef.current) {
          connectSocket();
        }
      }, delay);
    };

    websocket.onerror = () => {
      websocket.close();
    };
  }

  function cleanupResources() {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    stopSendLoop();

    const socket = socketRef.current;
    socketRef.current = null;

    if (
      socket &&
      (socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING)
    ) {
      socket.close();
    }

    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  async function handleStart() {
    if (isRunningRef.current) return;

    isRunningRef.current = true;
    setIsRunning(true);
    reconnectDelayRef.current =
      RECONNECT_INITIAL_DELAY_MS;
    setConnectionState('connecting');
    setStatus('Requesting camera access...');

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });

      if (!isRunningRef.current) {
        stream.getTracks().forEach((track) =>
          track.stop()
        );

        return;
      }

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      connectSocket();
    } catch (error) {
      console.error(
        'Unable to start camera:',
        error
      );

      setStatus(`Camera error: ${error.message}`);
      setConnectionState('error');
      isRunningRef.current = false;
      setIsRunning(false);
      cleanupResources();
    }
  }

  function handleStop() {
    isRunningRef.current = false;
    setIsRunning(false);
    setConnectionState('idle');
    setStatus('Camera is stopped.');
    cleanupResources();
  }

  useEffect(() => {
    return () => {
      isRunningRef.current = false;
      cleanupResources();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        className={bannerClassName(connectionState)}
        style={{
          marginBottom: 20,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 16,
          flexWrap: 'wrap',
        }}
      >
        <span>
          <strong>Status:</strong> {status}
        </span>

        <button
          type="button"
          className={
            isRunning ? 'btn btn-danger' : 'btn btn-primary'
          }
          onClick={isRunning ? handleStop : handleStart}
        >
          {isRunning ? 'Stop Camera' : 'Start Camera'}
        </button>
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

          {!isRunning && (
            <div style={styles.placeholder}>
              Camera is stopped
            </div>
          )}
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

  placeholder: {
    position: 'absolute',
    inset: 0,
    zIndex: 5,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'var(--text-muted)',
    fontSize: 14,
  },
};


export default WebcamPanel;
