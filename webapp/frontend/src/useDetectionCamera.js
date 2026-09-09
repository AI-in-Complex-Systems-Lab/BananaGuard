import { useEffect, useRef, useState } from 'react';
import { websocketUrl } from './api';
import { useAuth } from './AuthContext';


const RECONNECT_INITIAL_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 10000;


function drawBoxes(canvas, detections) {
  if (!canvas) return;

  const context = canvas.getContext('2d');

  context.clearRect(0, 0, canvas.width, canvas.height);

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


/**
 * Captures one browser-visible camera (this device's webcam, or a
 * specific USB camera by deviceId), streams frames to the detection
 * WebSocket, and draws returned bounding boxes onto a canvas. Each
 * camera tile gets its own instance of this hook, so multiple cameras
 * plugged into the same machine can each run detection independently.
 */
function useDetectionCamera(deviceId) {
  const { token } = useAuth();

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const deviceIdRef = useRef(deviceId);
  const reconnectTimeoutRef = useRef(null);
  const reconnectDelayRef = useRef(RECONNECT_INITIAL_DELAY_MS);
  const isRunningRef = useRef(false);
  const tokenRef = useRef(token);

  const [isRunning, setIsRunning] = useState(false);
  const [connectionState, setConnectionState] = useState('idle');
  const [status, setStatus] = useState('Camera is stopped.');
  const [lastDetections, setLastDetections] = useState([]);

  tokenRef.current = token;
  deviceIdRef.current = deviceId;

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

      if (!video || video.readyState !== 4 || video.paused) {
        return;
      }

      const temporaryCanvas = document.createElement('canvas');

      temporaryCanvas.width = 640;
      temporaryCanvas.height = 480;

      const context = temporaryCanvas.getContext('2d', {
        willReadFrequently: true,
      });

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
            activeSocket?.readyState === WebSocket.OPEN &&
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
    const websocket = new WebSocket(websocketUrl(tokenRef.current));

    socketRef.current = websocket;

    websocket.onopen = () => {
      reconnectDelayRef.current = RECONNECT_INITIAL_DELAY_MS;

      setConnectionState('connected');
      setStatus('Connected. Watching for firearms...');

      startSendLoop();
    };

    websocket.onmessage = (event) => {
      try {
        const detections = JSON.parse(event.data);

        setLastDetections(detections);

        if (detections.length > 0) {
          setStatus(`Alert: ${detections.length} gun detection(s)`);
        } else {
          setStatus('Camera active — no gun detected');
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

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  async function start() {
    if (isRunningRef.current) return;

    isRunningRef.current = true;
    setIsRunning(true);
    reconnectDelayRef.current = RECONNECT_INITIAL_DELAY_MS;
    setConnectionState('connecting');
    setStatus('Requesting camera access...');

    try {
      const requestedDeviceId = deviceIdRef.current;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          ...(requestedDeviceId
            ? { deviceId: { exact: requestedDeviceId } }
            : {}),
        },
        audio: false,
      });

      if (!isRunningRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      connectSocket();
    } catch (error) {
      console.error('Unable to start camera:', error);

      setStatus(`Camera error: ${error.message}`);
      setConnectionState('error');
      isRunningRef.current = false;
      setIsRunning(false);
      cleanupResources();
    }
  }

  function stop() {
    isRunningRef.current = false;
    setIsRunning(false);
    setConnectionState('idle');
    setStatus('Camera is stopped.');
    setLastDetections([]);
    cleanupResources();
  }

  useEffect(() => {
    return () => {
      isRunningRef.current = false;
      cleanupResources();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    refs: { video: videoRef, canvas: canvasRef },
    state: { isRunning, connectionState, status, lastDetections },
    start,
    stop,
  };
}


export default useDetectionCamera;
