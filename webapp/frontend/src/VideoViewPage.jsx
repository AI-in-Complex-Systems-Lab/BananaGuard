import { useCallback, useEffect, useMemo, useState } from 'react';
import CameraTile from './CameraTile';
import useCameraDevices from './useCameraDevices';


const PLACEHOLDER_CAMERAS = [
  { id: 'placeholder-1', name: 'Front Entrance', kind: 'placeholder' },
  { id: 'placeholder-2', name: 'Parking Lot', kind: 'placeholder' },
  { id: 'placeholder-3', name: 'Lobby', kind: 'placeholder' },
  { id: 'placeholder-4', name: 'Rear Exit', kind: 'placeholder' },
];

const DEFAULT_BROWSER_CAMERA = {
  id: 'browser-default',
  name: 'This Device',
  kind: 'browser',
  deviceId: undefined,
};

const LAYOUTS = [
  { key: '2x2', label: '2×2', columns: 2 },
  { key: '3x3', label: '3×3', columns: 3 },
  { key: '4x4', label: '4×4', columns: 4 },
];

function formatClock(date) {
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function VideoViewPage() {
  const { devices, permission, hasLabeledDevices, requestAccess } =
    useCameraDevices();

  const [layout, setLayout] = useState(LAYOUTS[1]);
  const [now, setNow] = useState(() => new Date());
  const [liveMap, setLiveMap] = useState({});

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const browserCameras = useMemo(() => {
    if (!hasLabeledDevices || devices.length === 0) {
      return [DEFAULT_BROWSER_CAMERA];
    }

    return devices.map((device, index) => ({
      id: `device-${device.deviceId}`,
      name: device.label || `Camera ${index + 1}`,
      kind: 'browser',
      deviceId: device.deviceId,
    }));
  }, [devices, hasLabeledDevices]);

  const cameras = useMemo(
    () => [...browserCameras, ...PLACEHOLDER_CAMERAS],
    [browserCameras]
  );

  const [focusedId, setFocusedId] = useState(cameras[0].id);

  const effectiveFocusedId = cameras.some(
    (camera) => camera.id === focusedId
  )
    ? focusedId
    : cameras[0]?.id;

  const handleLiveChange = useCallback((id, isLive) => {
    setLiveMap((previous) => {
      if (previous[id] === isLive) return previous;
      return { ...previous, [id]: isLive };
    });
  }, []);

  const liveCount = Object.values(liveMap).filter(Boolean).length;

  return (
    <div className="video-view">
      <div className="video-view-header">
        <div>
          <h2 className="page-title">Video View</h2>

          <p className="page-subtitle">
            {liveCount} of {cameras.length} cameras live &middot;{' '}
            {formatClock(now)}
          </p>
        </div>

        <div className="video-view-layout-switch">
          {LAYOUTS.map((option) => (
            <button
              key={option.key}
              type="button"
              className={`btn btn-sm ${
                layout.key === option.key ? 'btn-primary' : 'btn-ghost'
              }`}
              onClick={() => setLayout(option)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="video-view-body">
        <aside className="video-view-sidebar">
          <span className="nav-label">Cameras</span>

          <div className="camera-list">
            {cameras.map((camera) => (
              <button
                key={camera.id}
                type="button"
                className={`camera-list-item${
                  effectiveFocusedId === camera.id ? ' active' : ''
                }`}
                onClick={() => setFocusedId(camera.id)}
              >
                <span
                  className={`camera-status-dot${
                    liveMap[camera.id] ? ' live' : ''
                  }`}
                />
                {camera.name}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="btn btn-sm btn-ghost"
            style={{ marginTop: 12 }}
            onClick={requestAccess}
          >
            Detect Connected Cameras
          </button>

          <p className="video-view-sidebar-note">
            {permission === 'denied'
              ? 'Camera access was denied. Check your browser’s site permissions and try again.'
              : browserCameras.length > 1
              ? `${browserCameras.length} real cameras detected on this device. The rest are placeholders for when this platform is connected to a real camera system.`
              : 'Only "This Device" is a real, connected camera today. Plug in a USB camera and click "Detect Connected Cameras" to add it. The rest are placeholders for when this platform is connected to a real camera system.'}
          </p>
        </aside>

        <div
          className="camera-grid"
          style={{
            gridTemplateColumns: `repeat(${layout.columns}, 1fr)`,
          }}
        >
          {cameras.map((camera) => (
            <CameraTile
              key={camera.id}
              camera={camera}
              focused={effectiveFocusedId === camera.id}
              onFocus={setFocusedId}
              onLiveChange={handleLiveChange}
            />
          ))}
        </div>
      </div>
    </div>
  );
}


export default VideoViewPage;
