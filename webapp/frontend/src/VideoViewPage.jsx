import { useEffect, useState } from 'react';
import CameraTile from './CameraTile';
import useDetectionCamera from './useDetectionCamera';


const CAMERAS = [
  { id: 'browser', name: 'This Device', kind: 'browser' },
  { id: 'placeholder-1', name: 'Front Entrance', kind: 'placeholder' },
  { id: 'placeholder-2', name: 'Parking Lot', kind: 'placeholder' },
  { id: 'placeholder-3', name: 'Lobby', kind: 'placeholder' },
  { id: 'placeholder-4', name: 'Rear Exit', kind: 'placeholder' },
  { id: 'placeholder-5', name: 'Hallway North', kind: 'placeholder' },
  { id: 'placeholder-6', name: 'Loading Dock', kind: 'placeholder' },
];

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
  const liveCamera = useDetectionCamera();

  const [focusedId, setFocusedId] = useState(CAMERAS[0].id);
  const [layout, setLayout] = useState(LAYOUTS[1]);
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const liveCount = liveCamera.state.isRunning ? 1 : 0;

  return (
    <div className="video-view">
      <div className="video-view-header">
        <div>
          <h2 className="page-title">Video View</h2>

          <p className="page-subtitle">
            {liveCount} of {CAMERAS.length} cameras live &middot;{' '}
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
            {CAMERAS.map((camera) => {
              const isLive =
                camera.kind === 'browser' && liveCamera.state.isRunning;

              return (
                <button
                  key={camera.id}
                  type="button"
                  className={`camera-list-item${
                    focusedId === camera.id ? ' active' : ''
                  }`}
                  onClick={() => setFocusedId(camera.id)}
                >
                  <span
                    className={`camera-status-dot${
                      isLive ? ' live' : ''
                    }`}
                  />
                  {camera.name}
                </button>
              );
            })}
          </div>

          <p className="video-view-sidebar-note">
            Only &ldquo;This Device&rdquo; is a real, connected camera
            today. The rest are placeholders for when this platform is
            connected to a real camera system.
          </p>
        </aside>

        <div
          className="camera-grid"
          style={{
            gridTemplateColumns: `repeat(${layout.columns}, 1fr)`,
          }}
        >
          {CAMERAS.map((camera) => (
            <CameraTile
              key={camera.id}
              camera={camera}
              focused={focusedId === camera.id}
              onFocus={setFocusedId}
              liveCamera={liveCamera}
            />
          ))}
        </div>
      </div>
    </div>
  );
}


export default VideoViewPage;
